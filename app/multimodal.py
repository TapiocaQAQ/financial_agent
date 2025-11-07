# app/multimodal.py
from __future__ import annotations
import os, io, json, hashlib, uuid, time, shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import fitz  # PyMuPDF
import pdfplumber
from rapidocr_onnxruntime import RapidOCR

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 單例 OCR（避免每次載入）
_OCR = RapidOCR()

def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def _normalize_chunk(text: str) -> str:
    return (text or "").strip()

def _mk_id(prefix="doc") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

@dataclass
class ExtractedDoc:
    text: str
    metadata: Dict[str, Any]

# ---------- 圖片 OCR ----------
def ocr_image(bin_data: bytes) -> str:
    """用 RapidOCR 辨識圖片成純文字。"""
    # RapidOCR 支援 bytes 直接傳入
    result, _ = _OCR(bin_data)
    if not result:
        return ""
    # result: List[(text, conf, [[x1,y1]...])]
    lines = [r[0] for r in result]
    return "\n".join(lines)

# ---------- PDF 抽取 ----------
def extract_pdf(bin_data: bytes, filename: str) -> List[ExtractedDoc]:
    docs: List[ExtractedDoc] = []
    # 1) pdfplumber：文字 + 簡單表格
    with pdfplumber.open(io.BytesIO(bin_data)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            # 表格
            tables = page.extract_tables(table_settings={
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
            }) or []
            table_md_parts = []
            for t in tables:
                if not t: 
                    continue
                # t 是 2D list；轉 md
                # 若第一列當表頭
                header = [c or "" for c in t[0]]
                rows = t[1:] if len(t) > 1 else []
                md = []
                md.append("| " + " | ".join(h or "" for h in header) + " |")
                md.append("| " + " | ".join("---" for _ in header) + " |")
                for r in rows:
                    md.append("| " + " | ".join((c or "") for c in r) + " |")
                table_md_parts.append("\n".join(md))
            table_md = "\n\n".join(table_md_parts)

            content = ""
            if txt:
                content += txt.strip()
            if table_md:
                if content: content += "\n\n"
                content += "## 表格\n" + table_md

            # 2) OCR 補漏字（若文字太少，或頁面本身是掃描件）
            need_ocr = (len(content) < 50)
            if need_ocr:
                # 轉頁圖 → OCR
                with fitz.open(stream=bin_data, filetype="pdf") as doc:
                    page_img = doc.load_page(idx - 1).get_pixmap(dpi=200)  # 200 DPI 足夠
                    img_bytes = page_img.tobytes("png")
                ocr_txt = ocr_image(img_bytes)
                if ocr_txt:
                    if content: content += "\n\n"
                    content += ocr_txt

            if content.strip():
                docs.append(ExtractedDoc(
                    text=_normalize_chunk(content),
                    metadata={
                        "source": filename,
                        "page": idx,
                        "type": "pdf",
                        "has_table": bool(table_md),
                    }
                ))
    return docs

# ---------- 圖片抽取 ----------
def extract_image(bin_data: bytes, filename: str) -> List[ExtractedDoc]:
    text = ocr_image(bin_data)
    if not text.strip():
        return []
    return [ExtractedDoc(
        text=_normalize_chunk(text),
        metadata={"source": filename, "type": "image"}
    )]

# ---------- 上傳檔案保存 ----------
def save_upload(filename: str, bin_data: bytes) -> str:
    base = os.path.basename(filename)
    root, ext = os.path.splitext(base)
    digest = _md5(bin_data)[:10]
    safe = f"{root}_{digest}{ext.lower()}"
    path = os.path.join(UPLOAD_DIR, safe)
    with open(path, "wb") as f:
        f.write(bin_data)
    return path

# ---------- 將抽取結果轉為入庫 payload ----------
def build_documents(extracted: List[ExtractedDoc]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    ids, texts, metas = [], [], []
    for i, d in enumerate(extracted):
        ids.append(_mk_id("mm"))
        texts.append(d.text)
        metas.append(d.metadata)
    return ids, texts, metas
