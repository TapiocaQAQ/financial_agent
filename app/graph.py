# app/graph.py
from __future__ import annotations
from typing import Dict, List, Tuple
import os, re

import chromadb

from app.tools import bus, ToolContext
from app.embeddings import EMB

# ========= 基礎：連到 Chroma =========
def _get_db():
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./index/chroma"))
    name   = os.getenv("CHROMA_COLLECTION", "kb_main")
    try:
        col = client.get_collection(name, embedding_function=EMB)    # ✅
    except Exception:
        col = client.create_collection(name, embedding_function=EMB) # ✅
    return col

def retrieve(query: str, k: int = 3) -> List[Dict]:
    db = _get_db()
    rs = db.query(query_texts=[query], n_results=k)
    docs   = rs.get("documents", [[]])[0]
    metas  = rs.get("metadatas", [[]])[0]
    out = []
    for i, d in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        out.append({"text": d, "source": meta.get("source") or meta.get("file") or "unknown"})
    return out

def _summarize_tool_results(tool_results: dict) -> str:
    """
    把工具執行的重點輸出壓成一段可讀的摘要文字，給 LLM 當作『可用中間量』。
    """
    if not tool_results:
        return "(無工具結果)"

    lines = []
    # 常見關鍵：手續費/報價/換匯/報價計算...
    fee = tool_results.get("fee.lookup") or {}
    if fee:
        vip = fee.get("vip"); market = fee.get("market"); side = fee.get("side"); fr = fee.get("fee")
        if fr is not None:
            lines.append(f"[手續費] VIP{vip} {market}/{side}: {fr:.4%}（= {fr}）")

    mkt = tool_results.get("market.price") or {}
    if mkt:
        sym = mkt.get("symbol"); px = mkt.get("price")
        if px is not None:
            lines.append(f"[行情] {sym}: {px}")

    tq = tool_results.get("trade.quote") or {}
    if tq:
        lines.append(f"[報價] 單價 {tq.get('price')}, 數量 {tq.get('qty')}, 手續費比率 {float(tq.get('fee_ratio',0)):.4%} → 淨額 {tq.get('net')} USDT")

    fx = tool_results.get("fx.rate") or {}
    if fx:
        lines.append(f"[匯率] {fx.get('base')}/{fx.get('quote')}: {fx.get('rate')}")

    cf = tool_results.get("convert.fiat") or {}
    if cf:
        lines.append(f"[折算] TWD 約 {cf.get('amount_twd')}")

    # 如果你還有其它工具輸出，可在此繼續追加格式化
    known = {"fee.lookup", "market.price", "trade.quote", "fx.rate", "convert.fiat"}
    for k, v in tool_results.items():
        if k in known:
            continue
        try:
            if isinstance(v, dict):
                lines.append(f"[{k}] { {kk: vv for kk, vv in v.items() if vv is not None} }")
            else:
                lines.append(f"[{k}] {v}")
        except Exception:
            pass

    return "\n".join(lines) if lines else "(無工具結果)"

# ========= 舊工具（規則）— 可保留相容 =========
FEE_TABLE_DEFAULT = {
    ("spot","maker",1): 0.0010, ("spot","taker",1): 0.0010,
    ("spot","maker",2): 0.0008, ("spot","taker",2): 0.0009,
    ("spot","maker",4): 0.0006, ("spot","taker",4): 0.0007,
    ("futures","maker",2): 0.00012, ("futures","taker",2): 0.00032,
}

def _parse_vip_fee_rule(q: str) -> Dict:
    """從問題裡抓 vip/maker|taker/spot|futures 的簡單規則"""
    ql = q.lower()
    vip = None
    m = re.search(r"vip\s*([0-9]+)", ql)
    if m: vip = int(m.group(1))
    side = "taker" if "taker" in ql else ("maker" if "maker" in ql else None)
    market = "futures" if ("合約" in q or "永續" in ql or "perp" in ql) else "spot"
    return {"vip": vip, "side": side, "market": market}

def legacy_tool_results(question: str) -> Dict[str, float]:
    """舊：直接回傳單一費率數值（讓你現有 prompt 不破）"""
    info = _parse_vip_fee_rule(question)
    vip, side, market = info["vip"], info["side"], info["market"]
    if vip and side:
        fee = FEE_TABLE_DEFAULT.get((market, side, vip))
        if fee is not None:
            # 和你原來 key 的風格類似（示例）
            key = f"vip{vip}_{market}_{side}_fee"
            return {key: fee}
    return {}

# ========= 新：ToolBus 規劃 + 互用 =========
def simple_planner(question: str, ctx: ToolContext) -> List[Tuple[str, Dict]]:
    """非常簡單的規則規劃器：看關鍵字排一條 pipeline"""
    q = question.lower()
    plan: List[Tuple[str, Dict]] = []

    # 例：詢問 VIPx maker/taker 成本、可能附帶台幣
    info = _parse_vip_fee_rule(question)
    if info["vip"] and info["side"]:
        vip = info["vip"]
        side = info["side"]
        market = info["market"]
        qty = 0.5 if re.search(r"\b0\.5\b", q) else 1.0

        plan += [
            ("market.price", {"symbol": "BTCUSDT"}),  # 先拿行情
            ("fee.lookup",   {"vip": vip, "market": market, "side": side}),  # 查費率
            ("trade.quote",  {"price": "${market.price.price}", "qty": qty, "fee": "${fee.lookup.fee}"}),  # 互用
        ]
        # 若包含台幣
        if ("twd" in q) or ("台幣" in q):
            plan += [
                ("fx.rate", {"base": "USD", "quote": "TWD"}),
                ("convert.fiat", {"amount_usd": "${trade.quote.net}", "rate": "${fx.rate.rate}"}),
            ]
    return plan

def render_answer(q: str, contexts: List[Dict], tool_results: Dict, tctx: ToolContext) -> str:
    """把工具/檢索合併成一段自然語言（非必須，給 /chat 用，串流時你已有 prompt）"""
    parts = []

    # 舊的費率直答（保相容）
    for k, v in tool_results.items():
        if k.endswith("_fee"):
            parts.append(f"{k.replace('_',' ')} = {v*100:.3f}%（亦即 {v:.4%}）。")

    # 新的匯流排結果
    tq = tctx.results.get("trade.quote")
    if tq and tq.ok:
        parts.append(
            f"估算成本：單價 {tq.data['price']}, 數量 {tq.data['qty']}, "
            f"手續費 {tq.data['fee_ratio']*100:.3f}% → 淨支出 {tq.data['net']:.2f} USDT。"
        )
    cf = tctx.results.get("convert.fiat")
    if cf and cf.ok:
        parts.append(f"折合新臺幣：約 {cf.data['amount_twd']:,} TWD。")

    # 檢索來源
    if contexts:
        uniq = sorted({c["source"] for c in contexts})
        parts.append("資料來源：" + "、".join(uniq))
    else:
        parts.append("（未命中知識庫段落）")

    # 工具摘要（除錯用）
    if tctx.results:
        parts.append("\n[工具摘要]\n" + tctx.summarize())

    return "\n".join(parts)

# ========= 對外主函式 =========
def run_once(question: str, history: List[Dict]) -> Dict:
    # 1) RAG 檢索
    contexts = retrieve(question, k=3)

    # 2) 舊工具：費率直答（維持你原本 prompt 行為）
    tool_results = legacy_tool_results(question)

    # 3) 新工具匯流排：讓多工具互相吃資料
    tctx = ToolContext()
    # 可把全域表塞進去（覆蓋預設表）
    tctx.put("fee_table", FEE_TABLE_DEFAULT)

    plan = simple_planner(question, tctx)
    if plan:
        tctx = bus.run(tctx, plan)

    # 把 ToolBus 的每個結果展開到 tool_results（方便前端/提示詞直接用）
    for name, tr in tctx.results.items():
        if tr.ok:
            tool_results[name] = tr.data  # 注意：這裡是一個 dict，和舊 fee_*_fee 的 float 並存

    # 4) 非串流模式下，產一段可讀回答
    answer = render_answer(question, contexts, tool_results, tctx)

    tool_summary = _summarize_tool_results(tool_results or {})
    return {
        "query": question,
        "history": history,
        "contexts": contexts,
        "tool_results": tool_results,
        "tool_summary": tool_summary,   # ⬅ 新增這行
        "answer": answer,
    }
