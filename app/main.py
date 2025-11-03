from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import os, time, json
import ollama
from httpx import ConnectError  

from app.ingest import ingest_data
from app.graph import run_once

import chromadb

# USE_STREAM_BACKEND = os.getenv("STREAM_BACKEND", "NONE").upper()  # "NONE" 或 "OLLAMA"
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")         
USE_STREAM_BACKEND = "OLLAMA"  # "NONE" 或 "OLLAMA"
OLLAMA_MODEL = "llama3.2:3b"



app = FastAPI()

# 👇 加入這段（開發階段先全開；之後可改成白名單）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 或改成 ["http://localhost:8001", "http://127.0.0.1:8001"]
    allow_credentials=True,
    allow_methods=["*"],        # 讓 OPTIONS / POST 都通過
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ingest")
def ingest():
    ingest_data()
    return {"ok": True}

# -------- Streaming SSE ----------
def gen_none_backend(q: str):
    """不使用任何模型，直接把 run_once 的 answer 切 token 做假流。"""
    out = run_once(q, history=[])
    text = out.get("answer") or "(無回答)"
    yield "data: [thinking] 正在檢索知識庫與執行工具...\n\n"
    for tok in text.split(" "):
        yield f"data: {tok}\n\n"
        time.sleep(0.02)
    yield "data: [meta] " + json.dumps(out, ensure_ascii=False) + "\n\n"
    yield "event: end\ndata: [DONE]\n\n"

def build_prompt_from_rag(q: str):
    """把 RAG 證據與工具結果組成提示詞，提供給 Ollama。"""
    out = run_once(q, history=[])
    ctx = "\n\n".join([f"[{c['source']}]\n{c['text']}" for c in out['contexts']]) or "(無檢索命中)"
    tools = json.dumps(out['tool_results'], ensure_ascii=False)
    prompt = f"""你是加密交易所客服助理，請用簡潔中文回答。
                使用規則：
                - 先給出直接答案（若有百分比，請同時提供 0.090% 與 0.0009 這種兩種形式）
                - 如用到工具或知識庫，結尾列出來源檔名（不需要段落）
                - 不確定就說無法確定，不要胡編

                問題：{q}
                工具結果：{tools}
                可用資料片段：
                {ctx}
                """
    return prompt, out  # 回傳 out 便於 meta 顯示

def gen_ollama_backend(q: str, model: str):
    import ollama, os
    from httpx import ConnectError
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"

    host = os.getenv("OLLAMA_HOST")

    client = ollama.Client(host=host)


    prompt, meta_out = build_prompt_from_rag(q)
    yield f"data: [thinking] 連線到 Ollama（{host}）並產生回答...\n\n"
    try:
        for chunk in client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        ):
            msg = chunk.get("message", {})
            if isinstance(msg, dict) and msg.get("content"):
                yield "data: " + msg["content"] + "\n\n"
    except ConnectError:
        yield "data: [error] 無法連線到 Ollama，請確認已啟動 `ollama serve`，且 OLLAMA_HOST 指向 http://127.0.0.1:11434。\n\n"
    yield "event: end\ndata: [DONE]\n\n"

@app.get("/stream")
def stream(q: str):
    if USE_STREAM_BACKEND == "OLLAMA":
        return StreamingResponse(gen_ollama_backend(q, OLLAMA_MODEL), media_type="text/event-stream")
    else:
        return StreamingResponse(gen_none_backend(q), media_type="text/event-stream")



@app.post("/chat")
def chat(payload: dict = Body(...)):
    q = payload.get("q", "")
    history = payload.get("history", [])
    out = run_once(q, history)
    return out



@app.get("/health")
def health():
    client = chromadb.PersistentClient(path="./index/chroma")
    db = client.get_collection("kb_main")
    return {
        "collection": "kb_main",
        "count": db.count(),
        "stream_backend": USE_STREAM_BACKEND,
        "ollama_model": OLLAMA_MODEL if USE_STREAM_BACKEND == "OLLAMA" else None
    }
