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
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"



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
    import time, json
    # 先回報：已接到請求
    yield "data: [debug] stream start\n\n"

    # 1) 開始 RAG
    yield "data: [debug] run_once:start\n\n"
    t0 = time.time()
    out = run_once(q, history=[])
    yield f"data: [debug] run_once:done ({time.time()-t0:.2f}s)\n\n"

    # 2) 推答案（假流）
    text = out.get("answer") or "(無回答)"
    yield "data: [thinking] 正在輸出...\n\n"
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

# def gen_ollama_backend(q: str, model: str):
#     import ollama, os, json, time
#     from httpx import ConnectError
#     os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
#     os.environ["NO_PROXY"] = "127.0.0.1,localhost"

#     host = os.getenv("OLLAMA_HOST")

#     client = ollama.Client(host=host)


#     prompt, meta_out = build_prompt_from_rag(q)
#     yield f"data: [thinking] 連線到 Ollama（{host}）並產生回答...\n\n"
#     try:
#         for chunk in client.chat(
#             model=model,
#             messages=[{"role": "user", "content": prompt}],
#             stream=True
#         ):
#             msg = chunk.get("message", {})
#             if isinstance(msg, dict) and msg.get("content"):
#                 yield "data: " + msg["content"] + "\n\n"
#     except ConnectError:
#         yield "data: [error] 無法連線到 Ollama，請確認已啟動 `ollama serve`，且 OLLAMA_HOST 指向 http://127.0.0.1:11434。\n\n"
#     yield "event: end\ndata: [DONE]\n\n"

def gen_ollama_backend(q: str, model: str):
    import ollama, os, json, time
    from httpx import ConnectError, ReadTimeout

    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    client = ollama.Client(host=host, timeout=120.0)

    # === 1) RAG 前處理 ===
    yield f"data: [debug] stream start (ollama={model} host={host})\n\n"
    yield "data: [debug] run_once:start\n\n"
    t0 = time.time()
    out = run_once(q, history=[])
    yield f"data: [debug] run_once:done ({time.time()-t0:.2f}s)\n\n"

    # 組 prompt
    yield "data: [debug] prompt:build\n\n"
    ctx = "\n\n".join([f"[{c['source']}]\n{c['text']}" for c in out['contexts']]) or "(無檢索命中)"
    tools = json.dumps(out['tool_results'], ensure_ascii=False)
    prompt = f"""你是加密交易所客服助理，請用簡潔中文回答。
若有百分比，提供 0.090% 與 0.0009 兩種表示；結尾列出來源檔名。
問題：{q}
工具結果：{tools}
可用資料片段：
{ctx}
"""

    # === 2) 先嘗試真正串流 ===
    yield "data: [debug] ollama:generate:start\n\n"
    got_any = False
    last_ts = time.time()

    try:
        for chunk in client.generate(model=model, prompt=prompt, stream=True, options={"temperature": 0.2}):
            text = ""
            if isinstance(chunk, dict):
                # generate(stream=True) 的鍵
                if chunk.get("response"):
                    text = chunk["response"]
                # 兼容 chat(stream=True) 的鍵（以防你改回 chat）
                elif isinstance(chunk.get("message"), dict):
                    text = chunk["message"].get("content", "")

            if text:
                if not got_any:
                    yield "data: [thinking] 模型已開始回覆...\n\n"
                got_any = True
                yield "data: " + text + "\n\n"
                last_ts = time.time()

            # 若 3 秒沒 token，視為串流異常 → 觸發 fallback
            if time.time() - last_ts > 3.0 and not got_any:
                yield "data: [warn] 3s 無串流輸出，啟用 fallback（非串流生成再分段輸出）\n\n"
                raise TimeoutError("no_stream_tokens")

    except (TimeoutError, ReadTimeout):
        # === 3) Fallback：非串流生成，手動切片吐出 ===
        try:
            r = client.generate(model=model, prompt=prompt, stream=False, options={"temperature": 0.2})
            text = r.get("response", "") or ""
            if not text:
                yield "data: [error] fallback 也沒有內容\n\n"
            else:
                yield "data: [thinking]（fallback）\n\n"
                # 你可以改成逐字或逐句；這裡先用空白切
                for tok in text.split(" "):
                    yield "data: " + tok + "\n\n"
                    time.sleep(0.01)
        except Exception as e:
            yield "data: [error] fallback 失敗：" + f"{type(e).__name__}: {e}" + "\n\n"

    except ConnectError:
        yield "data: [error] 無法連線到 Ollama；請確認已啟動 `ollama serve` 並設定 OLLAMA_HOST。\n\n"
    except Exception as e:
        yield "data: [error] " + f"{type(e).__name__}: {e}" + "\n\n"

    # === 4) 附上 meta 便於除錯 ===
    yield "data: [meta] " + json.dumps(out, ensure_ascii=False) + "\n\n"
    yield "event: end\ndata: [DONE]\n\n"



@app.get("/diag/ollama_gen")
def diag_ollama_gen():
    import os, ollama
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    client = ollama.Client(host=host)
    r = client.generate(model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"), prompt="hi", stream=False)
    return {"host": host, "len": len(r.get("response","")), "preview": r.get("response","")[:50]}



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
