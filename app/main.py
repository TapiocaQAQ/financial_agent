# app/main.py
from fastapi import FastAPI, Body, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import os, time, json, re, pathlib
import httpx
from httpx import ReadTimeout, ConnectError, HTTPStatusError

# 可選 Redis（有設 REDIS_URL 才用）
try:
    import redis  # pip install redis
except Exception:
    redis = None

from app.ingest import ingest_data
from app.graph import run_once
import chromadb

# ================== 環境變數 ==================
USE_STREAM_BACKEND = os.getenv("STREAM_BACKEND", "OLLAMA").upper()  # "NONE" or "OLLAMA"
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "gpt-oss")  #llama3.2:3b
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

SESS_DIR     = os.getenv("SESSION_DIR", "./sessions")
REDIS_URL    = os.getenv("REDIS_URL", "").strip()
MAX_TURNS    = int(os.getenv("MAX_TURNS", "12"))     # 最多保留最近 12 則訊息（user/assistant 各算1）
MAX_CHARS    = int(os.getenv("MAX_CHARS", "8000"))   # 最多保留 8k 字的歷史
TRIM_STRICT  = os.getenv("TRIM_STRICT", "1") == "1"  # 嚴格修剪模式

# ================== Session Store ==================
class SessionStore:
    """
    兩種後端：
    - Redis（有 REDIS_URL 時）
    - 檔案 JSON（預設）
    格式：list[{"role":"user"|"assistant","content":"..."}]
    """
    def __init__(self, redis_url: str | None, dirpath: str):
        self.dir = pathlib.Path(dirpath)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.r = None
        if redis_url and redis:
            try:
                self.r = redis.from_url(redis_url, decode_responses=True)
                self.r.ping()
            except Exception:
                self.r = None  # 失敗就退回檔案
        # 清理正則
        self.re_debug = re.compile(r"\[(debug|thinking|warn|error|meta)\][^\n]*", re.IGNORECASE)
        self.re_done  = re.compile(r"\[DONE\]", re.IGNORECASE)
        self.re_ws    = re.compile(r"[ \t]+")

    def _sanitize_text(self, s: str) -> str:
        # 去除 SSE 產生的雜訊、重複標籤
        s = self.re_debug.sub("", s)
        s = self.re_done.sub("", s)
        # 常見 streaming 重複的方括號提示已移除；刪多餘空白
        s = s.replace("\r", "")
        # 把一堆空行收斂
        lines = [ln.strip() for ln in s.split("\n")]
        s = "\n".join([ln for ln in lines if ln != ""])
        s = self.re_ws.sub(" ", s).strip()
        return s

    def _file_path(self, sid: str) -> pathlib.Path:
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", sid)[:80]
        return self.dir / f"{safe}.jsonl"

    def get(self, sid: str) -> list:
        if self.r:
            raw = self.r.get(f"sess:{sid}")
            if raw:
                try:
                    return json.loads(raw)
                except Exception:
                    return []
            return []
        fp = self._file_path(sid)
        if not fp.exists():
            return []
        try:
            # 用 JSONL 也行，但這裡簡單整包 JSON
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return []

    def set(self, sid: str, hist: list) -> None:
        # 修剪長度 & 字數
        hist = self._trim(hist)
        if self.r:
            self.r.set(f"sess:{sid}", json.dumps(hist, ensure_ascii=False))
            return
        fp = self._file_path(sid)
        fp.write_text(json.dumps(hist, ensure_ascii=False), encoding="utf-8")

    def append(self, sid: str, role: str, content: str) -> list:
        hist = self.get(sid)
        content = self._sanitize_text(content or "")
        if not content:
            return hist
        hist.append({"role": role, "content": content})
        hist = self._trim(hist)
        self.set(sid, hist)
        return hist

    def clear(self, sid: str):
        if self.r:
            self.r.delete(f"sess:{sid}")
            return
        fp = self._file_path(sid)
        if fp.exists():
            fp.unlink()

    def _trim(self, hist: list) -> list:
        # 1) 只保留需要的 key
        cleaned = []
        for m in hist:
            role = m.get("role", "").strip()
            content = self._sanitize_text(m.get("content", ""))
            if role not in ("user", "assistant"):
                continue
            if not content:
                continue
            cleaned.append({"role": role, "content": content})

        # 2) 限回合數
        if len(cleaned) > MAX_TURNS:
            cleaned = cleaned[-MAX_TURNS:]

        # 3) 限總字數（從最舊開始丟）
        if TRIM_STRICT:
            total = sum(len(m["content"]) for m in cleaned)
            while total > MAX_CHARS and len(cleaned) > 2:
                total -= len(cleaned[0]["content"])
                cleaned = cleaned[1:]

        # 4) 簡單去重：連續兩則 assistant 一樣就刪前一則
        dedup = []
        for m in cleaned:
            if dedup and m["role"] == "assistant" and dedup[-1]["role"] == "assistant":
                if m["content"] == dedup[-1]["content"]:
                    continue
            dedup.append(m)

        return dedup

    def analyze(self, sid: str) -> dict:
        hist = self.get(sid)
        issues = []
        # 檢查是否仍殘留 debug/DONE
        joined = "\n".join(m.get("content","") for m in hist)
        if self.re_debug.search(joined) or self.re_done.search(joined):
            issues.append("history 含有未清理的 [debug]/[DONE] 片段")
        # 檢查是否有「不要告訴我手續費」後仍連續回答費率
        said_no_fee = any(("不要告訴我手續費" in m.get("content","")) for m in hist if m.get("role")=="user")
        if said_no_fee:
            fee_keywords = ("手續費", "maker", "taker", "%", "費率")
            after_idx = max(i for i,m in enumerate(hist) if m["role"]=="user" and "不要告訴我手續費" in m["content"])
            viol = [m for m in hist[after_idx+1:] if m["role"]=="assistant" and any(k in m["content"] for k in fee_keywords)]
            if viol:
                issues.append(f"使用者要求不要提費率，但後續 {len(viol)} 則回覆仍提到費用/百分比")
        # 回合/字數
        total_chars = sum(len(m["content"]) for m in hist)
        return {
            "turns": len(hist),
            "chars": total_chars,
            "over_turns": len(hist) > MAX_TURNS,
            "over_chars": total_chars > MAX_CHARS,
            "issues": issues,
            "history": hist,
        }

STORE = SessionStore(REDIS_URL if redis else None, SESS_DIR)

# ================== FastAPI & CORS ==================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發先全開；上線請改白名單
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== 基本端點 ==================
@app.get("/")
async def root():
    return {"message": "RAG Agent is running", "backend": USE_STREAM_BACKEND, "model": OLLAMA_MODEL}

@app.post("/ingest")
def ingest():
    ingest_data()
    return {"ok": True}

# 非串流（一次性）── 前端只給 q 與 session_id
@app.post("/chat")
def chat(payload: dict = Body(...)):
    q = payload.get("q", "")
    sid = payload.get("session_id", "default")
    # 取後端歷史（已清理/修剪）
    history = STORE.get(sid)
    out = run_once(q, history)
    # 存歷史
    STORE.append(sid, "user", q)
    STORE.append(sid, "assistant", out.get("answer") or "")
    return out

# ================== Streaming 實作 ==================
def _resolve_ollama_host():
    raw = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if raw in {"0.0.0.0", "http://0.0.0.0", "https://0.0.0.0", ""}:
        raw = "http://127.0.0.1:11434"
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    return raw.rstrip("/")

def gen_none_backend(q: str, history, sid: str):
    yield "data: [debug] stream start\n\n"
    yield "data: [debug] run_once:start\n\n"
    t0 = time.time()
    out = run_once(q, history=history or [])
    yield f"data: [debug] run_once:done ({time.time()-t0:.2f}s)\n\n"
    text = out.get("answer") or "(無回答)"
    yield "data: [thinking] 正在輸出...\n\n"
    acc = []
    for tok in text.split(" "):
        yield f"data: {tok}\n\n"
        acc.append(tok)
        time.sleep(0.02)
    final = " ".join(acc).strip()
    # 寫回歷史
    STORE.append(sid, "user", q)
    STORE.append(sid, "assistant", final)
    yield "data: [meta] " + json.dumps(out, ensure_ascii=False) + "\n\n"
    yield "event: end\ndata: [DONE]\n\n"

def gen_ollama_backend(q: str, history, model: str, sid: str):
    host = _resolve_ollama_host()
    chat_url = f"{host}/api/chat"
    gen_url  = f"{host}/api/generate"

    # 1) RAG
    yield f"data: [debug] stream start (ollama={model} host={host})\n\n"
    yield "data: [debug] run_once:start\n\n"
    t0 = time.time()
    out = run_once(q, history=history or [])
    yield f"data: [debug] run_once:done ({time.time()-t0:.2f}s)\n\n"

    # 2) 準備 messages（system + history + 當前問題 + RAG）
    ctx = "\n\n".join([f"[{c['source']}]\n{c['text']}" for c in out['contexts']]) or "(無檢索命中)"
    tools = json.dumps(out['tool_results'], ensure_ascii=False)
    system_prompt = (
        "你是加密交易所客服助理，請用簡潔中文回答；"
        "若有百分比，提供 0.090% 與 0.0009 兩種表示；"
        "必要時結尾列出資料來源檔名。"
    )

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    user_content = f"問題：{q}\n\n工具結果：{tools}\n可用資料片段：\n{ctx}"
    messages.append({"role": "user", "content": user_content})

    def build_generate_prompt():
        convo = []
        for m in history or []:
            role = "使用者" if m.get("role") == "user" else "助理"
            convo.append(f"{role}：{m.get('content','')}")
        convo_text = "\n".join(convo[-MAX_TURNS:])
        return (
            f"{system_prompt}\n\n"
            f"【前文對話】\n{convo_text}\n\n"
            f"【目前問題】\n{q}\n\n"
            f"【工具結果】\n{tools}\n\n"
            f"【可用資料片段】\n{ctx}\n"
        )

    limits = httpx.Limits(max_keepalive_connections=1, max_connections=2)
    timeout = httpx.Timeout(connect=5.0, read=120.0, write=120.0, pool=5.0)

    def stream_chat():
        payload = {"model": model, "messages": messages, "stream": True, "options": {"temperature": 0.2}}
        with httpx.Client(timeout=timeout, limits=limits, proxy=None, trust_env=False) as client:
            yield "data: [debug] ollama:httpx:chat:start\n\n"
            got_any = False
            acc = []
            with client.stream("POST", chat_url, json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    msg = obj.get("message", {})
                    if isinstance(msg, dict) and msg.get("content"):
                        if not got_any:
                            yield "data: [thinking] 模型已開始回覆...\n\n"
                        got_any = True
                        txt = msg["content"]
                        acc.append(txt)
                        yield "data: " + txt + "\n\n"
                    if obj.get("done"):
                        break
            return got_any, "".join(acc).strip()

    def stream_generate():
        prompt = build_generate_prompt()
        payload = {"model": model, "prompt": prompt, "stream": True, "options": {"temperature": 0.2}}
        with httpx.Client(timeout=timeout, limits=limits, proxy=None, trust_env=False) as client:
            yield "data: [debug] ollama:httpx:generate:start\n\n"
            got_any = False
            acc = []
            with client.stream("POST", gen_url, json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("response"):
                        if not got_any:
                            yield "data: [thinking] 模型已開始回覆...(generate)\n\n"
                        got_any = True
                        txt = obj["response"]
                        acc.append(txt)
                        yield "data: " + txt + "\n\n"
                    if obj.get("done"):
                        break
            return got_any, "".join(acc).strip()

    # 4) 先試 chat；失敗即 fallback generate
    final_text = ""
    try:
        for y in stream_chat():
            if isinstance(y, str):
                yield y
            else:
                got, text = y  # 不會走到這裡，因為上面 yield 了；留作 safety
    except (ConnectError, ReadTimeout, HTTPStatusError, json.JSONDecodeError) as e:
        yield f"data: [warn] chat 串流失敗（{type(e).__name__}），改用 generate。\n\n"
        got, text = False, ""
        try:
            for y in stream_generate():
                if isinstance(y, str):
                    yield y
                else:
                    got, text = y
        except Exception as ee:
            yield "data: [error] generate 也失敗：" + f"{type(ee).__name__}: {ee}\n\n"
    except Exception as e:
        yield "data: [error] chat 未預期錯誤：" + f"{type(e).__name__}: {e}\n\n"
    else:
        # 正常情況下，我們需要把最後累積的文字寫回歷史
        pass

    # 5) 把問題與最終回答寫回歷史（若 final_text 還是空，嘗試從 out.answer 帶）
    STORE.append(sid, "user", q)
    # final_text 可能在上面無法直接取到；為保險，在前面已逐步累積 acc並串回 text。
    # 這裡嘗試從前端看不到的情況下再補上 out.answer
    if not final_text:
        # 盡力用 generate prompt 的情境下已累積；如無，就用 RAG 的自然語言答案
        final_text = (out.get("answer") or "").strip()
    STORE.append(sid, "assistant", final_text or "(無回答)")

    yield "data: [meta] " + json.dumps(out, ensure_ascii=False) + "\n\n"
    yield "event: end\ndata: [DONE]\n\n"

@app.post("/stream")
async def stream(request: Request):
    """
    Body:
    {
      "q": "問題內容",
      "session_id": "abc123",
      "reset": false
    }
    回傳 text/event-stream（SSE），逐字輸出。
    """
    data = await request.json()
    q   = data.get("q", "")
    sid = data.get("session_id", "default")
    if data.get("reset"):
        STORE.clear(sid)

    # 從後端取出清理/修剪後的 history
    history = STORE.get(sid)

    backend = os.getenv("STREAM_BACKEND", USE_STREAM_BACKEND).upper()
    if backend == "OLLAMA":
        model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
        return StreamingResponse(gen_ollama_backend(q, history, model, sid), media_type="text/event-stream")
    else:
        return StreamingResponse(gen_none_backend(q, history, sid), media_type="text/event-stream")

# ================== 健康/診斷 ==================
@app.get("/health")
def health():
    try:
        client = chromadb.PersistentClient(path="./index/chroma")
        db = client.get_collection("kb_main")
        count = db.count()
    except Exception:
        count = -1
    return {
        "collection": "kb_main",
        "count": count,
        "stream_backend": USE_STREAM_BACKEND,
        "ollama_model": OLLAMA_MODEL if USE_STREAM_BACKEND == "OLLAMA" else None
    }

@app.get("/diag/rag")
def diag_rag(q: str = "VIP2 現貨 taker 手續費是多少？"):
    t0 = time.time()
    out = run_once(q, history=[])
    return {
        "elapsed_sec": round(time.time()-t0, 2),
        "contexts": [c["source"] for c in out["contexts"]],
        "tool_keys": list(out["tool_results"].keys()),
    }

@app.get("/diag/history")
def diag_history(session_id: str = Query("default")):
    """檢視並檢查特定 session 的會話歷史（已做清理/修剪）。"""
    return STORE.analyze(session_id)

@app.post("/session/reset")
def session_reset(payload: dict = Body(...)):
    sid = payload.get("session_id", "default")
    STORE.clear(sid)
    return {"ok": True, "session_id": sid}

@app.get("/diag/ollama_gen")
def diag_ollama_gen():
    """確認 /api/generate 是否可用（一次性生成，不走串流）。"""
    import ollama
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    client = ollama.Client(host=host)
    r = client.generate(model=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL), prompt="hi", stream=False)
    return {"host": host, "len": len(r.get("response","")), "preview": r.get("response","")[:50]}
