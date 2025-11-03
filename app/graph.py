from typing import Dict, List, TypedDict
from app.retriever import hybrid_search
from app.tools import pnl_calc, lookup_fee

class S(TypedDict):
    query: str
    history: List[Dict]
    contexts: List[Dict]
    tool_results: Dict
    answer: str

def planner(q: str):
    ql = q.lower()
    need_tool = any(k in ql for k in ["pnl","盈虧","強平","leverage","槓桿","fee","手續費"])
    return {"need_tool": need_tool}

def retrieve(q: str):
    return {"contexts": hybrid_search(q, k=4)}

def maybe_run_tools(q: str):
    # 極簡判斷：若句中有「VIP2 手續費」→查表；若有「pnl」→示範計算
    out = {}
    if "vip2" in q.lower() and "手續費" in q:
        out["vip2_spot_taker_fee"] = lookup_fee("VIP2","spot","taker")
    if "pnl" in q.lower():
        out["pnl_demo"] = pnl_calc("long", entry=2500, price=2620, qty=3, fee_rate=0.0004, leverage=20)
    return {"tool_results": out}

def render_answer(q: str, ctxs: List[Dict], tools: Dict):
    lines = []
    if tools: lines.append(f"工具結果: {tools}")
    if ctxs:
        lines.append("依據資料：")
        for c in ctxs:
            lines.append(f"- {c['source']}")
    else:
        lines.append("（未命中知識庫段落）")
    return "\n".join(lines)

def run_once(query:str, history:List[Dict]):
    plan = planner(query)
    ctx = retrieve(query)["contexts"]
    tools = maybe_run_tools(query)["tool_results"] if plan["need_tool"] else {}
    ans = render_answer(query, ctx, tools)
    return {"query": query, "history": history, "contexts": ctx, "tool_results": tools, "answer": ans}
