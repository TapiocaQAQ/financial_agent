# app/tools.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Callable, List, Optional, Tuple
import time, functools, math

# ===== 介面層 =====
@dataclass
class ToolResult:
    name: str
    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    took_ms: int = 0

@dataclass
class ToolContext:
    """工作記憶：跨工具共享資料；也會累積執行過程與可供 RAG/LLM 消化的文字"""
    bag: Dict[str, Any] = field(default_factory=dict)      # 原始資料池（工具可自由取用）
    results: Dict[str, ToolResult] = field(default_factory=dict)  # 按工具名存放結果
    trace: List[str] = field(default_factory=list)         # 人讀得懂的步驟紀錄（寫進 prompt）
    def put(self, key: str, value: Any): self.bag[key] = value
    def get(self, key: str, default=None): return self.bag.get(key, default)
    def push(self, tr: ToolResult):
        self.results[tr.name] = tr
        if tr.ok:
            self.trace.append(f"[{tr.name}] {tr.data}")
        else:
            self.trace.append(f"[{tr.name}:ERROR] {tr.error}")
    def summarize(self) -> str:
        """壓成 prompt 可讀字串，供模型/回答使用"""
        lines = []
        for n, tr in self.results.items():
            if tr.ok: lines.append(f"- {n}: {tr.data}")
        return "\n".join(lines) if lines else "(無工具結果)"

ToolFn = Callable[[ToolContext, Dict[str, Any]], ToolResult]

class ToolBus:
    """工具註冊、執行、依賴輸入互用"""
    def __init__(self):
        self._registry: Dict[str, ToolFn] = {}
        self._descs: Dict[str, str] = {}
    def register(self, name: str, fn: ToolFn, desc: str = ""):
        self._registry[name] = fn
        self._descs[name] = desc
        return fn
    def list(self) -> Dict[str, str]:
        return dict(self._descs)
    def run(self, ctx: ToolContext, plan: List[Tuple[str, Dict[str, Any]]]) -> ToolContext:
        for name, kwargs in plan:
            fn = self._registry.get(name)
            if not fn:
                ctx.push(ToolResult(name=name, ok=False, error="tool not found"))
                continue
            t0 = time.time()
            try:
                # kwargs 可包含從 ctx 取值的占位符：例如 {"price": "${market.price}"}
                resolved = resolve_inputs(ctx, kwargs)
                out = fn(ctx, resolved)
                out.took_ms = int((time.time() - t0) * 1000)
                ctx.push(out)
            except Exception as e:
                ctx.push(ToolResult(name=name, ok=False, error=f"{type(e).__name__}: {e}",
                                    took_ms=int((time.time()-t0)*1000)))
        return ctx

def resolve_inputs(ctx: ToolContext, params: Dict[str, Any]) -> Dict[str, Any]:
    """支援用 ${tool.key} 或 ${bag.key} 取先前資料"""
    def _resolve(v):
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            path = v[2:-1]  # tool.key 或 bag.key
            if path.startswith("bag."):
                return ctx.get(path[4:])
            if "." in path:
                t, k = path.split(".", 1)
                tr = ctx.results.get(t)
                return (tr.data.get(k) if tr and tr.ok else None)
        return v
    return {k: _resolve(v) for k, v in params.items()}

# ===== 內建工具 =====
bus = ToolBus()

def tool(name: str, desc: str = ""):
    def deco(fn: ToolFn):
        return bus.register(name, fn, desc)
    return deco

@tool("fee.lookup", "查手續費（例如 vip=2, market=spot, side=taker）")
def fee_lookup(ctx: ToolContext, p: Dict[str, Any]) -> ToolResult:
    vip = int(p.get("vip", 2))
    market = p.get("market", "spot")
    side = p.get("side", "taker")
    # 你已有 fees 邏輯；這裡簡化：寫死或從 ctx.bag 取表格
    table = ctx.get("fee_table") or {
        ("spot","maker",1): 0.0010, ("spot","taker",1): 0.0010,
        ("spot","maker",2): 0.0008, ("spot","taker",2): 0.0009,
        ("spot","maker",4): 0.0006, ("spot","taker",4): 0.0007,
        ("futures","maker",2): 0.00012, ("futures","taker",2): 0.00032,
    }
    fee = table.get((market, side, vip))
    if fee is None:
        return ToolResult(name="fee.lookup", ok=False, error=f"no fee for {market}/{side}/VIP{vip}")
    return ToolResult(name="fee.lookup", ok=True, data={"vip": vip, "market": market, "side": side, "fee": fee})

@tool("market.price", "取得即時價格（symbol，例如 BTCUSDT）")
def market_price(ctx: ToolContext, p: Dict[str, Any]) -> ToolResult:
    symbol = p.get("symbol","BTCUSDT")
    # 這裡用假資料或接你未來的行情工具
    price = float(p.get("mock_price", 65000.0))
    return ToolResult(name="market.price", ok=True, data={"symbol": symbol, "price": price})

@tool("fx.rate", "法幣匯率（base/quote）")
def fx_rate(ctx: ToolContext, p: Dict[str, Any]) -> ToolResult:
    base, quote = p.get("base","USD"), p.get("quote","TWD")
    rate = float(p.get("mock_rate", 32.5))  # 假資料；之後可接匯率 API
    return ToolResult(name="fx.rate", ok=True, data={"base": base, "quote": quote, "rate": rate})

@tool("trade.quote", "用 price, qty, fee 計算下單成本（可自動相依）")
def trade_quote(ctx: ToolContext, p: Dict[str, Any]) -> ToolResult:
    price = float(p.get("price", 0) or 0)
    qty   = float(p.get("qty",   0) or 0)
    fee   = float(p.get("fee",   0) or 0)  # 比率
    if price <= 0 or qty <= 0:
        return ToolResult(name="trade.quote", ok=False, error="price/qty required")
    gross = price * qty
    fee_amt = gross * fee
    net = gross + fee_amt
    return ToolResult(name="trade.quote", ok=True, data={
        "price": price, "qty": qty, "fee_ratio": fee,
        "gross": round(gross, 8), "fee_amt": round(fee_amt, 8), "net": round(net, 8)
    })

@tool("convert.fiat", "把 USD 成本換算成 TWD（依賴 fx.rate）")
def convert_fiat(ctx: ToolContext, p: Dict[str, Any]) -> ToolResult:
    amount_usd = float(p.get("amount_usd", 0))
    rate = float(p.get("rate", 0))
    if amount_usd <= 0 or rate <= 0:
        return ToolResult(name="convert.fiat", ok=False, error="amount_usd/rate required")
    twd = amount_usd * rate
    return ToolResult(name="convert.fiat", ok=True, data={"amount_twd": round(twd, 2)})
