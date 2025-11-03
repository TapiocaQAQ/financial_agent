def lookup_fee(vip="VIP0", market="spot", role="taker"):
    table = {
        ("VIP0","spot","taker"):0.001, ("VIP0","spot","maker"):0.001,
        ("VIP2","spot","taker"):0.0009, ("VIP2","spot","maker"):0.0008,
        ("一般用戶","perp","taker"):0.0004, ("一般用戶","perp","maker"):0.0002
    }
    return table.get((vip, market, role))

def pnl_calc(side: str, entry: float, price: float, qty: float, fee_rate: float=0.0004, leverage:int=1):
    pnl = (price - entry) * qty * (1 if side.lower()=="long" else -1)
    fees = (entry + price) * qty * fee_rate
    roe = pnl / max(entry*qty/leverage, 1e-9)
    return {"pnl": round(pnl,2), "fees": round(fees,2), "roe_pct": round(roe*100,2)}
