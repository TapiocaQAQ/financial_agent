# API 參考摘要

## 一、行情查詢

GET /api/v1/ticker/price?symbol=BTCUSDT
回傳:
{
"symbol": "BTCUSDT",
"price": "68250.12"
}

## 二、帳戶餘額

GET /api/v1/account
Header: Authorization: Bearer <token>

## 三、下單接口

POST /api/v1/order
Body:
{
"symbol": "ETHUSDT",
"side": "BUY",
"type": "LIMIT",
"price": 2500,
"quantity": 2
}

回傳:
{"orderId": 123456789, "status": "NEW"}

---

## 四、計算範例
若使用者請求「查詢 ETHUSDT 最新價格」，Agent 應該：
1. 呼叫 `/api/v1/ticker/price?symbol=ETHUSDT`
2. 解析返回的 `"price"` 值
3. 回覆使用者：「目前 ETH 價格為 2500 USDT。」

---

## 五、安全性與限制
- 每秒最多 10 次請求
- 所有私有接口均需 JWT Token 驗證
- 伺服器回傳 HTTP 429 時應進行退避重試