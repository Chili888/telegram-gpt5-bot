# app/main.py
import os
import asyncio
import time
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

# ================== 环境变量 ==================
BOT_TOKEN       = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 默认使用 gpt-4o-mini（对所有账号可用）
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4.1")

WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "dev-secret")
TIMEOUT_SEC     = int(os.getenv("TIMEOUT_SEC", "60"))

# 并发/防抖配置
OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
CHAT_COOLDOWN_SEC      = float(os.getenv("CHAT_COOLDOWN_SEC", "1.2"))

# ================== 常量与客户端 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
SYSTEM_PROMPT = (
    "You are ChatGPT. Be concise, helpful, and safe. "
    "Respond in the same language the user used."
)

client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
_last_call_ts: dict[int, float] = {}

app = FastAPI()


# ================== 工具函数 ==================
async def tg_send_message(chat_id: int, text: str, reply_to: int | None = None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    r = await client.post(f"{TELEGRAM_API}/sendMessage", json=payload)
    r.raise_for_status()
    return r.json()


async def openai_chat(messages: List[dict[str, str]]) -> str:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.7}

    async with _openai_sema:
        delay = 1.0
        for attempt in range(5):
            try:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in (429, 503):
                    ra = e.response.headers.get("retry-after")
                    wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                    wait = max(1.0, min(wait, 20.0))
                    await asyncio.sleep(wait)
                    delay = min(delay * 2, 20.0)
                    continue
                raise
            except httpx.HTTPError:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 20.0)
                continue

        raise httpx.HTTPError("OpenAI服务繁忙，请稍后再试")


# ================== 健康检查/根路由 ==================
@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@app.get("/")
async def root():
    return PlainTextResponse("ok")


# ================== Telegram Webhook ==================
@app.post(f"/webhook/{WEBHOOK_SECRET}")
async def telegram_webhook(request: Request):
    update: Dict[str, Any] = await request.json()
    print("UPDATE >>>", update)

    msg = update.get("message") or update.get("edited_message") or update.get("channel_post")
    if not msg:
        return PlainTextResponse("ok")

    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = msg.get("text")
    message_id = msg.get("message_id")

    if not text:
        if chat_id is not None:
            await tg_send_message(chat_id, "我目前只支持文字消息～", reply_to=message_id)
        return PlainTextResponse("ok")

    if text.strip().lower() == "/start":
        await tg_send_message(chat_id, "✅ 你好！我是 GPT-机器人，直接发消息和我对话吧～")
        return PlainTextResponse("ok")

    # 防抖：避免同一 chat 瞬间多次触发
    now = time.time()
    last = _last_call_ts.get(chat_id, 0.0)
    if now - last < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "我正在处理上一条消息，请稍等片刻～", reply_to=message_id)
        return PlainTextResponse("ok")
    _last_call_ts[chat_id] = now

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    try:
        reply = await openai_chat(messages)
        await tg_send_message(chat_id, reply, reply_to=message_id)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            await tg_send_message(chat_id, "⚠️ OpenAI 限流啦，我会自动重试；如果仍失败请稍后再试～", reply_to=message_id)
        elif e.response.status_code in (401, 403):
            await tg_send_message(chat_id, "❌ OpenAI API Key 错误或无权限，请检查。", reply_to=message_id)
        else:
            await tg_send_message(chat_id, f"❌ OpenAI 错误：{e.response.status_code}", reply_to=message_id)
    except httpx.HTTPError as e:
        await tg_send_message(chat_id, f"❌ 网络异常：{e}", reply_to=message_id)

    return PlainTextResponse("ok")

