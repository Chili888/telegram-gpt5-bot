# app/main.py
import os, asyncio, json
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import httpx

# ===== 环境变量 =====
BOT_TOKEN        = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")  # 如无自定义，保持默认
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-5")  # 你可以改成你账号可用的具体型号
WEBHOOK_SECRET   = os.getenv("WEBHOOK_SECRET", "dev-secret")  # 用于 webhook 路径
TIMEOUT_SEC      = int(os.getenv("TIMEOUT_SEC", "60"))

if not BOT_TOKEN or not OPENAI_API_KEY:
    print("⚠️ 环境变量缺失：请设置 BOT_TOKEN 和 OPENAI_API_KEY")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

app = FastAPI()

# ====== 简单的系统提示（可按需修改）======
SYSTEM_PROMPT = (
    "You are ChatGPT (GPT-5). Be concise, helpful, and safe. "
    "Respond in the same language the user used."
)

# ===== 公共 HTTP 客户端 =====
client = httpx.AsyncClient(timeout=TIMEOUT_SEC)

async def tg_send_message(chat_id: int, text: str, reply_to: int | None = None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    r = await client.post(f"{TELEGRAM_API}/sendMessage", json=payload)
    r.raise_for_status()
    return r.json()

async def openai_chat(messages: list[dict[str, str]]) -> str:
    """
    调 OpenAI Chat Completions（兼容常见 SDK 接口风格）。
    如果你的账号只开了 responses API，也可以改为 /responses。
    """
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }
    r = await client.post(url, headers=headers, json=body)
    r.raise_for_status()
    data = r.json()
    # 标准 chat.completions 输出
    return data["choices"][0]["message"]["content"]

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@app.post(f"/webhook/{WEBHOOK_SECRET}")
async def telegram_webhook(request: Request):
    update = await request.json()
    print("UPDATE >>>", update)  # 打印整包，方便在 Render Logs 里看
    msg = update.get("message") or update.get("edited_message") or update.get("channel_post")
    if not msg:
        return PlainTextResponse("ok")  # 忽略非文本更新

    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = msg.get("text")
    message_id = msg.get("message_id")

    # 如果不是文本，给个兜底回复，避免“不可访问消息”困惑
    if not text:
        await tg_send_message(chat_id, "我目前只支持文字消息～", reply_to=message_id if message_id else None)
        return PlainTextResponse("ok")

    if text.strip().lower() == "/start":
        await tg_send_message(chat_id, "✅ 你好！我是 GPT-5 机器人，直接发消息和我对话吧～")
        return PlainTextResponse("ok")

    # 调 OpenAI
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        reply = await openai_chat(messages)
    except Exception as e:
        await tg_send_message(chat_id, f"❌ OpenAI 调用失败：{e}", reply_to=message_id)
        return PlainTextResponse("ok")

    await tg_send_message(chat_id, reply, reply_to=message_id)
    return PlainTextResponse("ok")

