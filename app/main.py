# app/main.py
import os
import asyncio
import time
from typing import Any, Dict, List, Deque, Optional
from collections import deque

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

# ================== 环境变量 ==================
BOT_TOKEN       = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4.1")
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "dev-secret")
TIMEOUT_SEC     = int(os.getenv("TIMEOUT_SEC", "60"))

# 并发/防抖
OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
CHAT_COOLDOWN_SEC      = float(os.getenv("CHAT_COOLDOWN_SEC", "1.2"))

# 会话记忆
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "8"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "16000"))

# ================== 常量与客户端 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
SYSTEM_PROMPT = (
    "You are ChatGPT with web access. Always answer with the latest web results. "
    "Be concise, helpful, and safe. Respond in the same language the user used."
)

client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
_last_call_ts: Dict[int, float] = {}
_conversations: Dict[int, Deque[Dict[str, str]]] = {}
_last_query: Dict[int, str] = {}  # 每个 chat 的最后一次查询

app = FastAPI()

# ================== Telegram 工具 ==================
async def tg_send_message(chat_id: int, text: str, reply_to: Optional[int] = None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    r = await client.post(f"{TELEGRAM_API}/sendMessage", json=payload)
    r.raise_for_status()
    return r.json()

# ================== 会话记忆 ==================
def _get_history(chat_id: int) -> Deque[Dict[str, str]]:
    if chat_id not in _conversations:
        _conversations[chat_id] = deque(maxlen=HISTORY_MAX_TURNS * 2)
    return _conversations[chat_id]

def _append_history(chat_id: int, role: str, content: str):
    _get_history(chat_id).append({"role": role, "content": content})

def _build_messages(chat_id: int, user_text: str, web_snippets: str = "") -> List[Dict[str, str]]:
    hist = list(_get_history(chat_id))
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    acc: List[Dict[str, str]] = []
    total = 0
    for m in reversed(hist):
        c = m.get("content") or ""
        if total + len(c) > HISTORY_MAX_CHARS:
            break
        acc.append(m)
        total += len(c)
    messages.extend(reversed(acc))

    if web_snippets:
        messages.append({"role": "system", "content": f"Latest web results:\n{web_snippets}"})

    messages.append({"role": "user", "content": user_text})
    return messages

# ================== 联网搜索 ==================
async def web_search(query: str, num: int = 5) -> str:
    serpapi_key = os.getenv("SERPAPI_KEY", "")
    if not serpapi_key:
        print("⚠️ [WARN] SERPAPI_KEY 未设置，无法联网搜索")
        return ""

    url = "https://serpapi.com/search"
    params = {"engine": "google", "q": query, "num": num, "api_key": serpapi_key}

    try:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        js = resp.json()

        print("🔍 [DEBUG] Web query:", query)
        print("🔍 [DEBUG] Raw response keys:", list(js.keys()))

        snippets = []
        for item in js.get("organic_results", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            if snippet:
                snippets.append(f"{title} - {snippet} ({link})")

        if not snippets:
            print("⚠️ [WARN] 搜索结果为空，返回原始 JSON：", js)

        return "\n".join(snippets[:num])
    except Exception as e:
        print(f"❌ [ERROR] Web search failed: {e}")
        return ""

# ================== OpenAI 调用 ==================
async def _chat_once(model: str, messages: List[Dict[str, str]]) -> str:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {"model": model, "messages": messages, "temperature": 0.7}

    async with _openai_sema:
        delay = 1.0
        for _ in range(5):
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
                    await asyncio.sleep(wait)
                    delay = min(delay * 2, 20.0)
                    continue
                raise
            except httpx.HTTPError:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 20.0)
                continue
        raise httpx.HTTPError("OpenAI服务繁忙，请稍后再试")

async def openai_chat(messages: List[Dict[str, str]]) -> str:
    preferred = [OPENAI_MODEL, "gpt-4.1", "gpt-4o", "gpt-4o-mini"]
    for model in preferred:
        try:
            return await _chat_once(model, messages)
        except Exception as e:
            print(f"⚠️ [WARN] 模型 {model} 调用失败: {e}")
            continue
    raise RuntimeError("OpenAI 调用失败（所有模型不可用）")

# ================== 健康检查 ==================
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

    cmd = text.strip().lower()
    if cmd == "/start":
        await tg_send_message(chat_id, "✅ 你好！我已开启联网和上下文记忆，直接发消息继续对话吧～")
        return PlainTextResponse("ok")
    if cmd == "/clear":
        _conversations.pop(chat_id, None)
        await tg_send_message(chat_id, "✅ 已清空本会话上下文。")
        return PlainTextResponse("ok")

    now = time.time()
    if now - _last_call_ts.get(chat_id, 0.0) < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "我正在处理上一条消息，请稍等片刻～", reply_to=message_id)
        return PlainTextResponse("ok")
    _last_call_ts[chat_id] = now

    # 默认联网搜索
    web_snippets = await web_search(text, num=8)
    if web_snippets:
        _last_query[chat_id] = text

    messages = _build_messages(chat_id, text, web_snippets)

    try:
        reply = await openai_chat(messages)
        await tg_send_message(chat_id, reply, reply_to=message_id)
        _append_history(chat_id, "user", text)
        _append_history(chat_id, "assistant", reply)
    except Exception as e:
        await tg_send_message(chat_id, f"❌ 错误：{e}", reply_to=message_id)

    return PlainTextResponse("ok")
