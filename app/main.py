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
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4.1")  # 默认模型
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "dev-secret")
TIMEOUT_SEC     = int(os.getenv("TIMEOUT_SEC", "60"))

# 并发/防抖
OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
CHAT_COOLDOWN_SEC      = float(os.getenv("CHAT_COOLDOWN_SEC", "1.2"))

# 会话记忆
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "8"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "16000"))

# SerpAPI
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

# ================== 常量与客户端 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
SYSTEM_PROMPT = (
    "You are ChatGPT with browsing ability. Always search the web (SerpAPI) "
    "before answering if user asks a question. Be concise, safe, and reply in "
    "the same language as the user."
)

client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
_last_call_ts: Dict[int, float] = {}
_conversations: Dict[int, Deque[Dict[str, str]]] = {}

app = FastAPI()

# ================== Telegram 工具 ==================
async def tg_send_message(chat_id: int, text: str, reply_to: Optional[int] = None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    r = await client.post(f"{TELEGRAM_API}/sendMessage", json=payload)
    r.raise_for_status()
    return r.json()

# ================== 会话记忆工具 ==================
def _get_history(chat_id: int) -> Deque[Dict[str, str]]:
    if chat_id not in _conversations:
        _conversations[chat_id] = deque(maxlen=HISTORY_MAX_TURNS * 2)
    return _conversations[chat_id]

def _append_history(chat_id: int, role: str, content: str):
    _get_history(chat_id).append({"role": role, "content": content})

def _build_messages(chat_id: int, user_text: str) -> List[Dict[str, str]]:
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
    messages.append({"role": "user", "content": user_text})
    return messages

# ================== SerpAPI 搜索 ==================
async def web_search(query: str) -> str:
    if not SERPAPI_KEY:
        return ""
    url = "https://serpapi.com/search"
    params = {"q": query, "hl": "zh-cn", "gl": "cn", "api_key": SERPAPI_KEY}
    try:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("organic_results", [])[:3]:
            title = item.get("title")
            link = item.get("link")
            snippet = item.get("snippet", "")
            results.append(f"{title}\n{snippet}\n{link}")
        return "\n\n".join(results) if results else ""
    except Exception as e:
        return f"(搜索失败: {e})"

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
                    await asyncio.sleep(delay)
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
    last_err: Optional[Exception] = None
    for model in preferred:
        try:
            return await _chat_once(model, messages)
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("OpenAI 调用失败")

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

    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return PlainTextResponse("ok")

    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = msg.get("text")
    message_id = msg.get("message_id")

    if not text:
        if chat_id:
            await tg_send_message(chat_id, "我目前只支持文字消息～", reply_to=message_id)
        return PlainTextResponse("ok")

    cmd = text.strip().lower()
    if cmd == "/start":
        await tg_send_message(chat_id, "✅ 你好！我已开启上下文记忆 + 联网搜索，直接发消息即可～")
        return PlainTextResponse("ok")
    if cmd == "/clear":
        _conversations.pop(chat_id, None)
        await tg_send_message(chat_id, "✅ 已清空本会话上下文。")
        return PlainTextResponse("ok")

    now = time.time()
    if now - _last_call_ts.get(chat_id, 0.0) < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "⏳ 我正在处理上一条消息，请稍等～", reply_to=message_id)
        return PlainTextResponse("ok")
    _last_call_ts[chat_id] = now

    # 自动联网搜索
    search_result = await web_search(text)
    user_input = text
    if search_result:
        user_input += f"\n\n以下是最新搜索结果供参考:\n{search_result}"

    messages = _build_messages(chat_id, user_input)

    try:
        reply = await openai_chat(messages)
        await tg_send_message(chat_id, reply, reply_to=message_id)
        _append_history(chat_id, "user", text)
        _append_history(chat_id, "assistant", reply)
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code == 429:
            await tg_send_message(chat_id, "⚠️ OpenAI 限流，请稍后再试～", reply_to=message_id)
        elif code in (401, 403):
            await tg_send_message(chat_id, "❌ OpenAI API Key 或权限问题。", reply_to=message_id)
        elif code in (400, 404):
            await tg_send_message(chat_id, "❌ 模型不可用（已尝试切换）。", reply_to=message_id)
        else:
            await tg_send_message(chat_id, f"❌ OpenAI 错误：HTTP {code}", reply_to=message_id)
    except Exception as e:
        await tg_send_message(chat_id, f"❌ 出错：{e}", reply_to=message_id)

    return PlainTextResponse("ok")
