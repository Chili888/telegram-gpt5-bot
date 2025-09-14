# app/main.py
import os
import asyncio
import time
from typing import Any, Dict, List, Deque, Optional, Tuple
from collections import deque

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from serpapi import GoogleSearch

# ================== 环境变量 ==================
BOT_TOKEN       = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4.1")  # 优先模型
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "dev-secret")
SERPAPI_KEY     = os.getenv("SERPAPI_KEY", "")
TIMEOUT_SEC     = int(os.getenv("TIMEOUT_SEC", "60"))

# 并发/防抖
OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
CHAT_COOLDOWN_SEC      = float(os.getenv("CHAT_COOLDOWN_SEC", "1.2"))

# 会话记忆
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "12"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "24000"))
SESSION_SCOPE     = os.getenv("SESSION_SCOPE", "per_user").lower().strip()  # per_user | per_chat

# ================== 常量与客户端 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
SYSTEM_PROMPT = (
    "You are ChatGPT with web access. "
    "Always combine recent search results with conversation history. "
    "Be concise, helpful, safe, and reply in the same language the user used."
)

client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
_last_call_ts: Dict[int, float] = {}

# 内存会话：session_key -> deque(messages)
_conversations: Dict[Tuple[int, Optional[int]], Deque[Dict[str, str]]] = {}

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
def _session_key(chat_id: int, user_id: Optional[int]) -> Tuple[int, Optional[int]]:
    if SESSION_SCOPE == "per_chat":
        return (chat_id, None)
    return (chat_id, user_id)

def _get_history(session_key: Tuple[int, Optional[int]]) -> Deque[Dict[str, str]]:
    if session_key not in _conversations:
        _conversations[session_key] = deque(maxlen=HISTORY_MAX_TURNS * 2)
    return _conversations[session_key]

def _append_history(session_key: Tuple[int, Optional[int]], role: str, content: str):
    _get_history(session_key).append({"role": role, "content": content})

def _build_messages(session_key: Tuple[int, Optional[int]], user_text: str) -> List[Dict[str, str]]:
    hist = list(_get_history(session_key))
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    acc: List[Dict[str, str]] = []
    total = 0
    for m in reversed(hist):
        c = m.get("content") or ""
        if total + len(c) > HISTORY_MAX_CHARS:
            break
        acc.append(m); total += len(c)
    messages.extend(reversed(acc))
    messages.append({"role": "user", "content": user_text})
    return messages

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

async def openai_chat(messages: List[Dict[str, str]]) -> str:
    preferred = [OPENAI_MODEL, "gpt-4.1", "gpt-4o", "gpt-4o-mini"]
    seen = set(); models_to_try: List[str] = []
    for m in preferred:
        if m not in seen:
            models_to_try.append(m); seen.add(m)

    last_err: Optional[Exception] = None
    for model in models_to_try:
        try:
            return await _chat_once(model, messages)
        except Exception as e:
            last_err = e
            continue
    if last_err: raise last_err
    raise RuntimeError("OpenAI 调用失败")

# ================== SerpAPI 联网 ==================
async def web_search(query: str) -> str:
    if not SERPAPI_KEY:
        return ""
    params = {"engine": "google", "q": query, "api_key": SERPAPI_KEY, "num": 5}
    search = GoogleSearch(params)
    results = search.get_dict()
    ans = []
    if "answer_box" in results:
        ans.append(results["answer_box"].get("answer") or results["answer_box"].get("snippet", ""))
    if "organic_results" in results:
        for r in results["organic_results"][:3]:
            ans.append(r.get("snippet", ""))
    return "\n".join([a for a in ans if a]).strip()

# ================== 健康检查 ==================
@app.get("/healthz")
async def healthz(): return PlainTextResponse("ok")

@app.get("/")
async def root(): return PlainTextResponse("ok")

# ================== Telegram Webhook ==================
@app.post(f"/webhook/{WEBHOOK_SECRET}")
async def telegram_webhook(request: Request):
    update: Dict[str, Any] = await request.json()
    print("UPDATE >>>", update)

    msg = update.get("message") or update.get("edited_message") or update.get("channel_post")
    if not msg: return PlainTextResponse("ok")

    chat = msg.get("chat", {})
    user = msg.get("from", {}) or {}
    chat_id = chat.get("id")
    user_id = user.get("id")
    session_key = _session_key(chat_id, user_id)
    text = msg.get("text")
    message_id = msg.get("message_id")

    if not text:
        await tg_send_message(chat_id, "我目前只支持文字消息～", reply_to=message_id)
        return PlainTextResponse("ok")

    cmd = text.strip().lower()
    if cmd == "/start":
        await tg_send_message(chat_id, "✅ 你好！我已开启上下文记忆（最近多轮），并默认联网搜索。")
        return PlainTextResponse("ok")
    if cmd == "/clear":
        _conversations.pop(session_key, None)
        await tg_send_message(chat_id, "✅ 已清空你在此群的个人上下文。")
        return PlainTextResponse("ok")

    now = time.time()
    if now - _last_call_ts.get(chat_id, 0.0) < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "我正在处理上一条消息，请稍等片刻～", reply_to=message_id)
        return PlainTextResponse("ok")
    _last_call_ts[chat_id] = now

    # ==== 联网搜索 ====
    await tg_send_message(chat_id, "🔎 正在联网搜索，请稍候…", reply_to=message_id)
    search_result = await web_search(text)
    user_input = text
    if search_result:
        user_input += f"\n\n以下是最新搜索结果供参考:\n{search_result}"

    # ==== 组装 messages ====
    messages = _build_messages(session_key, user_input)

    try:
        reply = await openai_chat(messages)
        await tg_send_message(chat_id, reply, reply_to=message_id)
        _append_history(session_key, "user", text)
        _append_history(session_key, "assistant", reply)
    except Exception as e:
        await tg_send_message(chat_id, f"❌ 出错：{e}", reply_to=message_id)

    return PlainTextResponse("ok")
