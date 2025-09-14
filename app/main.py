# app/main.py
import os, asyncio, time
from typing import Any, Dict, List, Deque, Optional, Tuple
from collections import deque
import httpx

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

# ========= 环境变量 =========
BOT_TOKEN       = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o")  # 默认用 gpt-4o
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "dev-secret")
SERPAPI_KEY     = os.getenv("SERPAPI_KEY", "")  # 联网搜索用

TIMEOUT_SEC     = int(os.getenv("TIMEOUT_SEC", "60"))
OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
CHAT_COOLDOWN_SEC = float(os.getenv("CHAT_COOLDOWN_SEC", "1.2"))

HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "8"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "16000"))

SESSION_SCOPE = os.getenv("SESSION_SCOPE", "per_user").lower().strip()  # per_user | per_chat

# ========= 客户端 =========
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
SYSTEM_PROMPT = "你是 ChatGPT，始终结合联网搜索结果回答用户，确保回答最新、真实、有帮助。"

client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)

# ========= 会话存储 =========
_conversations: Dict[Tuple[int, Optional[int]], Deque[Dict[str, str]]] = {}
_last_call_ts: Dict[Tuple[int, Optional[int]], float] = {}

app = FastAPI()

# ========= Telegram =========
async def tg_send_message(chat_id: int, text: str, reply_to: Optional[int] = None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    r = await client.post(f"{TELEGRAM_API}/sendMessage", json=payload)
    r.raise_for_status()
    return r.json()

# ========= 上下文工具 =========
def _session_key(chat_id: int, user_id: Optional[int]) -> Tuple[int, Optional[int]]:
    return (chat_id, None) if SESSION_SCOPE == "per_chat" else (chat_id, user_id)

def _get_history(session_key: Tuple[int, Optional[int]]) -> Deque[Dict[str, str]]:
    if session_key not in _conversations:
        _conversations[session_key] = deque(maxlen=HISTORY_MAX_TURNS * 2)
    return _conversations[session_key]

def _append_history(session_key: Tuple[int, Optional[int]], role: str, content: str):
    _get_history(session_key).append({"role": role, "content": content})

def _build_messages(session_key: Tuple[int, Optional[int]], user_text: str, web_snippets: str) -> List[Dict[str, str]]:
    hist = list(_get_history(session_key))
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if web_snippets:
        messages.append({"role": "system", "content": "以下是最新的网络搜索结果:\n" + web_snippets})

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

# ========= 搜索（SerpAPI） =========
async def web_search(query: str, num: int = 5) -> str:
    if not SERPAPI_KEY:
        return ""
    url = "https://serpapi.com/search"
    params = {"q": query, "hl": "zh-cn", "gl": "us", "num": num, "api_key": SERPAPI_KEY}
    try:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        snippets = []
        for item in data.get("organic_results", [])[:num]:
            title = item.get("title"); link = item.get("link"); snip = item.get("snippet")
            if title and snip:
                snippets.append(f"{title}: {snip} ({link})")
        return "\n".join(snippets)
    except Exception as e:
        return f"(⚠️ 搜索失败: {e})"

# ========= OpenAI =========
async def _chat_once(model: str, messages: List[Dict[str, str]]) -> str:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {"model": model, "messages": messages, "temperature": 0.7}
    async with _openai_sema:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

async def openai_chat(messages: List[Dict[str, str]]) -> str:
    for model in [OPENAI_MODEL, "gpt-4o", "gpt-4o-mini"]:
        try:
            return await _chat_once(model, messages)
        except Exception:
            continue
    raise RuntimeError("OpenAI 调用失败")

# ========= 健康检查 =========
@app.get("/healthz")
async def healthz(): return PlainTextResponse("ok")

@app.get("/")
async def root(): return PlainTextResponse("ok")

# ========= Telegram Webhook =========
@app.post(f"/webhook/{WEBHOOK_SECRET}")
async def telegram_webhook(request: Request):
    update: Dict[str, Any] = await request.json()
    msg = update.get("message") or update.get("edited_message")
    if not msg: return PlainTextResponse("ok")

    chat = msg.get("chat", {}); user = msg.get("from", {}) or {}
    chat_id, user_id = chat.get("id"), user.get("id")
    session_key = _session_key(chat_id, user_id)

    text = msg.get("text"); message_id = msg.get("message_id")
    if not text: return PlainTextResponse("ok")

    cmd = text.strip().lower()
    if cmd == "/start":
        await tg_send_message(chat_id, "✅ 我已开启联网 + 上下文记忆，直接聊天吧！")
        return PlainTextResponse("ok")
    if cmd == "/clear":
        _conversations.pop(session_key, None)
        await tg_send_message(chat_id, "✅ 已清空你在此群的个人上下文。")
        return PlainTextResponse("ok")

    now = time.time()
    if now - _last_call_ts.get(session_key, 0.0) < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "⏳ 正在处理上一条，请稍等…", reply_to=message_id)
        return PlainTextResponse("ok")
    _last_call_ts[session_key] = now

    # 🔎 每次先联网搜索
    web_snippets = await web_search(text)

    messages = _build_messages(session_key, text, web_snippets)
    try:
        reply = await openai_chat(messages)
        await tg_send_message(chat_id, reply, reply_to=message_id)
        _append_history(session_key, "user", text)
        _append_history(session_key, "assistant", reply)
    except Exception as e:
        await tg_send_message(chat_id, f"❌ 出错: {e}", reply_to=message_id)

    return PlainTextResponse("ok")
