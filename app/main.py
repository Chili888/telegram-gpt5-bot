import os
import asyncio
import time
import re
from typing import Any, Dict, List, Deque, Optional
from collections import deque

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

# ================== 环境变量 ==================
BOT_TOKEN       = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o")   # 默认 GPT-4o（GPT-5 内核）
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "dev-secret")
SERPAPI_KEY     = os.getenv("SERPAPI_KEY", "")
TIMEOUT_SEC     = int(os.getenv("TIMEOUT_SEC", "60"))

OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
CHAT_COOLDOWN_SEC      = float(os.getenv("CHAT_COOLDOWN_SEC", "1.2"))

HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "8"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "16000"))

# ================== 常量与客户端 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
SYSTEM_PROMPT = (
    "You are ChatGPT with real-time browsing ability. "
    "Always use the latest search results provided to you. "
    "Respond in the same language as the user."
)

client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
_last_call_ts: Dict[str, float] = {}

# ================== 会话存储 ==================
# chat_id:user_id -> deque of {role, content, ts}
_conversations: Dict[str, Deque[Dict[str, Any]]] = {}

def _conv_key(chat_id: int, user_id: int) -> str:
    return f"{chat_id}:{user_id}"

def _get_history(chat_id: int, user_id: int) -> Deque[Dict[str, Any]]:
    key = _conv_key(chat_id, user_id)
    if key not in _conversations:
        _conversations[key] = deque(maxlen=HISTORY_MAX_TURNS * 2)
    return _conversations[key]

def _append_history(chat_id: int, user_id: int, role: str, content: str):
    _get_history(chat_id, user_id).append({
        "role": role,
        "content": content,
        "ts": time.time()
    })

def _build_messages(chat_id: int, user_id: int, user_text: str, search_results: str) -> List[Dict[str, str]]:
    # 只保留 24 小时内的历史
    hist = [m for m in _get_history(chat_id, user_id) if time.time() - m.get("ts", 0) < 86400]
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Here are the latest search results:\n{search_results}"}
    ]
    acc: List[Dict[str, str]] = []
    total = 0
    for m in reversed(hist):
        c = m.get("content") or ""
        if total + len(c) > HISTORY_MAX_CHARS:
            break
        acc.append({"role": m["role"], "content": c})
        total += len(c)
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
                    await asyncio.sleep(wait)
                    delay = min(delay * 2, 20.0)
                    continue
                raise
            except httpx.HTTPError:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 20.0)
                continue
        raise httpx.HTTPError("OpenAI 服务繁忙，请稍后再试")

async def openai_chat(messages: List[Dict[str, str]]) -> str:
    preferred = [OPENAI_MODEL, "gpt-4o", "gpt-4o-mini"]
    for model in preferred:
        try:
            return await _chat_once(model, messages)
        except Exception:
            continue
    raise RuntimeError("OpenAI 调用失败")

# ================== 搜索功能 (SerpAPI) ==================
async def search_web(query: str) -> str:
    if not SERPAPI_KEY:
        return "⚠️ 未配置 SERPAPI_KEY，无法联网搜索。"
    url = "https://serpapi.com/search"
    params = {"q": query, "hl": "zh-cn", "gl": "cn", "api_key": SERPAPI_KEY}
    try:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("organic_results", [])[:5]:
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            results.append(f"- {title}\n{snippet}\n{link}")
        return "\n\n".join(results) if results else "未找到相关搜索结果"
    except Exception as e:
        return f"❌ 搜索失败: {e}"

# ================== 文本清理 ==================
def clean_text(text: str) -> str:
    return re.sub(r"[#*`>]", "", text)

# ================== Telegram 工具 ==================
async def tg_send_message(chat_id: int, text: str, reply_to: Optional[int] = None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    r = await client.post(f"{TELEGRAM_API}/sendMessage", json=payload)
    r.raise_for_status()
    return r.json()

# ================== 自动清理过期上下文 ==================
async def cleanup_history():
    """清理超过24小时的上下文"""
    now = time.time()
    expired = 0
    for key, dq in list(_conversations.items()):
        new_dq = deque([m for m in dq if now - m.get("ts", 0) < 86400], maxlen=dq.maxlen)
        if len(new_dq) < len(dq):
            expired += len(dq) - len(new_dq)
        _conversations[key] = new_dq
    if expired:
        print(f"[CLEANUP] 清理了 {expired} 条过期上下文")

async def background_tasks():
    while True:
        await asyncio.sleep(86400)  # 每24小时执行一次
        await cleanup_history()

# ================== FastAPI 路由 ==================
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_tasks())

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@app.get("/")
async def root():
    return PlainTextResponse("ok")

@app.post(f"/webhook/{WEBHOOK_SECRET}")
async def telegram_webhook(request: Request):
    update: Dict[str, Any] = await request.json()
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return PlainTextResponse("ok")

    chat_id = msg.get("chat", {}).get("id")
    user_id = msg.get("from", {}).get("id")
    text = msg.get("text")
    message_id = msg.get("message_id")

    if not text:
        await tg_send_message(chat_id, "我目前只支持文字消息～", reply_to=message_id)
        return PlainTextResponse("ok")

    cmd = text.strip().lower()
    if cmd == "/start":
        await tg_send_message(chat_id, "✅ 你好！我已开启上下文记忆 + 实时联网搜索。直接发消息继续对话吧～")
        return PlainTextResponse("ok")
    if cmd == "/clear":
        _conversations.pop(_conv_key(chat_id, user_id), None)
        await tg_send_message(chat_id, "✅ 已清空你的上下文。")
        return PlainTextResponse("ok")

    now = time.time()
    key = _conv_key(chat_id, user_id)
    if now - _last_call_ts.get(key, 0.0) < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "我正在处理上一条消息，请稍等片刻～", reply_to=message_id)
        return PlainTextResponse("ok")
    _last_call_ts[key] = now

    # 🔍 先搜索
    search_results = await search_web(text)

    # 构造上下文
    messages = _build_messages(chat_id, user_id, text, search_results)

    try:
        reply = await openai_chat(messages)
        reply = clean_text(reply)
        await tg_send_message(chat_id, reply, reply_to=message_id)
        _append_history(chat_id, user_id, "user", text)
        _append_history(chat_id, user_id, "assistant", reply)
    except Exception as e:
        await tg_send_message(chat_id, f"❌ 错误: {e}", reply_to=message_id)

    return PlainTextResponse("ok")
