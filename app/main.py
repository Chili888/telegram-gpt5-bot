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
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4.1")  # 首选；不可用时会自动降级
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "dev-secret")
TIMEOUT_SEC     = int(os.getenv("TIMEOUT_SEC", "60"))

# 并发/防抖
OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
CHAT_COOLDOWN_SEC      = float(os.getenv("CHAT_COOLDOWN_SEC", "1.2"))

# 会话记忆（每会话最近 N 轮；粗略字符上限）
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "8"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "16000"))

# ================== 常量与客户端 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
SYSTEM_PROMPT = (
    "You are ChatGPT. Be concise, helpful, and safe. "
    "Respond in the same language the user used."
)

client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
_last_call_ts: Dict[int, float] = {}

# 内存会话：chat_id -> deque(messages)
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
        _conversations[chat_id] = deque(maxlen=HISTORY_MAX_TURNS * 2)  # 一轮= user+assistant 两条
    return _conversations[chat_id]

def _append_history(chat_id: int, role: str, content: str):
    _get_history(chat_id).append({"role": role, "content": content})

def _build_messages(chat_id: int, user_text: str) -> List[Dict[str, str]]:
    hist = list(_get_history(chat_id))
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # 从后往前累加，限制总字符
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

# ================== OpenAI 调用（自动降级 + 退避重试） ==================
async def _chat_once(model: str, messages: List[Dict[str, str]]) -> str:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {"model": model, "messages": messages, "temperature": 0.7}

    async with _openai_sema:
        delay = 1.0
        for _ in range(5):  # 最多重试 5 次
            try:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                # 限流/服务繁忙：读 Retry-After 或指数退避
                if status in (429, 503):
                    ra = e.response.headers.get("retry-after")
                    wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                    wait = max(1.0, min(wait, 20.0))
                    await asyncio.sleep(wait)
                    delay = min(delay * 2, 20.0)
                    continue
                # 其他交给上层（如 400/404 模型不可用、401/403 权限问题）
                raise
            except httpx.HTTPError:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 20.0)
                continue
        raise httpx.HTTPError("OpenAI服务繁忙，请稍后再试")

async def openai_chat(messages: List[Dict[str, str]]) -> str:
    preferred = [OPENAI_MODEL, "gpt-4.1", "gpt-4o", "gpt-4o-mini"]
    models_to_try: List[str] = []
    seen = set()
    for m in preferred:
        if m not in seen:
            models_to_try.append(m); seen.add(m)

    last_err: Optional[Exception] = None
    for model in models_to_try:
        try:
            return await _chat_once(model, messages)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code in (401, 403):  # Key/权限问题，换模型也多半没用
                last_err = e
                break
            if code in (400, 404):  # 模型不可用 → 换下一个
                last_err = e
                continue
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("OpenAI 调用失败（未知原因）")

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
    print("UPDATE >>>", update)  # Render Logs 调试

    msg = update.get("message") or update.get("edited_message") or update.get("channel_post")
    if not msg:
        return PlainTextResponse("ok")

    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = msg.get("text")
    message_id = msg.get("message_id")

    # 非文本兜底
    if not text:
        if chat_id is not None:
            await tg_send_message(chat_id, "我目前只支持文字消息～", reply_to=message_id)
        return PlainTextResponse("ok")

    # /start & /clear
    cmd = text.strip().lower()
    if cmd == "/start":
        await tg_send_message(chat_id, "✅ 你好！我已开启上下文记忆（最近多轮），直接发消息继续对话吧～")
        return PlainTextResponse("ok")
    if cmd == "/clear":
        _conversations.pop(chat_id, None)
        await tg_send_message(chat_id, "✅ 已清空本会话上下文。")
        return PlainTextResponse("ok")

    # 每 chat 防抖
    now = time.time()
    if now - _last_call_ts.get(chat_id, 0.0) < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "我正在处理上一条消息，请稍等片刻～", reply_to=message_id)
        return PlainTextResponse("ok")
    _last_call_ts[chat_id] = now

    # 组装 messages（含历史）
    messages = _build_messages(chat_id, text)

    try:
        reply = await openai_chat(messages)
        await tg_send_message(chat_id, reply, reply_to=message_id)
        # 写入历史
        _append_history(chat_id, "user", text)
        _append_history(chat_id, "assistant", reply)
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code == 429:
            await tg_send_message(chat_id, "⚠️ OpenAI 限流（我会退避重试）；若仍失败请稍后再试～", reply_to=message_id)
        elif code in (401, 403):
            await tg_send_message(chat_id, "❌ OpenAI API Key 或权限问题，请检查 OPENAI_API_KEY / 模型权限。", reply_to=message_id)
        elif code in (400, 404):
            await tg_send_message(chat_id, "❌ 目标模型不可用（已尝试自动切换）。请稍后重试。", reply_to=message_id)
        else:
            await tg_send_message(chat_id, f"❌ OpenAI 错误：HTTP {code}", reply_to=message_id)
    except httpx.HTTPError as e:
        await tg_send_message(chat_id, f"❌ 网络异常：{e}", reply_to=message_id)
    except Exception as e:
        await tg_send_message(chat_id, f"❌ 未知错误：{e}", reply_to=message_id)

    return PlainTextResponse("ok")
