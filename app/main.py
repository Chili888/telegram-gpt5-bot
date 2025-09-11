# app/main.py
import os
import asyncio
import time
import json
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from redis.asyncio import Redis

# ================== 环境变量 ==================
BOT_TOKEN       = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4.1")  # 首选；不可用时自动降级
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "dev-secret")
TIMEOUT_SEC     = int(os.getenv("TIMEOUT_SEC", "60"))

# 并发/防抖
OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
CHAT_COOLDOWN_SEC      = float(os.getenv("CHAT_COOLDOWN_SEC", "1.2"))

# 会话记忆（每会话最近 N 轮；粗略字符上限）
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "8"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "16000"))

# Redis
REDIS_URL       = os.getenv("REDIS_URL", "")  # 形如：redis://:password@host:port/0 或 rediss://
REDIS_PREFIX    = os.getenv("REDIS_PREFIX", "tg:conv")  # key 前缀

# ================== 常量与客户端 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
SYSTEM_PROMPT = (
    "You are ChatGPT. Be concise, helpful, and safe. "
    "Respond in the same language the user used."
)

# 全局 http 客户端、并发闸
client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
_last_call_ts: Dict[int, float] = {}

# 全局 Redis 客户端
redis: Optional[Redis] = None

app = FastAPI()


# ================== Telegram 工具 ==================
async def tg_send_message(chat_id: int, text: str, reply_to: Optional[int] = None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    r = await client.post(f"{TELEGRAM_API}/sendMessage", json=payload)
    r.raise_for_status()
    return r.json()


# ================== 会话记忆（Redis） ==================
def _conv_key(chat_id: int) -> str:
    # 列表（LPUSH 时间倒序，读取时再反转）
    return f"{REDIS_PREFIX}:{chat_id}"

async def history_append(chat_id: int, role: str, content: str):
    """把一条消息写入 Redis 列表开头，并裁剪到最大条数（2 * HISTORY_MAX_TURNS）"""
    if not redis:
        return
    key = _conv_key(chat_id)
    data = json.dumps({"role": role, "content": content}, ensure_ascii=False)
    # LPUSH -> 最新在左；LTRIM 保留 2*N 条（user/assistant 各一条为一轮）
    await redis.lpush(key, data)
    await redis.ltrim(key, 0, HISTORY_MAX_TURNS * 2 - 1)

async def history_clear(chat_id: int):
    if not redis:
        return
    await redis.delete(_conv_key(chat_id))

async def build_messages(chat_id: int, user_text: str) -> List[Dict[str, str]]:
    """
    从 Redis 取出最近若干条（最多 2*N），反转为正序，并按字符上限做粗裁剪。
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if not redis:
        messages.append({"role": "user", "content": user_text})
        return messages

    key = _conv_key(chat_id)
    raw_items = await redis.lrange(key, 0, HISTORY_MAX_TURNS * 2 - 1)  # bytes list（最新在前）
    items: List[Dict[str, str]] = []
    total = 0
    # 反向遍历（旧→新），直到达到字符阈值
    for b in reversed(raw_items or []):
        try:
            m = json.loads(b)
            c = (m.get("content") or "")
            if total + len(c) > HISTORY_MAX_CHARS:
                break
            items.append({"role": m.get("role") or "user", "content": c})
            total += len(c)
        except Exception:
            continue

    messages.extend(items)
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
            if code in (401, 403):  # key/权限问题（换模型多半无效）
                last_err = e
                break
            if code in (400, 404):  # 模型不可用 → 尝试下一个
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
    # 简单验证 Redis 连通性（不阻塞）
    ok = "ok"
    try:
        if redis:
            await redis.ping()
    except Exception:
        ok = "redis_error"
    return PlainTextResponse(ok)

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

    # 非文本兜底
    if not text:
        if chat_id is not None:
            await tg_send_message(chat_id, "我目前只支持文字消息～", reply_to=message_id)
        return PlainTextResponse("ok")

    # /start & /clear
    cmd = text.strip().lower()
    if cmd == "/start":
        await tg_send_message(chat_id, "✅ 你好！我已开启持久化的上下文记忆（保留最近多轮）。直接发消息继续对话吧～")
        return PlainTextResponse("ok")
    if cmd == "/clear":
        await history_clear(chat_id)
        await tg_send_message(chat_id, "✅ 已清空本会话上下文（Redis）。")
        return PlainTextResponse("ok")

    # 每 chat 防抖
    now = time.time()
    if now - _last_call_ts.get(chat_id, 0.0) < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "我正在处理上一条消息，请稍等片刻～", reply_to=message_id)
        return PlainTextResponse("ok")
    _last_call_ts[chat_id] = now

    # 组装 messages（含 Redis 历史）
    messages = await build_messages(chat_id, text)

    try:
        reply = await openai_chat(messages)
        await tg_send_message(chat_id, reply, reply_to=message_id)
        # 写入历史（成功后再写）
        await history_append(chat_id, "user", text)
        await history_append(chat_id, "assistant", reply)
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


# ================== 启动/关闭钩子（连接 Redis） ==================
@app.on_event("startup")
async def on_startup():
    global redis
    if REDIS_URL:
        try:
            redis = Redis.from_url(REDIS_URL, decode_responses=False)  # 存字节更稳
            await redis.ping()
            print("Redis connected")
        except Exception as e:
            print("Redis connect failed:", e)
            redis = None
    else:
        print("No REDIS_URL provided; context memory will NOT persist.")

@app.on_event("shutdown")
async def on_shutdown():
    try:
        if redis:
            await redis.close()
    except Exception:
        pass
    try:
        await client.aclose()
    except Exception:
        pass
