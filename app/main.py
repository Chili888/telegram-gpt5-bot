# app/main.py
import os
import asyncio
import time
from typing import Any, Dict, List, Deque, Optional, Tuple
from collections import deque

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

# ================== 环境变量 ==================
BOT_TOKEN       = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4.1")  # 优先模型（自动降级）
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "dev-secret")
SERPAPI_KEY     = os.getenv("SERPAPI_KEY", "")
TIMEOUT_SEC     = int(os.getenv("TIMEOUT_SEC", "60"))

# 并发 / 防抖
OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
CHAT_COOLDOWN_SEC      = float(os.getenv("CHAT_COOLDOWN_SEC", "1.2"))

# 上下文窗口
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "12"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "24000"))

# ================== 常量与客户端 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
SYSTEM_PROMPT = (
    "You are ChatGPT with web access. Use the retrieved web snippets ONLY as context; "
    "reason over the whole conversation and answer concisely, in the user's language. "
    "If web info conflicts, explain uncertainties briefly. Cite inline with short names when useful."
)

client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
_last_call_ts: Dict[int, float] = {}

# 会话：按 (chat_id, user_id) 保存，防止群聊串线
_conversations: Dict[Tuple[int, Optional[int]], Deque[Dict[str, str]]] = {}
# 记录每个会话最近一次“检索主题”，便于“继续/最新”等延续主题
_last_query: Dict[Tuple[int, Optional[int]], str] = {}

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
    return (chat_id, user_id)

def _get_history(sk: Tuple[int, Optional[int]]) -> Deque[Dict[str, str]]:
    if sk not in _conversations:
        _conversations[sk] = deque(maxlen=HISTORY_MAX_TURNS * 2)  # 一轮=两条
    return _conversations[sk]

def _append_history(sk: Tuple[int, Optional[int]], role: str, content: str):
    _get_history(sk).append({"role": role, "content": content})

def _build_messages(sk: Tuple[int, Optional[int]], user_text: str) -> List[Dict[str, str]]:
    hist = list(_get_history(sk))
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    acc: List[Dict[str, str]] = []
    total = 0
    for m in reversed(hist):
        c = m.get("content") or ""
        if total + len(c) > HISTORY_MAX_CHARS:
            break
        acc.append(m); total += len(c)
    msgs.extend(reversed(acc))
    msgs.append({"role": "user", "content": user_text})
    return msgs

# ================== 生成“上下文检索主题” ==================
# 一些应答词，不触发新主题
ACK_WORDS = {"了解","好的","OK","ok","明白","收到","行","嗯","是的","谢谢","多谢"}
FOLLOW_WORDS = {"继续","更多","再多点","扩展","拓展","延伸","详解","详细","最新","动态","新闻","近况","近况如何","最近"}
def _build_contextual_query(sk: Tuple[int, Optional[int]], user_text: str) -> Optional[str]:
    """
    基于【上下文 + 本次输入】生成一个较稳的检索主题：
    - 纯应答（如“了解”）→ 优先沿用上次主题；无上次主题则返回 None
    - 包含“继续/最新/更多”等指令 → 在上次主题基础上加关键字
    - 正常问题 → 取最近若干条 user/assistant 内容拼一个简洁 query
    """
    text = (user_text or "").strip()

    # 1) 应答词：沿用上次主题
    if text in ACK_WORDS:
        return _last_query.get(sk)

    # 2) “继续/最新/更多/动态”等：基于上次主题扩展
    if any(w in text for w in FOLLOW_WORDS) and _last_query.get(sk):
        base = _last_query[sk]
        if "最新" in text or "动态" in text or "新闻" in text or "最近" in text:
            return f"{base} 最新 动态 新闻"
        return f"{base} 资料 扩展"

    # 3) 正常问题：从历史抽取上下文要点
    hist = list(_get_history(sk))[-6:]  # 最近 6 条（3轮）
    context_bits: List[str] = []
    for m in hist:
        c = (m.get("content") or "").strip()
        if c and len(c) > 2:
            context_bits.append(c)
    context_bits.append(text)

    # 简单裁剪到 ~400 字以内，避免过长
    joined = " ".join(context_bits)[-400:]
    # 给检索加一个“最新”的偏好，让搜索倾向于新信息
    query = f"{joined} 最新"
    return query

# ================== 联网搜索（SerpAPI） ==================
async def web_search(query: str, num: int = 5) -> str:
    """
    调用 SerpAPI，返回简短的多条摘要（标题 + 摘要 + 链接），用于贴给模型。
    返回纯文本，模型在上下文中使用。
    """
    if not SERPAPI_KEY or not query:
        return ""
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "num": num,
        "hl": "zh-cn",
        "gl": "cn",
        "api_key": SERPAPI_KEY,
    }
    try:
        r = await client.get(url, params=params, timeout=TIMEOUT_SEC)
        r.raise_for_status()
        js = r.json()
        bullets: List[str] = []
        # 优先用 answer_box
        ab = js.get("answer_box") or {}
        ab_text = ab.get("answer") or ab.get("snippet") or ""
        if ab_text:
            bullets.append(f"【直答】{ab_text}")

        # 再取有代表性的 organic_results
        for item in (js.get("organic_results") or [])[:3]:
            title = item.get("title", "").strip()
            snippet = item.get("snippet", "").strip()
            link = item.get("link", "").strip()
            if title or snippet:
                bullets.append(f"• {title}\n{snippet}\n{link}")
        return "\n\n".join(bullets).strip()
    except Exception as e:
        # 静默失败，回退到纯模型
        print("web_search error:", e)
        return ""

# ================== OpenAI 调用（自动降级 + 退避重试） ==================
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
                sc = e.response.status_code
                if sc in (429, 503):
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
    prefer = [OPENAI_MODEL, "gpt-4.1", "gpt-4o", "gpt-4o-mini"]
    tried = set()
    last_err: Optional[Exception] = None
    for m in prefer:
        if m in tried:
            continue
        tried.add(m)
        try:
            return await _chat_once(m, messages)
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("OpenAI 调用失败")

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
    sk = _session_key(chat_id, user_id)

    text = msg.get("text")
    message_id = msg.get("message_id")

    if not text:
        await tg_send_message(chat_id, "我目前只支持文字消息～", reply_to=message_id)
        return PlainTextResponse("ok")

    # 命令
    cmd = text.strip().lower()
    if cmd == "/start":
        await tg_send_message(
            chat_id,
            "✅ 你好！我会在回答前自动参考互联网最新信息，并结合上下文连续作答。\n"
            "提示：发送“继续 / 最新 / 更多”等短语，我会基于上个话题继续联网检索。\n"
            "发送 /clear 可清空你在此群的个人上下文。"
        )
        return PlainTextResponse("ok")
    if cmd == "/clear":
        _conversations.pop(sk, None)
        _last_query.pop(sk, None)
        await tg_send_message(chat_id, "✅ 已清空你在此群的个人上下文。")
        return PlainTextResponse("ok")

    # 防抖
    now = time.time()
    if now - _last_call_ts.get(chat_id, 0.0) < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "⏳ 我正在处理上一条消息，请稍等～", reply_to=message_id)
        return PlainTextResponse("ok")
    _last_call_ts[chat_id] = now

    # ===== 生成“上下文检索主题”，联网抓取 =====
    query = _build_contextual_query(sk, text)
    web_snippets = ""
    if query:
        web_snippets = await web_search(query)
        if web_snippets:
            _last_query[sk] = query  # 更新最近主题

    # ===== 组装给模型的 messages =====
    # 先把用户原话写入历史，以便下一轮使用（也可以放到成功后写入）
    # 这里选择成功后写，避免失败污染历史
    msgs = _build_messages(sk, text)

    # 将联网结果注入上下文（system 提供“参考材料”）
    if web_snippets:
        msgs.append({
            "role": "system",
            "content": f"以下是与当前话题相关的互联网最新检索摘录（仅供参考，可能不完整或有偏差）：\n{web_snippets}"
        })

    # ===== 调用 GPT 生成回答 =====
    try:
        reply = await openai_chat(msgs)
        await tg_send_message(chat_id, reply, reply_to=message_id)
        _append_history(sk, "user", text)
        _append_history(sk, "assistant", reply)
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code == 429:
            await tg_send_message(chat_id, "⚠️ OpenAI 限流，请稍后再试～", reply_to=message_id)
        elif code in (401, 403):
            await tg_send_message(chat_id, "❌ OpenAI API Key 或权限问题。", reply_to=message_id)
        elif code in (400, 404):
            await tg_send_message(chat_id, "❌ 模型不可用（已尝试自动切换）。", reply_to=message_id)
        else:
            await tg_send_message(chat_id, f"❌ OpenAI 错误：HTTP {code}", reply_to=message_id)
    except httpx.HTTPError as e:
        await tg_send_message(chat_id, f"❌ 网络异常：{e}", reply_to=message_id)
    except Exception as e:
        await tg_send_message(chat_id, f"❌ 未知错误：{e}", reply_to=message_id)

    return PlainTextResponse("ok")
