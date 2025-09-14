import os
import asyncio
import time
import re
import json
from typing import Any, Dict, List, Deque, Optional, Tuple
from collections import deque

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

# ================== 环境变量 ==================
BOT_TOKEN       = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o")   # 默认 GPT-4o
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "mysecret123")
SERPAPI_KEY     = os.getenv("SERPAPI_KEY", "")
TIMEOUT_SEC     = int(os.getenv("TIMEOUT_SEC", "60"))

OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
CHAT_COOLDOWN_SEC      = float(os.getenv("CHAT_COOLDOWN_SEC", "1.2"))

HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "12"))      # 提升记忆轮数
HISTORY_LIFESPAN  = float(os.getenv("HISTORY_LIFESPAN_SEC", "172800"))  # 记忆 48h
STREAM_REPLY      = os.getenv("STREAM_REPLY", "1") == "1"          # 是否流式编辑输出
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML")     # HTML 渲染更稳
DISABLE_LINK_PREVIEW = os.getenv("DISABLE_LINK_PREVIEW", "1") == "1"

# ================== 常量与客户端 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

SYSTEM_PROMPT = (
    "You are ChatGPT with real-time browsing ability.\n"
    "You must write concise, helpful answers in the user's language.\n"
    "When search results are provided, rely on them primarily and synthesize a clear answer.\n"
    "Cite sources concisely when helpful.\n"
    "Prefer bullet points, code blocks and examples when applicable.\n"
)

client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
_last_call_ts: Dict[str, float] = {}

# ================== 会话存储（内存：chat_id:user_id 维度） ==================
_conversations: Dict[str, Deque[Dict[str, Any]]] = {}

def _conv_key(chat_id: int | str, user_id: int | str) -> str:
    return f"{chat_id}:{user_id}"

def _get_history(chat_id: int, user_id: int) -> Deque[Dict[str, Any]]:
    key = _conv_key(chat_id, user_id)
    if key not in _conversations:
        _conversations[key] = deque(maxlen=HISTORY_MAX_TURNS * 2)
    return _conversations[key]

def _append_history(chat_id: int, user_id: int, role: str, content: str):
    _get_history(chat_id, user_id).append({
        "role": role, "content": content, "ts": time.time()
    })

def _prune_old():
    now = time.time()
    for key, dq in list(_conversations.items()):
        _conversations[key] = deque(
            [m for m in dq if now - m.get("ts", 0) < HISTORY_LIFESPAN],
            maxlen=dq.maxlen
        )

def _build_messages(chat_id: int, user_id: int, user_text: str, search_block: Optional[str]) -> List[Dict[str, str]]:
    hist = [m for m in _get_history(chat_id, user_id) if time.time() - m.get("ts", 0) < HISTORY_LIFESPAN]
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if search_block:
        messages.append({"role": "system", "content": f"Here are fresh search notes:\n{search_block}"})
    # 组装最近对话历史（控制字符数）
    total_chars = 0
    acc: List[Dict[str, str]] = []
    for m in reversed(hist):
        c = m.get("content") or ""
        if total_chars + len(c) > 16000:
            break
        acc.append({"role": m["role"], "content": c})
        total_chars += len(c)
    messages.extend(reversed(acc))
    messages.append({"role": "user", "content": user_text})
    return messages

# ================== 工具：HTML 安全处理 & 拼接 ==================
def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )

def make_safe_html(text: str) -> str:
    # 只做基础转义，保留用户/模型中的反引号代码块为纯文本显示
    return escape_html(text)

def telegram_split(text: str, limit: int = 4096) -> List[str]:
    # Telegram 单条消息最大 4096 字符；尽量在段落边界切分
    chunks: List[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:]
    if text:
        chunks.append(text)
    return chunks

# ================== Telegram 工具 ==================
async def tg_send_chat_action(chat_id: int, action: str = "typing"):
    try:
        await client.post(f"{TELEGRAM_API}/sendChatAction", json={"chat_id": chat_id, "action": action})
    except Exception:
        pass

async def tg_send_message(chat_id: int, text: str, reply_to: Optional[int] = None) -> Dict[str, Any]:
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": TELEGRAM_PARSE_MODE,
        "disable_web_page_preview": DISABLE_LINK_PREVIEW,
    }
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    r = await client.post(f"{TELEGRAM_API}/sendMessage", json=payload)
    r.raise_for_status()
    return r.json()

async def tg_edit_message(chat_id: int, message_id: int, text: str):
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": TELEGRAM_PARSE_MODE,
        "disable_web_page_preview": DISABLE_LINK_PREVIEW,
    }
    r = await client.post(f"{TELEGRAM_API}/editMessageText", json=payload)
    r.raise_for_status()
    return r.json()

# ================== 搜索功能（SerpAPI） ==================
def _format_search_results(data: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    返回 (用于塞给模型的纯文本 block, [(title, url)] 供可选脚注)
    """
    results_txt_lines: List[str] = []
    footnotes: List[Tuple[str, str]] = []

    def add_item(title: str, snippet: str, link: str):
        if not title and not snippet:
            return
        results_txt_lines.append(f"- {title or '(no title)'}: {snippet or ''} [{link}]")
        if title and link:
            footnotes.append((title, link))

    # Organic
    for item in data.get("organic_results", [])[:5]:
        add_item(item.get("title", ""), item.get("snippet", ""), item.get("link", ""))

    # News
    for item in data.get("news_results", [])[:3]:
        add_item(item.get("title", ""), item.get("snippet", ""), item.get("link", ""))

    # Answer box
    abox = data.get("answer_box") or {}
    if abox:
        add_item(abox.get("title", "Answer"), abox.get("snippet", "") or abox.get("answer", ""), abox.get("link", ""))

    return ("\n".join(results_txt_lines) if results_txt_lines else "No fresh search results."), footnotes

async def search_web(query: str) -> Tuple[str, List[Tuple[str, str]]]:
    if not SERPAPI_KEY:
        return "Search disabled (SERPAPI_KEY not set).", []
    url = "https://serpapi.com/search"
    params = {"q": query, "hl": "zh-cn", "gl": "cn", "api_key": SERPAPI_KEY}
    try:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        return _format_search_results(data)
    except Exception as e:
        return f"Search error: {e}", []

# ================== OpenAI 调用（含流式） ==================
async def _chat_once(model: str, messages: List[Dict[str, str]], stream: bool = False):
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {"model": model, "messages": messages, "temperature": 0.7, "stream": stream}

    async with _openai_sema:
        if not stream:
            # 普通非流式
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
                        await asyncio.sleep(wait); delay = min(delay * 2, 20.0); continue
                    raise
                except httpx.HTTPError:
                    await asyncio.sleep(delay); delay = min(delay * 2, 20.0); continue
            raise httpx.HTTPError("OpenAI 服务繁忙，请稍后再试")
        else:
            # 流式：返回 async 生成器
            async def gen():
                delay = 1.0
                for attempt in range(5):
                    try:
                        async with client.stream("POST", url, headers=headers, json=body) as resp:
                            resp.raise_for_status()
                            async for line in resp.aiter_lines():
                                if not line or not line.startswith("data: "):
                                    continue
                                data = line[6:]
                                if data.strip() == "[DONE]":
                                    return
                                try:
                                    j = json.loads(data)
                                    delta = j["choices"][0]["delta"].get("content")
                                    if delta:
                                        yield delta
                                except Exception:
                                    continue
                        return
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code in (429, 503):
                            ra = e.response.headers.get("retry-after")
                            wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                            await asyncio.sleep(wait); delay = min(delay * 2, 20.0); continue
                        raise
                    except httpx.HTTPError:
                        await asyncio.sleep(delay); delay = min(delay * 2, 20.0); continue
                # 超过重试就结束
                return
            return gen()

async def openai_chat(messages: List[Dict[str, str]], use_stream: bool):
    preferred = [OPENAI_MODEL, "gpt-4o", "gpt-4o-mini"]
    for model in preferred:
        try:
            return await _chat_once(model, messages, stream=use_stream)
        except Exception:
            continue
    raise RuntimeError("OpenAI 调用失败")

# ================== 业务逻辑 ==================
def build_smart_query(text: str, hist: Deque[Dict[str, Any]]) -> str:
    text_strip = text.strip()
    if len(text_strip) >= 6:
        return text_strip
    # 短语时结合上文
    last_user = next((m["content"] for m in reversed(hist) if m["role"] == "user"), "")
    return (last_user + " " + text_strip).strip() if last_user else text_strip

def make_footnotes_html(footnotes: List[Tuple[str, str]]) -> str:
    if not footnotes:
        return ""
    lines = ["\n\n<b>参考来源</b>"]
    used = set()
    idx = 1
    for title, url in footnotes:
        if not url or url in used:
            continue
        used.add(url)
        title_safe = escape_html(title)[:120]
        url_safe = escape_html(url)
        lines.append(f"[{idx}] <a href=\"{url_safe}\">{title_safe}</a>")
        idx += 1
        if idx > 6:  # 最多 6 条
            break
    return "\n" + "\n".join(lines)

async def answer_and_stream(chat_id: int, user_id: int, text: str, reply_to: Optional[int]):
    # 限流（每个用户）
    now = time.time()
    key = _conv_key(chat_id, user_id)
    if now - _last_call_ts.get(key, 0.0) < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "我正在处理上一条消息，请稍等片刻～", reply_to=reply_to)
        return
    _last_call_ts[key] = now

    # 搜索
    hist = _get_history(chat_id, user_id)
    search_query = build_smart_query(text, hist)
    search_block, footnotes = await search_web(search_query)

    # 组装消息
    messages = _build_messages(chat_id, user_id, text, search_block)

    # 打字动作
    await tg_send_chat_action(chat_id, "typing")

    if STREAM_REPLY:
        # 先发一个占位消息
        placeholder = await tg_send_message(chat_id, "…", reply_to=reply_to)
        message_id = placeholder["result"]["message_id"]

        acc = ""
        last_edit = time.monotonic()
        edit_interval = 0.25      # 250ms 刷新频率
        max_chunk = 1200          # 累积到一定长度再编辑，减少请求
        try:
            stream = await openai_chat(messages, use_stream=True)
            async for delta in stream:
                acc += delta
                tnow = time.monotonic()
                if (tnow - last_edit > edit_interval) or (len(acc) - (len(acc) % max_chunk) != len(acc)):
                    # 安全 HTML
                    body = make_safe_html(acc)
                    # 如果超 4096，切块：前面固定，最后一块可编辑
                    parts = telegram_split(body)
                    # 先把前面的块发出去，最后一块用于 edit
                    if len(parts) > 1:
                        # 把第一块替换到当前消息
                        await tg_edit_message(chat_id, message_id, parts[0])
                        # 其余中间块用 sendMessage 发送
                        for mid in parts[1:-1]:
                            sent = await tg_send_message(chat_id, mid)
                            message_id = sent["result"]["message_id"]
                        # 最后一块回填到最后一条，继续后续编辑
                        await tg_edit_message(chat_id, message_id, parts[-1])
                    else:
                        await tg_edit_message(chat_id, message_id, body)
                    last_edit = tnow

            # 结束后补尾注
            final_body = make_safe_html(acc) + make_footnotes_html(footnotes)
            final_parts = telegram_split(final_body)
            # 把第一段编辑到当前消息
            await tg_edit_message(chat_id, message_id, final_parts[0])
            # 其余段落追加发送
            for extra in final_parts[1:]:
                await tg_send_message(chat_id, extra)

            _append_history(chat_id, user_id, "user", text)
            _append_history(chat_id, user_id, "assistant", acc)

        except Exception as e:
            # 出错降级成普通完整回复
            try:
                reply_text = await openai_chat(messages, use_stream=False)
                body = make_safe_html(reply_text) + make_footnotes_html(footnotes)
                parts = telegram_split(body)
                await tg_edit_message(chat_id, message_id, parts[0])
                for extra in parts[1:]:
                    await tg_send_message(chat_id, extra)
                _append_history(chat_id, user_id, "user", text)
                _append_history(chat_id, user_id, "assistant", reply_text)
            except Exception as e2:
                await tg_edit_message(chat_id, message_id, f"❌ 错误: {escape_html(str(e2))}")
    else:
        # 非流式：一次性发送
        try:
            reply_text = await openai_chat(messages, use_stream=False)
            body = make_safe_html(reply_text) + make_footnotes_html(footnotes)
            parts = telegram_split(body)
            first = await tg_send_message(chat_id, parts[0], reply_to=reply_to)
            for extra in parts[1:]:
                await tg_send_message(chat_id, extra)
            _append_history(chat_id, user_id, "user", text)
            _append_history(chat_id, user_id, "assistant", reply_text)
        except Exception as e:
            await tg_send_message(chat_id, f"❌ 错误: {escape_html(str(e))}", reply_to=reply_to)

# ================== FastAPI 路由 ==================
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # 24h 清理 → 提升为 12h
    async def background_tasks():
        while True:
            await asyncio.sleep(43200)
            _prune_old()
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
        tips = (
            "<b>✅ 已开启 ChatGPT 风格体验</b>\n"
            "• 实时联网搜索（自动判断是否需要）\n"
            "• 流式输出（边生成边刷新）\n"
            "• 48 小时上下文记忆，/clear 可清空\n"
            "• HTML 渲染更清晰，自动分片避免 4096 限制\n"
            "• 环境变量 STREAM_REPLY=1 可开关流式\n"
        )
        await tg_send_message(chat_id, tips)
        return PlainTextResponse("ok")

    if cmd == "/clear":
        _conversations.pop(_conv_key(chat_id, user_id), None)
        await tg_send_message(chat_id, "✅ 已清空你的上下文。")
        return PlainTextResponse("ok")

    # 主逻辑：回答并流式输出
    await answer_and_stream(chat_id, user_id, text, reply_to=message_id)
    return PlainTextResponse("ok")
