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

HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "12"))
# 改为 24 小时记忆
HISTORY_LIFESPAN  = float(os.getenv("HISTORY_LIFESPAN_SEC", "86400"))

# 统一取消流式输出（固定为 False）
STREAM_REPLY      = False
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML")
DISABLE_LINK_PREVIEW = os.getenv("DISABLE_LINK_PREVIEW", "1") == "1"

# 输出净化（去星号/项目符号/重复标点，代码块除外）
SANITIZE_OUTPUT   = os.getenv("SANITIZE_OUTPUT", "1") == "1"

# ================== 常量与客户端 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# 强化：始终基于搜索结果回答
SYSTEM_PROMPT = (
    "You are ChatGPT with real-time browsing ability.\n"
    "Always answer based ONLY on the latest search notes provided below.\n"
    "Use plain sentences with simple punctuation. Do not use bullets, asterisks, Markdown, or emojis unless the user asks.\n"
    "If the search notes are insufficient, say so briefly and keep the answer concise.\n"
    "Respond in the user's language."
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
    # 总是把搜索结果作为系统补充（每次自动联网）
    messages.append({"role": "system", "content": f"Here are the latest search notes:\n{search_block or 'No search results.'}"})
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
    return escape_html(text)

def telegram_split(text: str, limit: int = 4096) -> List[str]:
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

# ================== 输出清理（去星号/项目符号/Markdown符号/重复标点） ==================
_PUNCT_BULLETS = r"[*•·▪◦◆◇★☆✦✧❖▶►▷◁▸▹◀◁➤➔→⇒➜➤]"
_MD_MARKS      = r"[`_#>]"

def _sanitize_plain_text(text: str) -> str:
    # 避免破坏代码块
    if "```" in text:
        return text
    # 行首项目符号
    text = re.sub(r"(?m)^\s*[-–—"+_PUNCT_BULLETS+r"]\s+", "", text)
    # 移除 Markdown 符号与项目符号字符
    text = re.sub(_MD_MARKS, "", text)
    text = re.sub(_PUNCT_BULLETS, "", text)
    # 统一引号
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    # 合并重复标点
    text = re.sub(r"([!！?？.,，。;；:：\-—])\1{1,}", r"\1", text)
    # 去掉行首多余符号
    text = re.sub(r"(?m)^[\-\—\·\•\*]+\s*", "", text)
    return text.strip()

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
        # 用简单行文，不加花哨符号
        line = f"{title or '(no title)'}: {snippet or ''} [{link}]".strip()
        results_txt_lines.append(line)
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
        # 强制联网搜索：未配置时给出明确提示
        return "Search disabled (SERPAPI_KEY not set). Please set SERPAPI_KEY to enable live browsing.", []
    url = "https://serpapi.com/search"
    params = {"q": query, "hl": "zh-cn", "gl": "cn", "api_key": SERPAPI_KEY}
    try:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        return _format_search_results(data)
    except Exception as e:
        return f"Search error: {e}", []

# ================== OpenAI 调用（非流式） ==================
async def _chat_once(model: str, messages: List[Dict[str, str]]):
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

async def openai_chat(messages: List[Dict[str, str]]):
    preferred = [OPENAI_MODEL, "gpt-4o", "gpt-4o-mini"]
    for model in preferred:
        try:
            return await _chat_once(model, messages)
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
    lines = ["\n\n参考来源"]
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
        if idx > 6:
            break
    return "\n" + "\n".join(lines)

async def answer_once(chat_id: int, user_id: int, text: str, reply_to: Optional[int]):
    # 限流（每个用户）
    now = time.time()
    key = _conv_key(chat_id, user_id)
    if now - _last_call_ts.get(key, 0.0) < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "我正在处理上一条消息，请稍等片刻～", reply_to=reply_to)
        return
    _last_call_ts[key] = now

    # 每次都联网搜索
    hist = _get_history(chat_id, user_id)
    search_query = build_smart_query(text, hist)
    search_block, footnotes = await search_web(search_query)

    # 构造上下文并一次性生成
    messages = _build_messages(chat_id, user_id, text, search_block)
    try:
        reply_text = await openai_chat(messages)
        final_out = _sanitize_plain_text(reply_text) if SANITIZE_OUTPUT else reply_text
        body = make_safe_html(final_out) + make_footnotes_html(footnotes)
        parts = telegram_split(body)
        await tg_send_message(chat_id, parts[0], reply_to=reply_to)
        for extra in parts[1:]:
            await tg_send_message(chat_id, extra)
        _append_history(chat_id, user_id, "user", text)
        _append_history(chat_id, user_id, "assistant", reply_text)
    except Exception as e:
        await tg_send_message(chat_id, f"错误: {escape_html(str(e))}", reply_to=reply_to)

# ================== FastAPI 路由 ==================
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    async def background_tasks():
        while True:
            await asyncio.sleep(3600)  # 每小时清一次过期上下文
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
        await tg_send_message(chat_id, "我目前只支持文字消息", reply_to=message_id)
        return PlainTextResponse("ok")

    cmd = text.strip().lower()
    if cmd == "/start":
        tips = (
            "已开启：自动联网搜索 + 一次性完整回复 + 24小时上下文记忆\n"
            "输入 /clear 可清空上下文"
        )
        await tg_send_message(chat_id, tips)
        return PlainTextResponse("ok")

    if cmd == "/clear":
        _conversations.pop(_conv_key(chat_id, user_id), None)
        await tg_send_message(chat_id, "已清空你的上下文")
        return PlainTextResponse("ok")

    # 主逻辑：每次联网 + 非流式一次性回复
    await answer_once(chat_id, user_id, text, reply_to=message_id)
    return PlainTextResponse("ok")
