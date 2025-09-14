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
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o")
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "mysecret123")
SERPAPI_KEY     = os.getenv("SERPAPI_KEY", "")
TIMEOUT_SEC     = int(os.getenv("TIMEOUT_SEC", "60"))

OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
CHAT_COOLDOWN_SEC      = float(os.getenv("CHAT_COOLDOWN_SEC", "1.2"))

HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "12"))
HISTORY_LIFESPAN  = float(os.getenv("HISTORY_LIFESPAN_SEC", "86400"))   # 24 小时

# 输出风格与长度控制
DETAIL_LEVEL   = os.getenv("DETAIL_LEVEL", "deep")        # brief / standard / deep
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "1400"))     # 回答最大 token
STRICT_MODEL   = os.getenv("STRICT_MODEL", "0") == "1"    # 为 1 时禁止降级

# 搜索厚度与时效
SEARCH_MAX_ORGANIC = int(os.getenv("SEARCH_MAX_ORGANIC", "8"))
SEARCH_MAX_NEWS    = int(os.getenv("SEARCH_MAX_NEWS", "5"))
SEARCH_RECENCY     = os.getenv("SEARCH_RECENCY", "")      # "", "h" 小时, "d" 日, "w" 周, "m" 月
SEARCH_HL          = os.getenv("SEARCH_HL", "zh-cn")      # 搜索语言
SEARCH_GL          = os.getenv("SEARCH_GL", "us")         # 搜索地域

# 文本渲染与净化
TELEGRAM_PARSE_MODE   = os.getenv("TELEGRAM_PARSE_MODE", "HTML")
DISABLE_LINK_PREVIEW  = os.getenv("DISABLE_LINK_PREVIEW", "1") == "1"
SANITIZE_OUTPUT       = os.getenv("SANITIZE_OUTPUT", "1") == "1"

# ================== 常量与客户端 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

def _style_instruction() -> str:
    if DETAIL_LEVEL == "brief":
        return "Write 4-6 full sentences in 1-2 short paragraphs."
    if DETAIL_LEVEL == "standard":
        return "Write 8-12 sentences across 2-4 paragraphs with clear structure."
    return ("Write 12-20 sentences across 3-5 short paragraphs. "
            "Start with quick context, then the latest updates from sources, include nuance and uncertainties, "
            "end with what to watch next. Plain text paragraphs only.")

SYSTEM_PROMPT = (
    "You are ChatGPT with real-time browsing ability.\n"
    "Primarily rely on the fresh search notes below, but you may add stable background facts for context. "
    "Clearly separate background vs new updates when useful.\n"
    f"{_style_instruction()}\n"
    "Use clean plain sentences. Do not use bullets, asterisks, Markdown decorations, or emojis."
)

client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
_last_call_ts: Dict[str, float] = {}

# 跟踪信息（调试/状态）
_last_model_used: Dict[str, str] = {}
_last_search_query: Dict[str, str] = {}

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
    _get_history(chat_id, user_id).append({"role": role, "content": content, "ts": time.time()})

def _prune_old():
    now = time.time()
    for key, dq in list(_conversations.items()):
        _conversations[key] = deque(
            [m for m in dq if now - m.get("ts", 0) < HISTORY_LIFESPAN],
            maxlen=dq.maxlen
        )

def _build_messages(chat_id: int, user_id: int, user_text: str, search_block: str) -> List[Dict[str, str]]:
    hist = [m for m in _get_history(chat_id, user_id) if time.time() - m.get("ts", 0) < HISTORY_LIFESPAN]
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "system", "content": f"Here are the latest search notes:\n{search_block or 'No search results.'}"})
    # 控制历史拼接长度
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

# ================== HTML 与文本净化 ==================
def escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

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

_PUNCT_BULLETS = r"[*•·▪◦◆◇★☆✦✧❖▶►▷◁▸▹◀◁➤➔→⇒➜➤]"
_MD_MARKS      = r"[`_#>]"

def _sanitize_plain_text(text: str) -> str:
    # 不破坏代码块
    if "```" in text:
        return text
    t = text
    t = re.sub(r"(?m)^\s*[-–—"+_PUNCT_BULLETS+r"]\s+", "", t)  # 行首项目符号
    t = re.sub(_MD_MARKS, "", t)                              # Markdown 装饰
    t = re.sub(_PUNCT_BULLETS, "", t)                         # 其他符号
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r"([!！?？.,，。;；:：\-—])\1{1,}", r"\1", t)    # 合并重复标点
    t = re.sub(r"(?m)^[\-\—\·\•\*]+\s*", "", t)               # 去行首多余符号
    return t.strip()

# ================== Telegram 工具 ==================
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
    results_txt_lines: List[str] = []
    footnotes: List[Tuple[str, str]] = []

    def add_item(title: str, snippet: str, link: str):
        if not title and not snippet:
            return
        line = f"{title or '(no title)'}: {snippet or ''} [{link}]".strip()
        results_txt_lines.append(line)
        if title and link:
            footnotes.append((title, link))

    for item in data.get("organic_results", [])[:SEARCH_MAX_ORGANIC]:
        add_item(item.get("title", ""), item.get("snippet", ""), item.get("link", ""))
    for item in data.get("news_results", [])[:SEARCH_MAX_NEWS]:
        add_item(item.get("title", ""), item.get("snippet", ""), item.get("link", ""))

    abox = data.get("answer_box") or {}
    if abox:
        add_item(abox.get("title", "Answer"), abox.get("snippet", "") or abox.get("answer", ""), abox.get("link", ""))

    return ("\n".join(results_txt_lines) if results_txt_lines else "No fresh search results."), footnotes

async def search_web(query: str) -> Tuple[str, List[Tuple[str, str]]]:
    if not SERPAPI_KEY:
        return "Search disabled (SERPAPI_KEY not set). Please set SERPAPI_KEY to enable live browsing.", []
    url = "https://serpapi.com/search"
    params = {"q": query, "hl": SEARCH_HL, "gl": SEARCH_GL, "api_key": SERPAPI_KEY}
    if SEARCH_RECENCY:
        params["tbs"] = f"qdr:{SEARCH_RECENCY}"   # 例：qdr:w（最近一周）
    try:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        return _format_search_results(data)
    except Exception as e:
        return f"Search error: {e}", []

# ================== OpenAI 调用（一次性非流式） ==================
async def _chat_once(model: str, messages: List[Dict[str, str]]):
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {"model": model, "messages": messages, "temperature": 0.5, "max_tokens": MAX_TOKENS}

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

async def openai_chat(messages: List[Dict[str, str]], key: str = "global"):
    models = [OPENAI_MODEL] if STRICT_MODEL else [OPENAI_MODEL, "gpt-4o", "gpt-4o-mini"]
    for m in models:
        try:
            out = await _chat_once(m, messages)
            _last_model_used[key] = m
            return out
        except Exception:
            continue
    raise RuntimeError("OpenAI 调用失败")

# ================== 业务逻辑 ==================
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

def build_smart_query(text: str, hist: Deque[Dict[str, Any]], key: str) -> str:
    """
    仅做最小上下文补全：若本次消息过短（<=6 字符），拼接上一条用户提问。
    不做任何人物/地名/关键词的替换或纠错。
    """
    t = (text or "").strip()
    if len(t) > 6:
        return t
    last_user = next((m["content"] for m in reversed(hist) if m.get("role") == "user"), "")
    return (last_user + " " + t).strip() if last_user else t

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
    search_query = build_smart_query(text, hist, key)
    _last_search_query[key] = search_query
    search_block, footnotes = await search_web(search_query)

    # 组装消息并一次性生成
    messages = _build_messages(chat_id, user_id, text, search_block)
    try:
        reply_text = await openai_chat(messages, key=key)
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
            "✅已开启：自动联网搜索 + 一次性完整回复 + 24小时上下文记忆"
        )
        await tg_send_message(chat_id, tips)
        return PlainTextResponse("ok")

    if cmd == "/clear":
        _conversations.pop(_conv_key(chat_id, user_id), None)
        await tg_send_message(chat_id, "已清空你的上下文")
        return PlainTextResponse("ok")

    if cmd == "/status":
        key = _conv_key(chat_id, user_id)
        serp = "已配置" if SERPAPI_KEY else "未配置"
        model_used = _last_model_used.get(key, "(尚未调用)")
        last_q = _last_search_query.get(key, "(无)")
        await tg_send_message(
            chat_id,
            f"SerpAPI：{serp}\n最近使用模型：{model_used}\n最近搜索词：{last_q}\n"
            f"记忆窗口：24小时；历史上限：{HISTORY_MAX_TURNS} 轮\n"
            f"细节级别：{DETAIL_LEVEL}；max_tokens：{MAX_TOKENS}\n"
            f"搜索厚度：organic={SEARCH_MAX_ORGANIC}, news={SEARCH_MAX_NEWS}, recency={SEARCH_RECENCY or '不限'}"
        )
        return PlainTextResponse("ok")

    # 主逻辑：每次联网 + 非流式一次性回复
    await answer_once(chat_id, user_id, text, reply_to=message_id)
    return PlainTextResponse("ok")
