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

# 输出风格与长度
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "1400"))
SANITIZE_OUTPUT= os.getenv("SANITIZE_OUTPUT", "1") == "1"
TELEGRAM_PARSE_MODE   = os.getenv("TELEGRAM_PARSE_MODE", "HTML")
DISABLE_LINK_PREVIEW  = os.getenv("DISABLE_LINK_PREVIEW", "1") == "1"

# 搜索增强（推荐根据受众调整）
SEARCH_HL            = os.getenv("SEARCH_HL", "zh-cn")           # 搜索语言
SEARCH_GL            = os.getenv("SEARCH_GL", "hk")              # 搜索地域
SEARCH_GOOGLE_DOMAIN = os.getenv("SEARCH_GOOGLE_DOMAIN", "google.com.hk")
SEARCH_LOCATION      = os.getenv("SEARCH_LOCATION", "Hong Kong") # 地理偏好
SEARCH_LR            = os.getenv("SEARCH_LR", "lang_zh-CN")
SEARCH_CR            = os.getenv("SEARCH_CR", "countryHK")
SEARCH_RECENCY       = os.getenv("SEARCH_RECENCY", "w")          # google: h/d/w/m
USE_NEWS_ENGINE      = os.getenv("USE_NEWS_ENGINE", "1") == "1"
NEWS_RECENCY         = os.getenv("NEWS_RECENCY", "7d")           # google_news: 1h/1d/7d/1m
SEARCH_NUM           = int(os.getenv("SEARCH_NUM", "10"))        # 合并后最多取前 N 条

# 严格模型：为 1 时不降级
STRICT_MODEL   = os.getenv("STRICT_MODEL", "0") == "1"

# ================== 常量与客户端 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

SYSTEM_PROMPT = (
    "You are ChatGPT with real-time browsing ability.\n"
    "Answer ONLY using the search notes below. Do not add background facts not present in the notes. "
    "If the notes are insufficient to answer, say so briefly. "
    "Respond in the user's language with clean plain sentences (no bullets/markdown/emojis)."
)

client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
_last_call_ts: Dict[str, float] = {}

# 运行状态
_last_model_used: Dict[str, str] = {}
_last_search_query: Dict[str, str] = {}
_last_search_ms: Dict[str, int] = {}
_last_error: Dict[str, str] = {}

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
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Latest search notes:\n{search_block or 'No results.'}"}
    ]
    total = 0
    acc: List[Dict[str, str]] = []
    for m in reversed(hist):
        c = m.get("content") or ""
        if total + len(c) > 16000:
            break
        acc.append({"role": m["role"], "content": c})
        total += len(c)
    msgs.extend(reversed(acc))
    msgs.append({"role": "user", "content": user_text})
    return msgs

# ================== 文本与 Telegram ==================
def escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _sanitize_plain_text(text: str) -> str:
    if "```" in text:
        return text
    t = text
    t = re.sub(r"(?m)^\s*[-*•·]+\s+", "", t)
    t = re.sub(r"[`_#>]", "", t)
    t = re.sub(r"([!！?？.,，。;；:：\-—])\1{1,}", r"\1", t)
    return t.strip()

def telegram_split(text: str, limit: int = 4096) -> List[str]:
    parts: List[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:]
    if text:
        parts.append(text)
    return parts

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

# ================== 搜索功能（SerpAPI：网页 + 新闻 并行） ==================
_TRUSTED_DOMAINS = [
    "reuters.com","apnews.com","bloomberg.com","ft.com","wsj.com",
    "bbc.com","cnn.com","nytimes.com","theguardian.com",
    "caixin.com","scmp.com","xinhuanet.com","cctv.com","china.com.cn",
]
def _domain_score(url: str) -> int:
    try:
        host = re.sub(r"^https?://", "", url).split("/")[0].lower()
    except Exception:
        return 999
    for i, dom in enumerate(_TRUSTED_DOMAINS):
        if host.endswith(dom):
            return i
    return 500

def _format_items(items: List[Dict[str, Any]], is_news: bool) -> List[Tuple[str,str,str,str]]:
    out: List[Tuple[str,str,str,str]] = []
    for it in items:
        title = it.get("title", "")
        snippet = it.get("snippet", "") or it.get("content", "")
        link = it.get("link", "") or it.get("url", "")
        date  = it.get("date", "") if is_news else (it.get("date","") or it.get("published_at",""))
        if title or snippet:
            out.append((title, snippet, link, date))
    return out

def _merge_and_rank(web: List[Tuple[str,str,str,str]], news: List[Tuple[str,str,str,str]]) -> str:
    seen = set()
    merged: List[Tuple[str,str,str,str]] = []
    for bucket in (news, web):  # 新闻优先
        for t,s,link,d in bucket:
            key = (t.strip(), link.strip())
            if not link or key in seen:
                continue
            seen.add(key)
            merged.append((t,s,link,d))
    merged.sort(key=lambda x: _domain_score(x[2]))
    lines = []
    for t,s,link,d in merged[:SEARCH_NUM]:
        ds = f" ({d})" if d else ""
        lines.append(f"{t}{ds}: {s} [{link}]")
    return "\n".join(lines) if lines else "No fresh search results."

async def search_web(raw_query: str) -> str:
    if not SERPAPI_KEY:
        return "Search disabled: SERPAPI_KEY not set."
    q = raw_query.strip()
    if not any(k in q for k in ["最新","动态","新闻","进展","today","latest","update","news","breaking"]):
        q = f"{q} 最新 进展"

    common = {"hl": SEARCH_HL, "gl": SEARCH_GL, "api_key": SERPAPI_KEY, "num": SEARCH_NUM}
    if SEARCH_GOOGLE_DOMAIN: common["google_domain"] = SEARCH_GOOGLE_DOMAIN
    if SEARCH_LOCATION:      common["location"] = SEARCH_LOCATION
    if SEARCH_LR:            common["lr"] = SEARCH_LR
    if SEARCH_CR:            common["cr"] = SEARCH_CR
    if SEARCH_RECENCY:       common["tbs"] = f"qdr:{SEARCH_RECENCY}"

    t0 = time.monotonic()
    reqs = []
    reqs.append(client.get("https://serpapi.com/search", params={"engine":"google","q":q, **common}))
    if USE_NEWS_ENGINE:
        news_params = {"engine":"google_news","q":q,"when":NEWS_RECENCY, **{k:v for k,v in common.items() if k!="tbs"}}
        reqs.append(client.get("https://serpapi.com/search", params=news_params))
    resp = await asyncio.gather(*reqs, return_exceptions=True)
    t1 = time.monotonic()

    web_items: List[Tuple[str,str,str,str]] = []
    news_items: List[Tuple[str,str,str,str]] = []

    for r in resp:
        if isinstance(r, Exception):
            continue
        try:
            data = r.json()
        except Exception:
            continue
        engine = (data.get("search_metadata", {}) or {}).get("engine", "")
        if engine == "google_news":
            news_items += _format_items(data.get("news_results", []), True)
        else:
            web_items  += _format_items(data.get("organic_results", []), False)
            abox = data.get("answer_box") or {}
            if abox:
                news_items.append((
                    abox.get("title","Answer"),
                    abox.get("snippet","") or abox.get("answer",""),
                    abox.get("link","") or "",
                    ""
                ))

    notes = _merge_and_rank(web_items, news_items)
    return notes, int((t1 - t0)*1000)

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
                if e.response.status_code in (429, 503):
                    ra = e.response.headers.get("retry-after")
                    wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                    await asyncio.sleep(wait); delay = min(delay * 2, 20.0); continue
                raise
            except httpx.HTTPError:
                await asyncio.sleep(delay); delay = min(delay * 2, 20.0); continue
        raise httpx.HTTPError("OpenAI 服务繁忙，请稍后再试")

async def openai_chat(messages: List[Dict[str, str]], key: str):
    models = [OPENAI_MODEL] if STRICT_MODEL else [OPENAI_MODEL, "gpt-4o", "gpt-4o-mini"]
    for m in models:
        try:
            out = await _chat_once(m, messages)
            _last_model_used[key] = m
            return out
        except Exception as e:
            _last_error[key] = f"openai:{e}"
            continue
    raise RuntimeError("OpenAI 调用失败")

# ================== 业务逻辑 ==================
def build_smart_query(text: str, hist: Deque[Dict[str, Any]]) -> str:
    t = (text or "").strip()
    if len(t) > 6:
        return t
    last_user = next((m["content"] for m in reversed(hist) if m.get("role") == "user"), "")
    base = (last_user + " " + t).strip() if last_user else t
    return base or "最新"

async def answer_once(chat_id: int, user_id: int, text: str, reply_to: Optional[int]):
    now = time.time()
    key = _conv_key(chat_id, user_id)
    if now - _last_call_ts.get(key, 0.0) < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "我正在处理上一条消息，请稍等片刻～", reply_to=reply_to)
        return
    _last_call_ts[key] = now

    hist = _get_history(chat_id, user_id)
    search_query = build_smart_query(text, hist)
    _last_search_query[key] = search_query
    notes, ms = await search_web(search_query)
    _last_search_ms[key] = ms

    messages = _build_messages(chat_id, user_id, text, notes)
    try:
        reply_text = await openai_chat(messages, key)
        final_out = _sanitize_plain_text(reply_text) if SANITIZE_OUTPUT else reply_text
        for chunk in telegram_split(final_out):
            await tg_send_message(chat_id, chunk, reply_to=reply_to)
        _append_history(chat_id, user_id, "user", text)
        _append_history(chat_id, user_id, "assistant", reply_text)
    except Exception as e:
        _last_error[key] = f"answer:{e}"
        await tg_send_message(chat_id, f"错误: {escape_html(str(e))}", reply_to=reply_to)

# ================== FastAPI 路由 ==================
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    async def background_tasks():
        while True:
            await asyncio.sleep(3600)
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
            "已开启：上下文记忆 + 实时联网搜索\n"
            "- 每条回复先用 SerpAPI（网页+新闻并行）搜索，并只根据结果作答\n"
            "- 记忆：每用户独立，最多 12 轮，保留 24 小时\n"
            "- /clear 清空上下文，/status 查看诊断信息"
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
        last_ms = _last_search_ms.get(key, 0)
        last_err = _last_error.get(key, "无")
        await tg_send_message(
            chat_id,
            f"SerpAPI：{serp}\n模型：{model_used}\n最近搜索词：{last_q}\n最近搜索耗时：{last_ms} ms\n最近错误：{last_err}\n"
            f"记忆窗口：24小时；历史上限：{HISTORY_MAX_TURNS} 轮\n"
            f"搜索参数：hl={SEARCH_HL}, gl={SEARCH_GL}, recency={SEARCH_RECENCY or '不限'}, news={int(USE_NEWS_ENGINE)}"
        )
        return PlainTextResponse("ok")

    await answer_once(chat_id, user_id, text, reply_to=message_id)
    return PlainTextResponse("ok")
