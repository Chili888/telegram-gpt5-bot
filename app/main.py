# app/main.py
import os
import asyncio
import time
from typing import Any, Dict, List, Deque, Optional
from collections import deque
from datetime import datetime, timezone
import re
import dateparser

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

# ================== 环境变量 ==================
BOT_TOKEN       = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4.1")
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "dev-secret")
TIMEOUT_SEC     = int(os.getenv("TIMEOUT_SEC", "60"))

# 联网相关
SERPAPI_KEY     = os.getenv("SERPAPI_KEY", "")
BING_KEY        = os.getenv("BING_KEY", "")
REQUIRE_WEB     = os.getenv("REQUIRE_WEB", "1") == "1"
RECENCY         = os.getenv("RECENCY", "auto").lower()   # auto|today|7d|30d|none
MIN_RESULTS     = int(os.getenv("MIN_RESULTS", "1"))
MAX_AGE_DAYS    = int(os.getenv("MAX_AGE_DAYS", "180"))

# 防抖 / 并发
OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
CHAT_COOLDOWN_SEC      = float(os.getenv("CHAT_COOLDOWN_SEC", "1.2"))

# 上下文记忆
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "8"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "16000"))

# ================== 常量与客户端 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
SYSTEM_PROMPT = (
    "You are ChatGPT. Be concise, helpful, and safe. "
    "Always use the provided web search results as the primary source of truth."
)

client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
_last_call_ts: Dict[int, float] = {}
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

# ================== 会话记忆 ==================
def _get_history(chat_id: int) -> Deque[Dict[str, str]]:
    if chat_id not in _conversations:
        _conversations[chat_id] = deque(maxlen=HISTORY_MAX_TURNS * 2)
    return _conversations[chat_id]

def _append_history(chat_id: int, role: str, content: str):
    _get_history(chat_id).append({"role": role, "content": content})

def _build_messages(chat_id: int, user_text: str) -> List[Dict[str, str]]:
    hist = list(_get_history(chat_id))
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
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

# ================== 联网搜索 ==================
_DATE_PAT = re.compile(r"(\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4}\b|\b\d{1,2}\s*月\s*\d{4}\b|\b\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\b)",
                       re.IGNORECASE)

def _parse_when(text: str) -> Optional[datetime]:
    if not text: return None
    m = _DATE_PAT.search(text)
    probe = m.group(0) if m else text
    dt = dateparser.parse(
        probe,
        settings={
            "TIMEZONE": "UTC",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DAY_OF_MONTH": "first",
            "RELATIVE_BASE": datetime.now(timezone.utc),
        },
        languages=["zh", "en"]
    )
    return dt

def _tbs_for(rec: str) -> Optional[str]:
    return {"today": "qdr:d", "7d": "qdr:w", "30d": "qdr:m"}.get(rec)

def _bing_freshness(rec: str) -> Optional[str]:
    return {"today": "Day", "7d": "Week", "30d": "Month"}.get(rec)

async def _serpapi_fetch(query: str, rec: str, num: int) -> List[dict]:
    if not SERPAPI_KEY: return []
    params = {"engine": "google", "q": query, "num": num, "api_key": SERPAPI_KEY,
              "hl": "zh-cn", "gl": "us", "safe": "active"}
    tbs = _tbs_for(rec)
    if tbs: params["tbs"] = tbs
    r = await client.get("https://serpapi.com/search.json", params=params)
    r.raise_for_status()
    js = r.json()
    rows = []
    for it in (js.get("organic_results") or []):
        title = (it.get("title") or "").strip()
        snippet = (it.get("snippet") or "").strip()
        url = (it.get("link") or "").strip()
        date_str = (it.get("date") or "").strip()
        dt = _parse_when(date_str) or _parse_when(title) or _parse_when(snippet)
        rows.append({"title": title, "snippet": snippet, "url": url, "dt": dt})
    return rows

async def _bing_fetch(query: str, rec: str, num: int) -> List[dict]:
    if not BING_KEY: return []
    headers = {"Ocp-Apim-Subscription-Key": BING_KEY}
    params = {"q": query, "count": num, "mkt": "zh-CN", "safeSearch": "Strict"}
    fresh = _bing_freshness(rec)
    if fresh: params["freshness"] = fresh
    r = await client.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params)
    r.raise_for_status()
    js = r.json()
    rows = []
    for it in (js.get("webPages", {}).get("value") or []):
        title = (it.get("name") or "").strip()
        snippet = (it.get("snippet") or "").strip()
        url = (it.get("url") or "").strip()
        crawl = (it.get("dateLastCrawled") or "").strip()
        dt = _parse_when(crawl) or _parse_when(title) or _parse_when(snippet)
        rows.append({"title": title, "snippet": snippet, "url": url, "dt": dt})
    return rows

def _rank_and_format(rows: List[dict]) -> str:
    seen, uniq = set(), []
    for r in rows:
        u = r.get("url")
        if not u or u in seen: continue
        seen.add(u); uniq.append(r)

    if MAX_AGE_DAYS > 0:
        now = datetime.now(timezone.utc)
        kept = []
        for r in uniq:
            dt = r.get("dt")
            if dt is None or (now - dt).days <= MAX_AGE_DAYS:
                kept.append(r)
        uniq = kept

    uniq.sort(key=lambda r: (r["dt"] is None, -(r["dt"].timestamp() if r["dt"] else 0)))

    lines = []
    for r in uniq:
        t, s, u, dt = r.get("title",""), r.get("snippet",""), r.get("url",""), r.get("dt")
        when = f"（{dt.date().isoformat()}）" if dt else ""
        lines.append(f"• {t}{when}\n{s}\n{u}")
    return "\n\n".join(lines)

async def web_search(query: str, num: int = 8) -> str:
    steps = ["today","7d","30d","none"] if RECENCY=="auto" else [RECENCY]
    for rec in steps:
        s_rows, b_rows = [], []
        try: s_rows = await _serpapi_fetch(query, rec, num)
        except: pass
        try: b_rows = await _bing_fetch(query, rec, num)
        except: pass
        rows = s_rows + b_rows
        if not rows: continue
        text = _rank_and_format(rows)
        if text.strip():
            return text
    return ""

# ================== OpenAI ==================
async def _chat_once(model: str, messages: List[Dict[str, str]]) -> str:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {"model": model, "messages": messages, "temperature": 0.7}
    async with _openai_sema:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

async def openai_chat(messages: List[Dict[str, str]]) -> str:
    for model in [OPENAI_MODEL, "gpt-4.1", "gpt-4o"]:
        try:
            return await _chat_once(model, messages)
        except Exception as e:
            last_err = e
            continue
    raise last_err

# ================== 健康检查 ==================
@app.get("/healthz")
async def healthz(): return PlainTextResponse("ok")

@app.get("/")
async def root(): return PlainTextResponse("ok")

# ================== Telegram Webhook ==================
@app.post(f"/webhook/{WEBHOOK_SECRET}")
async def telegram_webhook(request: Request):
    update: Dict[str, Any] = await request.json()
    msg = update.get("message") or update.get("edited_message")
    if not msg: return PlainTextResponse("ok")

    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = msg.get("text")
    message_id = msg.get("message_id")

    if not text: return PlainTextResponse("ok")

    cmd = text.strip().lower()
    if cmd == "/start":
        await tg_send_message(chat_id, "✅ 你好！我已开启联网 + 上下文记忆。")
        return PlainTextResponse("ok")
    if cmd == "/clear":
        _conversations.pop(chat_id, None)
        await tg_send_message(chat_id, "✅ 已清空上下文。")
        return PlainTextResponse("ok")

    now = time.time()
    if now - _last_call_ts.get(chat_id, 0.0) < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "⏳ 正在处理上一条，请稍等…", reply_to=message_id)
        return PlainTextResponse("ok")
    _last_call_ts[chat_id] = now

    messages = _build_messages(chat_id, text)

    await tg_send_message(chat_id, "🔎 正在联网获取最新信息…", reply_to=message_id)
    web_snippets = await web_search(text, num=8)
    if not web_snippets and REQUIRE_WEB:
        await tg_send_message(chat_id, "❌ 未能获取最新资料，请稍后重试或更换关键词。", reply_to=message_id)
        return PlainTextResponse("ok")

    if web_snippets:
        messages.append({"role":"system","content":"以下是最新网络信息，请结合回答：\n"+web_snippets})

    try:
        reply = await openai_chat(messages)
        await tg_send_message(chat_id, reply, reply_to=message_id)
        _append_history(chat_id,"user",text)
        _append_history(chat_id,"assistant",reply)
    except Exception as e:
        await tg_send_message(chat_id, f"❌ 错误：{e}", reply_to=message_id)

    return PlainTextResponse("ok")
