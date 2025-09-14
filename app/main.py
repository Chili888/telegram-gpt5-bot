# app/main.py
import os
import asyncio
import time
import re
import html
from typing import Any, Dict, List, Deque, Optional
from collections import deque

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from bs4 import BeautifulSoup
from readability import Document

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
ALWAYS_WEB = os.getenv("ALWAYS_WEB", "0") == "1"

# 会话记忆（每会话最近 N 轮；粗略字符上限）
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "8"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "16000"))

# 联网搜索配置
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "").lower().strip()   # serpapi | bing
SERPAPI_KEY     = os.getenv("SERPAPI_KEY", "")
BING_KEY        = os.getenv("BING_KEY", "")
MAX_SOURCES     = int(os.getenv("MAX_SOURCES", "4"))
CRAWL_TIMEOUT   = int(os.getenv("CRAWL_TIMEOUT", "15"))

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

# ================== 工具函数 ==================
def _split_long(text: str, limit: int = 3900) -> List[str]:
    """分割长文本（Telegram 单条消息上限约 4096 字符）"""
    chunks = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks

def _clean_text(txt: str) -> str:
    txt = html.unescape(txt or "")
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

# ================== Telegram 工具 ==================
async def tg_send_message(chat_id: int, text: str, reply_to: Optional[int] = None):
    for part in _split_long(text):
        payload = {"chat_id": chat_id, "text": part}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
            reply_to = None  # 只在第一段回复原消息
        r = await client.post(f"{TELEGRAM_API}/sendMessage", json=payload)
        r.raise_for_status()
    return True

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

# ================== Web Search + 抓取正文 ==================
async def _search_web(query: str) -> List[dict]:
    """
    返回 [{title, url, snippet}]，按相关度排序，最多 MAX_SOURCES 条
    """
    results: List[dict] = []
    try:
        if SEARCH_PROVIDER == "serpapi" and SERPAPI_KEY:
            url = "https://serpapi.com/search.json"
            params = {"engine": "google", "q": query, "num": MAX_SOURCES, "api_key": SERPAPI_KEY, "hl": "zh-cn"}
            r = await client.get(url, params=params, timeout=CRAWL_TIMEOUT)
            r.raise_for_status()
            js = r.json()
            for item in (js.get("organic_results") or [])[:MAX_SOURCES]:
                results.append({
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "snippet": item.get("snippet")
                })
        elif SEARCH_PROVIDER == "bing" and BING_KEY:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": BING_KEY}
            params = {"q": query, "mkt": "zh-CN", "count": MAX_SOURCES}
            r = await client.get(url, headers=headers, params=params, timeout=CRAWL_TIMEOUT)
            r.raise_for_status()
            js = r.json()
            for item in (js.get("webPages", {}).get("value") or [])[:MAX_SOURCES]:
                results.append({
                    "title": item.get("name"),
                    "url": item.get("url"),
                    "snippet": item.get("snippet")
                })
        return results
    except Exception as e:
        print("search error:", e)
        return []

async def _fetch_main_text(url: str) -> str:
    """
    抓取网页并提取正文（readability + bs4 兜底），只保留纯文本
    """
    try:
        r = await client.get(url, timeout=CRAWL_TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        html_str = r.text
        # 先试 readability 抽正文
        try:
            doc = Document(html_str)
            content_html = doc.summary(html_partial=True)
        except Exception:
            content_html = html_str
        soup = BeautifulSoup(content_html, "lxml")
        # 去掉脚本/样式
        for bad in soup(["script", "style", "noscript"]):
            bad.decompose()
        text = soup.get_text(separator=" ")
        return _clean_text(text)
    except Exception as e:
        print("fetch error:", e, url)
        return ""

async def web_answer(query: str) -> str:
    """
    搜索 -> 抓取正文 -> 交给模型总结，最后附带来源链接
    """
    hits = await _search_web(query)
    if not hits:
        return "❌ 没找到相关网页或搜索服务不可用。请先配置 SEARCH_PROVIDER + Key。"

    # 抓取正文
    bundles = []
    for h in hits:
        if not h.get("url"):
            continue
        text = await _fetch_main_text(h["url"])
        if not text:
            continue
        bundles.append({
            "title": h.get("title") or "",
            "url": h["url"],
            "snippet": h.get("snippet") or "",
            "text": text[:5000]  # 控制每篇长度，避免超长
        })

    if not bundles:
        return "❌ 网页抓取失败，可能被站点阻挡。换个关键词或稍后再试。"

    # 组织给模型的上下文
    ctx_chunks = []
    for i, b in enumerate(bundles, 1):
        ctx_chunks.append(
            f"[{i}] {b['title']}\nURL: {b['url']}\n片段: {b['snippet']}\n正文摘录: {b['text']}\n"
        )
    context_block = "\n\n".join(ctx_chunks)

    prompt = (
        "你是一个联网助手。请基于给定的网页摘录，"
        "用简洁中文回答用户的问题，并在末尾给出引用来源编号（例如：[1][2]）。"
        "如果信息相互矛盾或不确定，要明确说明。\n\n"
        f"【用户问题】:\n{query}\n\n"
        f"【检索到的材料】:\n{context_block}"
    )

    messages = [
        {"role": "system", "content": "Answer in Chinese. Be concise and neutral. Cite like [1][2]."},
        {"role": "user", "content": prompt}
    ]
    try:
        answer = await openai_chat(messages)
    except Exception as e:
        return f"❌ OpenAI 调用失败：{e}"

    refs = "\n".join([f"[{i+1}] {b['title']} - {b['url']}" for i, b in enumerate(bundles)])
    return f"{answer}\n\n———\n来源：\n{refs}"

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
    cmd_raw = text.strip()
    cmd = cmd_raw.lower()
    if cmd == "/start":
        await tg_send_message(chat_id, "✅ 你好！我已开启上下文记忆（最近多轮）。\n联网搜索指令：/web 关键词 或 发送“搜索 xxx / 查一下 xxx”。")
        return PlainTextResponse("ok")
    if cmd == "/clear":
        _conversations.pop(chat_id, None)
        await tg_send_message(chat_id, "✅ 已清空本会话上下文。")
        return PlainTextResponse("ok")

    # 联网命令：/web 你的问题
    if cmd.startswith("/web"):
        q = cmd_raw[4:].strip() or "今天的重点新闻"
        await tg_send_message(chat_id, "🔎 正在联网搜索，请稍候…", reply_to=message_id)
        reply = await web_answer(q)
        await tg_send_message(chat_id, reply, reply_to=message_id)
        return PlainTextResponse("ok")

    # 中文关键词触发：搜索 / 查一下 / search
    lower = cmd_raw.lower()
    if cmd_raw.startswith("搜索 ") or cmd_raw.startswith("查一下") or lower.startswith("search "):
        q = cmd_raw.split(maxsplit=1)[1] if " " in cmd_raw else cmd_raw
        await tg_send_message(chat_id, "🔎 正在联网搜索，请稍候…", reply_to=message_id)
        reply = await web_answer(q)
        await tg_send_message(chat_id, reply, reply_to=message_id)
        return PlainTextResponse("ok")

    # 每 chat 防抖
        # 每 chat 防抖
    now = time.time()
    if now - _last_call_ts.get(chat_id, 0.0) < CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id, "我正在处理上一条消息，请稍等片刻～", reply_to=message_id)
        return PlainTextResponse("ok")
    _last_call_ts[chat_id] = now

    # === 默认走联网搜索（需要 ALWAYS_WEB=1） ===
    if ALWAYS_WEB:
        await tg_send_message(chat_id, "🔎 正在联网搜索，请稍候…", reply_to=message_id)
        reply = await web_answer(text)
        # 联网成功就直接返回；失败（返回以“❌”开头）再回落到本地对话
        if not reply.startswith("❌"):
            await tg_send_message(chat_id, reply)
            _append_history(chat_id, "user", text)
            _append_history(chat_id, "assistant", reply)
            return PlainTextResponse("ok")

    # === 回落：直接用模型（带历史） ===
    messages = _build_messages(chat_id, text)
    try:
        reply = await openai_chat(messages)
        await tg_send_message(chat_id, reply, reply_to=message_id)
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

