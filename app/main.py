import os
import io
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
BOT_TOKEN        = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
# 文本/视觉通用模型（强推 gpt-4o / gpt-4o-mini）
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o")
# 语音识别（Whisper 或 gpt-4o-mini-transcribe）
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
# 语音合成（可选）
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
# 图片生成（OpenAI Images）
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

WEBHOOK_SECRET   = os.getenv("WEBHOOK_SECRET", "dev-secret")
TIMEOUT_SEC      = int(os.getenv("TIMEOUT_SEC", "90"))
TTS_REPLY        = os.getenv("TTS_REPLY", "0") == "1"   # 是否回语音
STRICT_MODEL     = os.getenv("STRICT_MODEL", "1") == "1" # 文本严格只用首选模型

# 记忆：8~12 轮都行，24 小时
HISTORY_MAX_TURNS    = int(os.getenv("HISTORY_MAX_TURNS", "8"))
HISTORY_LIFESPAN_SEC = float(os.getenv("HISTORY_LIFESPAN_SEC", "86400"))

# 输出
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "1400"))
SANITIZE_OUTPUT = os.getenv("SANITIZE_OUTPUT", "1") == "1"
TELEGRAM_PARSE_MODE  = os.getenv("TELEGRAM_PARSE_MODE", "HTML")
DISABLE_LINK_PREVIEW = os.getenv("DISABLE_LINK_PREVIEW", "1") == "1"

# 搜索（默认全球 google.com，不限制地区；如需中文资讯可切换到 google.com.hk/zh-cn）
SERPAPI_KEY          = os.getenv("SERPAPI_KEY", "")
SEARCH_HL            = os.getenv("SEARCH_HL", "en")            # 语言
SEARCH_GL            = os.getenv("SEARCH_GL", "us")            # 地区
SEARCH_GOOGLE_DOMAIN = os.getenv("SEARCH_GOOGLE_DOMAIN", "google.com")
SEARCH_LOCATION      = os.getenv("SEARCH_LOCATION", "")        # 留空=不偏向地理位置
SEARCH_LR            = os.getenv("SEARCH_LR", "")              # e.g. lang_en
SEARCH_CR            = os.getenv("SEARCH_CR", "")              # e.g. countryUS
SEARCH_RECENCY       = os.getenv("SEARCH_RECENCY", "")         # h/d/w/m（网页）
USE_NEWS_ENGINE      = os.getenv("USE_NEWS_ENGINE", "1") == "1"
NEWS_RECENCY         = os.getenv("NEWS_RECENCY", "7d")         # 新闻
SEARCH_NUM           = int(os.getenv("SEARCH_NUM", "10"))

OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
CHAT_COOLDOWN_SEC      = float(os.getenv("CHAT_COOLDOWN_SEC", "1.2"))

# Assistants（文档分析 / 代码解释器）
ASSISTANT_MODEL   = os.getenv("ASSISTANT_MODEL", "gpt-4o-mini")
ASSISTANT_NAME    = os.getenv("ASSISTANT_NAME", "TG Code Interpreter")
# ================== 客户端与常量 ==================
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

SYSTEM_PROMPT = (
    "你是一个具备实时联网能力的助理。"
    "所有回答都要用中文，除非用户明确要求其他语言。"
    "当提供搜索笔记时，只根据这些笔记作答；笔记不足就直说。"
    "当用户上传图片时，像 ChatGPT Vision 一样理解并回答。"
    "当用户上传 PDF/CSV/XLSX 时，使用 code_interpreter 分析并给出结论和可执行建议。"
    "不要输出 markdown 装饰或项目符号，使用干净的中文句子。"
)

client = httpx.AsyncClient(timeout=TIMEOUT_SEC)
_openai_sema = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)

# 记忆：chat_id:user_id 维度
_conversations: Dict[str, Deque[Dict[str, Any]]] = {}
_last_call_ts: Dict[str, float] = {}
_threads: Dict[str, str] = {}         # Assistants 线程
_assistant_id: Optional[str] = None   # Assistants 模板

# 运行状态
_last_model_used: Dict[str, str] = {}
_last_search_query: Dict[str, str] = {}
_last_search_ms: Dict[str, int] = {}
_last_error: Dict[str, str] = {}

# =============== 工具函数 ===============
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
        _conversations[key] = deque([m for m in dq if now - m.get("ts", 0) < HISTORY_LIFESPAN_SEC], maxlen=dq.maxlen)

def escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _sanitize(text: str) -> str:
    if "```" in text: return text
    t = text
    t = re.sub(r"(?m)^\s*[-*•·]+\s+", "", t)
    t = re.sub(r"[`_#>]", "", t)
    t = re.sub(r"([!！?？.,，。;；:：\-—])\1{1,}", r"\1", t)
    return t.strip()

def telegram_split(text: str, limit: int = 4096) -> List[str]:
    parts: List[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit) or limit
        parts.append(text[:cut]); text = text[cut:]
    if text: parts.append(text)
    return parts

async def tg_send_message(chat_id: int, text: str, reply_to: Optional[int] = None):
    payload = {"chat_id": chat_id, "text": text, "parse_mode": TELEGRAM_PARSE_MODE, "disable_web_page_preview": DISABLE_LINK_PREVIEW}
    if reply_to: payload["reply_to_message_id"] = reply_to
    r = await client.post(f"{TELEGRAM_API}/sendMessage", json=payload); r.raise_for_status(); return r.json()

async def tg_send_photo_bytes(chat_id: int, image_bytes: bytes, caption: str = "", reply_to: Optional[int] = None):
    data = {"chat_id": str(chat_id)}
    if caption: data["caption"] = caption
    if reply_to: data["reply_to_message_id"] = str(reply_to)
    files = {"photo": ("image.jpg", image_bytes, "image/jpeg")}
    r = await client.post(f"{TELEGRAM_API}/sendPhoto", data=data, files=files); r.raise_for_status(); return r.json()

async def tg_send_audio_bytes(chat_id: int, audio_bytes: bytes, filename="reply.mp3", reply_to: Optional[int] = None):
    data = {"chat_id": str(chat_id)}
    if reply_to: data["reply_to_message_id"] = str(reply_to)
    files = {"audio": (filename, audio_bytes, "audio/mpeg")}
    r = await client.post(f"{TELEGRAM_API}/sendAudio", data=data, files=files); r.raise_for_status(); return r.json()

async def tg_get_file_url(file_id: str) -> str:
    r = await client.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id}); r.raise_for_status()
    fp = r.json()["result"]["file_path"]
    return f"https://api.telegram.org/file/bot{BOT_TOKEN}/{fp}"

# =============== 搜索：google + google_news（并行、去重、可信域名优先） ===============
_TRUSTED = [
    "reuters.com","apnews.com","bloomberg.com","ft.com","wsj.com",
    "bbc.com","cnn.com","nytimes.com","theguardian.com",
    "caixin.com","scmp.com","xinhuanet.com","cctv.com","china.com.cn"
]

def _domain_score(url: str) -> int:
    try:
        host = re.sub(r"^https?://", "", url).split("/")[0].lower()
    except Exception:
        return 999
    for i, dom in enumerate(_TRUSTED):
        if host.endswith(dom): return i
    return 500

def _fmt_items(items: List[Dict[str, Any]], is_news: bool) -> List[Tuple[str,str,str,str]]:
    out = []
    for it in items:
        t = it.get("title",""); s = it.get("snippet","") or it.get("content","")
        link = it.get("link","") or it.get("url",""); d = it.get("date","") if is_news else (it.get("date","") or it.get("published_at",""))
        if t or s: out.append((t,s,link,d))
    return out

def _merge_rank(web: List[Tuple[str,str,str,str]], news: List[Tuple[str,str,str,str]]) -> str:
    seen=set(); merged=[]
    for bucket in (news, web):
        for t,s,link,d in bucket:
            k=(t.strip(), link.strip())
            if not link or k in seen: continue
            seen.add(k); merged.append((t,s,link,d))
    merged.sort(key=lambda x: _domain_score(x[2]))
    lines=[]
    for t,s,link,d in merged[:SEARCH_NUM]:
        ds=f" ({d})" if d else ""
        lines.append(f"{t}{ds}: {s} [{link}]")
    return "\n".join(lines) if lines else "No fresh search results."

async def web_search(query: str) -> Tuple[str,int]:
    if not SERPAPI_KEY:
        return "Search disabled: SERPAPI_KEY not set.", 0
    q = query.strip() or "latest"
    common = {"hl": SEARCH_HL, "gl": SEARCH_GL, "api_key": SERPAPI_KEY, "num": SEARCH_NUM}
    if SEARCH_GOOGLE_DOMAIN: common["google_domain"] = SEARCH_GOOGLE_DOMAIN
    if SEARCH_LOCATION: common["location"] = SEARCH_LOCATION
    if SEARCH_LR: common["lr"] = SEARCH_LR
    if SEARCH_CR: common["cr"] = SEARCH_CR

    reqs=[]
    params_web={"engine":"google","q":q, **common}
    if SEARCH_RECENCY: params_web["tbs"] = f"qdr:{SEARCH_RECENCY}"
    reqs.append(client.get("https://serpapi.com/search", params=params_web))
    if USE_NEWS_ENGINE:
        params_news={"engine":"google_news","q":q,"when":NEWS_RECENCY, **{k:v for k,v in common.items() if k!="tbs"}}
        reqs.append(client.get("https://serpapi.com/search", params=params_news))

    t0=time.monotonic()
    resp=await asyncio.gather(*reqs, return_exceptions=True)
    t1=time.monotonic()

    web_items=[]; news_items=[]
    for r in resp:
        if isinstance(r, Exception): continue
        try: data=r.json()
        except Exception: continue
        engine=(data.get("search_metadata",{}) or {}).get("engine","")
        if engine=="google_news":
            news_items += _fmt_items(data.get("news_results",[]), True)
        else:
            web_items  += _fmt_items(data.get("organic_results",[]), False)
            abox=data.get("answer_box") or {}
            if abox:
                news_items.append((abox.get("title","Answer"), abox.get("snippet","") or abox.get("answer",""), abox.get("link","") or "", ""))
    notes=_merge_rank(web_items, news_items)
    return notes, int((t1-t0)*1000)

# =============== OpenAI 基础调用 ===============
async def openai_chat(messages: List[Dict[str, Any]], model: Optional[str]=None) -> str:
    m = model or OPENAI_MODEL
    url=f"{OPENAI_BASE_URL}/chat/completions"
    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body={"model": m, "messages": messages, "temperature": 0.5, "max_tokens": MAX_TOKENS}
    async with _openai_sema:
        r=await client.post(url, headers=headers, json=body)
        r.raise_for_status()
        data=r.json()
        return data["choices"][0]["message"]["content"]

async def openai_transcribe(file_bytes: bytes, filename: str) -> str:
    url=f"{OPENAI_BASE_URL}/audio/transcriptions"
    data={"model": OPENAI_STT_MODEL}
    files={"file": (filename, file_bytes, "audio/ogg")}
    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
    r=await client.post(url, headers=headers, data=data, files=files); r.raise_for_status()
    return r.json()["text"]

async def openai_tts(text: str) -> bytes:
    url=f"{OPENAI_BASE_URL}/audio/speech"
    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
    body={"model": OPENAI_TTS_MODEL, "input": text, "voice":"alloy", "format":"mp3"}
    r=await client.post(url, headers=headers, json=body); r.raise_for_status()
    return r.content

async def openai_image_generate(prompt: str) -> bytes:
    url=f"{OPENAI_BASE_URL}/images/generations"
    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
    body={"model": OPENAI_IMAGE_MODEL, "prompt": prompt, "size":"1024x1024", "n":1, "response_format":"b64_json"}
    r=await client.post(url, headers=headers, json=body); r.raise_for_status()
    b64=r.json()["data"][0]["b64_json"]
    import base64; return base64.b64decode(b64)

# Assistants：创建一次 Assistant（带 code_interpreter）
async def _ensure_assistant() -> str:
    global _assistant_id
    if _assistant_id: return _assistant_id
    url=f"{OPENAI_BASE_URL}/assistants"
    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
    body={"model": ASSISTANT_MODEL, "name": ASSISTANT_NAME, "tools":[{"type":"code_interpreter"}]}
    r=await client.post(url, headers=headers, json=body); r.raise_for_status()
    _assistant_id = r.json()["id"]; return _assistant_id

async def _ensure_thread(key: str) -> str:
    if key in _threads: return _threads[key]
    url=f"{OPENAI_BASE_URL}/threads"
    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
    r=await client.post(url, headers=headers); r.raise_for_status()
    tid=r.json()["id"]; _threads[key]=tid; return tid

async def _openai_upload_for_assistants(file_bytes: bytes, filename: str) -> str:
    url=f"{OPENAI_BASE_URL}/files"
    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data={"purpose":"assistants"}
    files={"file": (filename, file_bytes, "application/octet-stream")}
    r=await client.post(url, headers=headers, data=data, files=files); r.raise_for_status()
    return r.json()["id"]

async def assistants_analyze_files(key: str, user_prompt: str, files: List[Tuple[bytes,str]]) -> str:
    assistant_id = await _ensure_assistant()
    thread_id    = await _ensure_thread(key)
    file_ids=[]
    for b,fn in files:
        fid=await _openai_upload_for_assistants(b, fn); file_ids.append(fid)

    # 把消息写入线程
    url_msg=f"{OPENAI_BASE_URL}/threads/{thread_id}/messages"
    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
    content=[{"type":"text","text": user_prompt}]
    for fid in file_ids:
        content.append({"type":"input_image","image_file":{"file_id": fid}}) if fid.endswith((".jpg",".png",".jpeg")) else None
    # 文档作为附件
    attachments=[{"file_id": fid, "tools":[{"type":"code_interpreter"}]} for fid in file_ids]
    body={"role":"user","content":[{"type":"text","text": user_prompt}], "attachments": attachments}
    r=await client.post(url_msg, headers=headers, json=body); r.raise_for_status()

    # 运行
    url_run=f"{OPENAI_BASE_URL}/threads/{thread_id}/runs"
    body={"assistant_id": assistant_id}
    r=await client.post(url_run, headers=headers, json=body); r.raise_for_status()
    run_id=r.json()["id"]

    # 轮询直到完成
    for _ in range(120):
        rr=await client.get(f"{OPENAI_BASE_URL}/threads/{thread_id}/runs/{run_id}", headers=headers); rr.raise_for_status()
        st=rr.json()["status"]
        if st in ("completed","failed","cancelled","expired"): break
        await asyncio.sleep(1.5)
    if st!="completed":
        return f"抱歉，沙盒分析未完成，状态：{st}"

    # 取最后一条 assistant 消息
    mm=await client.get(f"{OPENAI_BASE_URL}/threads/{thread_id}/messages", headers=headers, params={"order":"desc","limit":1})
    mm.raise_for_status()
    msg=mm.json()["data"][0]
    parts=msg.get("content",[])
    text_out=[]
    for p in parts:
        if p.get("type")=="text":
            text_out.append(p["text"]["value"])
    return "\n".join(text_out) if text_out else "分析完成，但没有可读文本输出。"

# =============== 业务流程 ===============
def _build_messages(chat_id: int, user_id: int, user_text: str, search_notes: Optional[str]) -> List[Dict[str, Any]]:
    hist=[m for m in _get_history(chat_id,user_id) if time.time()-m.get("ts",0)<HISTORY_LIFESPAN_SEC]
    msgs=[{"role":"system","content": SYSTEM_PROMPT}]
    if search_notes:
        msgs.append({"role":"system","content": f"以下是最新搜索笔记：\n{search_notes}"})
    total=0; acc=[]
    for m in reversed(hist):
        c=m.get("content") or ""
        if total+len(c)>16000: break
        acc.append({"role": m["role"], "content": c}); total+=len(c)
    msgs.extend(reversed(acc))
    msgs.append({"role":"user","content": user_text})
    return msgs

async def handle_text(chat_id:int, user_id:int, text:str, reply_to:Optional[int]):
    now=time.time(); key=_conv_key(chat_id,user_id)
    if now-_last_call_ts.get(key,0.0)<CHAT_COOLDOWN_SEC:
        await tg_send_message(chat_id,"我正在处理上一条消息，请稍等片刻～",reply_to); return
    _last_call_ts[key]=now

    # 先搜再答（全球搜索）
    _last_search_query[key]=text
    notes,ms=await web_search(text); _last_search_ms[key]=ms

    messages=_build_messages(chat_id,user_id,text,notes)
    try:
        out=await openai_chat(messages, model=OPENAI_MODEL)
        _last_model_used[key]=OPENAI_MODEL
        final=_sanitize(out) if SANITIZE_OUTPUT else out
        for part in telegram_split(final): await tg_send_message(chat_id, part, reply_to)
        _append_history(chat_id,user_id,"user",text); _append_history(chat_id,user_id,"assistant",final)
    except Exception as e:
        _last_error[key]=f"openai:{e}"
        await tg_send_message(chat_id,f"错误: {escape_html(str(e))}",reply_to)

async def handle_photo(chat_id:int, user_id:int, file_id:str, caption:str, reply_to:Optional[int]):
    url=await tg_get_file_url(file_id)
    # 直接 Vision
    messages=[{"role":"system","content": SYSTEM_PROMPT},
              {"role":"user","content":[
                  {"type":"input_text","text": caption or "请详细描述这张图片，并提取关键信息。"},
                  {"type":"input_image","image_url": url}
              ]}]
    try:
        out=await openai_chat(messages, model=OPENAI_MODEL)
        final=_sanitize(out) if SANITIZE_OUTPUT else out
        for part in telegram_split(final): await tg_send_message(chat_id, part, reply_to)
        _append_history(chat_id,user_id,"user",f"[图片]{caption}"); _append_history(chat_id,user_id,"assistant",final)
    except Exception as e:
        await tg_send_message(chat_id,f"图片理解失败：{escape_html(str(e))}",reply_to)

async def handle_voice(chat_id:int, user_id:int, file_id:str, reply_to:Optional[int]):
    file_url=await tg_get_file_url(file_id)
    r=await client.get(file_url); r.raise_for_status()
    audio_bytes=r.content
    try:
        text=await openai_transcribe(audio_bytes,"voice.ogg")
    except Exception as e:
        await tg_send_message(chat_id,f"转写失败：{escape_html(str(e))}",reply_to); return
    # 当作文本继续走
    await handle_text(chat_id,user_id,text,reply_to)
    if TTS_REPLY:
        try:
            speak=await openai_tts("好的，我已根据你的语音进行了回答。")
            await tg_send_audio_bytes(chat_id,speak,"reply.mp3",reply_to)
        except Exception:
            pass

async def handle_document(chat_id:int, user_id:int, file_id:str, file_name:str, caption:str, reply_to:Optional[int]):
    key=_conv_key(chat_id,user_id)
    file_url=await tg_get_file_url(file_id)
    r=await client.get(file_url); r.raise_for_status()
    file_bytes=r.content
    # 交给 Assistants+code_interpreter
    user_prompt = caption or "请阅读并分析我上传的文件，给出关键结论与可执行建议。"
    try:
        result = await assistants_analyze_files(key, user_prompt, [(file_bytes, file_name)])
        final=_sanitize(result) if SANITIZE_OUTPUT else result
        for part in telegram_split(final): await tg_send_message(chat_id, part, reply_to)
        _append_history(chat_id,user_id,"user",f"[文档]{file_name} {user_prompt}")
        _append_history(chat_id,user_id,"assistant",final)
    except Exception as e:
        await tg_send_message(chat_id,f"文档分析失败：{escape_html(str(e))}",reply_to)

async def handle_img_cmd(chat_id:int, user_id:int, text:str, reply_to:Optional[int]):
    prompt=text.strip()[4:].strip()  # remove /img
    if not prompt:
        await tg_send_message(chat_id,"用法：/img 你的提示词",reply_to); return
    try:
        img=await openai_image_generate(prompt)
        await tg_send_photo_bytes(chat_id,img,caption="生成完成",reply_to=reply_to)
    except Exception as e:
        await tg_send_message(chat_id,f"图片生成失败：{escape_html(str(e))}",reply_to)

# =============== FastAPI 路由 ===============
app=FastAPI()

@app.on_event("startup")
async def startup_event():
    async def bg():
        while True:
            await asyncio.sleep(3600); _prune_old()
    asyncio.create_task(bg())
    # 预创建 Assistant（可选）
    try:
        await _ensure_assistant()
    except Exception:
        pass

@app.get("/healthz")
async def healthz(): return PlainTextResponse("ok")

@app.get("/")
async def root(): return PlainTextResponse("ok")

@app.post(f"/webhook/{WEBHOOK_SECRET}")
async def telegram_webhook(request: Request):
    upd=await request.json()
    msg=upd.get("message") or upd.get("edited_message")
    if not msg: return PlainTextResponse("ok")

    chat_id = msg.get("chat",{}).get("id")
    user_id = msg.get("from",{}).get("id")
    message_id = msg.get("message_id")

    # 命令
    text = msg.get("text","") or ""
    cmd  = text.strip().lower()
    if cmd == "/start":
        tips=("已开启：多轮推理 + 全球联网搜索 + 图像理解 + 语音转写 + 文档分析(code_interpreter) + 图片生成\n"
              "命令：/img 生成图片，/clear 清空上下文，/status 查看状态")
        await tg_send_message(chat_id,tips,reply_to=message_id); return PlainTextResponse("ok")
    if cmd == "/clear":
        _conversations.pop(_conv_key(chat_id,user_id), None); _threads.pop(_conv_key(chat_id,user_id), None)
        await tg_send_message(chat_id,"已清空你的上下文",reply_to=message_id); return PlainTextResponse("ok")
    if cmd == "/status":
        key=_conv_key(chat_id,user_id)
        serp="已配置" if SERPAPI_KEY else "未配置"
        used=_last_model_used.get(key,"(尚未调用)")
        lq=_last_search_query.get(key,"(无)")
        ms=_last_search_ms.get(key,0)
        err=_last_error.get(key,"无")
        await tg_send_message(chat_id, f"SerpAPI：{serp}\n模型：{used}\n最近搜索：{lq}\n搜索耗时：{ms} ms\n最近错误：{err}\n"
                                       f"记忆：24h / 上限 {HISTORY_MAX_TURNS} 轮", reply_to=message_id)
        return PlainTextResponse("ok")

    # 图片生成指令
    if text.startswith("/img"):
        await handle_img_cmd(chat_id,user_id,text,message_id); return PlainTextResponse("ok")

    # 语音
    voice = msg.get("voice")
    if voice and voice.get("file_id"):
        await handle_voice(chat_id,user_id,voice["file_id"],message_id); return PlainTextResponse("ok")

    # 照片（取最大分辨率的一个）
    photos = msg.get("photo") or []
    if photos:
        file_id = sorted(photos, key=lambda x: x.get("file_size",0))[-1]["file_id"]
        await handle_photo(chat_id,user_id,file_id,msg.get("caption",""),message_id); return PlainTextResponse("ok")

    # 文档：PDF/CSV/XLSX
    document = msg.get("document")
    if document and document.get("file_id"):
        file_id=document["file_id"]; file_name=document.get("file_name","file.bin").lower()
        await handle_document(chat_id,user_id,file_id,file_name,msg.get("caption",""),message_id); return PlainTextResponse("ok")

    # 普通文本
    if text:
        await handle_text(chat_id,user_id,text,message_id)
    else:
        await tg_send_message(chat_id,"我目前支持文本、语音、图片、PDF/CSV/Excel 与 /img 指令。",reply_to=message_id)

    return PlainTextResponse("ok")
