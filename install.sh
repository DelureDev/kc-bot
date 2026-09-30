#!/usr/bin/env bash
# Установка/обновление тестовой страницы помощника КЦ.
# Запуск:  curl -fsSL https://raw.githubusercontent.com/DelureDev/kc-bot/main/install.sh | sudo bash
# Удаление: systemctl disable --now kc-test && rm -rf /opt/kc-test /var/lib/kc-test /etc/systemd/system/kc-test.service
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Запустите через sudo"; exit 1; }
command -v python3 >/dev/null || { apt-get update -qq && apt-get install -y -qq python3; }
mkdir -p /opt/kc-test
cat > /opt/kc-test/app.py <<'PYEOF'
#!/usr/bin/env python3
"""Тестовая страница помощника КЦ.

Два движка для сравнения:
  * timeweb  — AI-агент Timeweb со встроенной базой знаний (OpenAI-совместимый API);
  * aitunnel — свой RAG: документы режутся на куски, векторы (эмбеддинги AITunnel) лежат в памяти,
               поиск косинусом -> реранк -> ответ модели AITunnel.
Только стандартная библиотека Python 3.8+.

Переменные окружения: PAGE_PASSWORD, PORT, TW_TOKEN, TW_AGENT_ID, AITUNNEL_KEY,
AIT_BASE, EMB_MODEL, RERANK_MODEL, DATA_DIR, CLOUDRU_KEY, CLOUDRU_BASE, TWGW_KEY, TWGW_BASE, GIGACHAT_*,
EMBED_KEY (ключ виджета /embed), EMBED_MODEL, EMBED_FALLBACK, EMBED_DAY_LIMIT.
"""
import hmac, http.cookies, json, math, os, re, secrets, threading, time, uuid
import urllib.error, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PASSWORD = os.environ["PAGE_PASSWORD"]
PORT = int(os.environ.get("PORT", "8787"))
TW_TOKEN = os.environ.get("TW_TOKEN", "")
TW_AGENT = os.environ.get("TW_AGENT_ID", "d94b4d73-2ce5-4006-a6f5-4039b60c2d9e")
TW_URL = "https://agent.timeweb.cloud/api/v1/cloud-ai/agents/%s/v1/chat/completions" % TW_AGENT
AIT_KEY = os.environ.get("AITUNNEL_KEY", "")
AIT_BASE = os.environ.get("AIT_BASE", "https://api.aitunnel.ru/v1").rstrip("/")
EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "rerank-4-pro")
DATA_DIR = os.environ.get("DATA_DIR") or os.environ.get("STATE_DIRECTORY") or "/var/lib/kc-test"
DOCS_DIR = os.path.join(DATA_DIR, "docs")
INDEX_FILE = os.path.join(DATA_DIR, "index.json")
os.makedirs(DOCS_DIR, exist_ok=True)

GIGA_KEY = os.environ.get("GIGACHAT_AUTH_KEY", "")
GIGA_SCOPE = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
GIGA_CA = os.environ.get("GIGACHAT_CA", "/opt/kc-test/russian_trusted_root_ca.pem")
GIGA_MODELS = [m.strip() for m in os.environ.get("GIGACHAT_MODELS", "GigaChat-2,GigaChat-2-Pro,GigaChat-2-Max").split(",") if m.strip()]
CR_KEY = os.environ.get("CLOUDRU_KEY") or os.environ.get("CLOUD_RU_FOUNDATION_MODELS_API_KEY", "")
CR_BASE = os.environ.get("CLOUDRU_BASE", "https://foundation-models.api.cloud.ru/v1").rstrip("/")
TWGW_KEY = os.environ.get("TWGW_KEY", "")
TWGW_BASE = os.environ.get("TWGW_BASE", "https://api.timeweb.ai/v1").rstrip("/")
# OpenAI-совместимые провайдеры: подпись, базовый URL, ключ. Каталог моделей берётся из их /models.
PROV = {
    "aitunnel": {"label": "AITunnel", "base": AIT_BASE, "key": AIT_KEY, "env": "AITUNNEL_KEY"},
    "cloudru": {"label": "Cloud.ru", "base": CR_BASE, "key": CR_KEY, "env": "CLOUDRU_KEY"},
    "twgw": {"label": "Timeweb AI Gateway", "base": TWGW_BASE, "key": TWGW_KEY, "env": "TWGW_KEY"},
}
CAT = {p: {"models": [], "raw": {}, "ts": 0, "err": ""} for p in PROV}
MODELS = ["gemini-3.5-flash-lite", "gpt-6-luna", "deepseek-v4.1-flash", "gpt-5.4-mini", "qwen3.8-flash", "claude-haiku-4.5"]
SESSIONS = {}
SESS_FILE = os.path.join(DATA_DIR, "sessions.json")
VERSION = "7"
# Встраиваемый виджет (/embed) для 1С и других сайтов: вход по ключу вместо пароля, одна модель, CORS.
EMBED_KEY = os.environ.get("EMBED_KEY", "")
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "aitunnel")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gpt-6-luna")
EMBED_FALLBACK = os.environ.get("EMBED_FALLBACK", "gemini-3.1-flash-lite")
EMBED_DAY_LIMIT = int(os.environ.get("EMBED_DAY_LIMIT", "3000"))
EMBED_COUNT = {"day": "", "n": 0}


def save_sessions():
    try:
        with open(SESS_FILE, "w") as f:
            json.dump(list(SESSIONS.keys()), f)
    except OSError:
        pass


def load_sessions():
    try:
        with open(SESS_FILE) as f:
            for s in json.load(f):
                SESSIONS[s] = 0
    except (OSError, ValueError):
        pass
INDEX = {"chunks": [], "built": None, "emb_tokens": 0}
INDEX_LOCK = threading.Lock()

SYSTEM = (
    "Ты помощник администратора колл-центра клиник Дентал Фэнтези, Белгравия и Фэнтези. "
    "Администратор спрашивает во время телефонного звонка. Отвечай ТОЛЬКО по фрагментам базы знаний ниже. "
    "Формат: 2-4 коротких пункта, до 60 слов, без вступлений и таблиц. Цифры, сроки, телефоны и фамилии бери точно из фрагментов. "
    "Если фрагменты противоречат друг другу, приоритет у «Общий регламент КЦ 2025». "
    "Цены из раздела про клиники Полидент — это цены Полидента (другая сеть), не выдавай их за наши. "
    "В конце отдельной строкой: «Источник: название документа». "
    "Если во фрагментах нет ответа — ответь одной фразой: «Не нашёл в базе, уточните у старшего администратора» — и ничего не придумывай. "
    "Не давай пациенту медицинских рекомендаций."
)


# ---------------------------------------------------------------- HTTP helpers
def post_json(url, payload, headers, timeout=90):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    h = {"Content-Type": "application/json"}
    h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError("HTTP %s: %s" % (e.code, e.read()[:300].decode("utf-8", "ignore")))


def ait(path, payload, timeout=90):
    if not AIT_KEY:
        raise RuntimeError("На сервере не задан AITUNNEL_KEY")
    return post_json(AIT_BASE + path, payload, {"Authorization": "Bearer " + AIT_KEY}, timeout)


# ---------------------------------------------------------------- chunking & index
def chunk_doc(name, text, size=1400, overlap=250):
    lines = [l.rstrip() for l in text.split("\n")]
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), name)
    body = [l for l in lines if l.strip() and not l.startswith("# ") and not l.startswith("Источник:")]
    chunks, cur, heading = [], "", ""
    for l in body:
        s = l.strip()
        is_head = s.startswith("#") or (len(s) < 90 and s.upper() == s and re.search("[А-ЯA-Z]{4}", s))
        if is_head:
            heading = s.lstrip("# ").strip()
        if len(cur) + len(s) + 1 > size and cur:
            chunks.append(cur)
            cur = cur[-overlap:] if overlap else ""
            if heading and not is_head:
                cur = "[" + heading + "] " + cur
        cur += ("\n" if cur else "") + s
    if cur.strip():
        chunks.append(cur)
    return [{"doc": title, "file": name, "text": c} for c in chunks]


def embed(texts):
    out, tokens = [], 0
    for i in range(0, len(texts), 64):
        j = ait("/embeddings", {"model": EMB_MODEL, "input": texts[i:i + 64]}, timeout=120)
        data = sorted(j["data"], key=lambda d: d.get("index", 0))
        out.extend(d["embedding"] for d in data)
        tokens += (j.get("usage") or {}).get("total_tokens", 0)
    return out, tokens


def norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def build_index():
    chunks = []
    for fn in sorted(os.listdir(DOCS_DIR)):
        if fn.endswith((".md", ".txt")):
            with open(os.path.join(DOCS_DIR, fn), encoding="utf-8") as f:
                chunks.extend(chunk_doc(fn, f.read()))
    if not chunks:
        raise RuntimeError("Нет документов")
    vecs, tokens = embed(["%s\n%s" % (c["doc"], c["text"]) for c in chunks])
    for c, v in zip(chunks, vecs):
        c["v"] = norm(v)
    idx = {"chunks": chunks, "built": time.strftime("%Y-%m-%d %H:%M"), "emb_tokens": tokens, "emb_model": EMB_MODEL}
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False)
    with INDEX_LOCK:
        INDEX.clear()
        INDEX.update(idx)
        bm25_prepare()
    return idx


def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, encoding="utf-8") as f:
            INDEX.update(json.load(f))
        bm25_prepare()


STOP = set("и в во на с со к ко по о об от до за из у не ни а но или же ли то что как так для при это этот эта эти его ее её их мы вы он она они же бы был была были есть нет да ли уже ещё еще".split())
QCACHE = {}


def toks(text):
    out = []
    for w in re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е")):
        if w in STOP or (len(w) < 3 and not w.isdigit()):
            continue
        out.append(w[:6] if len(w) > 6 else w)
    return out


def bm25_prepare():
    chunks = INDEX.get("chunks", [])
    df = {}
    for c in chunks:
        c["tf"] = {}
        for t in toks(c["doc"] + " " + c["text"]):
            c["tf"][t] = c["tf"].get(t, 0) + 1
        c["len"] = sum(c["tf"].values())
        for t in c["tf"]:
            df[t] = df.get(t, 0) + 1
    n = max(len(chunks), 1)
    INDEX["idf"] = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    INDEX["avglen"] = sum(c["len"] for c in chunks) / n if chunks else 1


def bm25(qt, c, k1=1.4, b=0.75):
    idf, avg, s = INDEX["idf"], INDEX["avglen"], 0.0
    for t in qt:
        f = c["tf"].get(t)
        if f:
            s += idf.get(t, 0) * f * (k1 + 1) / (f + k1 * (1 - b + b * c["len"] / avg))
    return s


def search(q, k=12, fusion="interleave"):
    """Гибридный поиск: смысл (эмбеддинги) + совпадение слов (BM25), объединение рангов (RRF)."""
    t0 = time.time()
    if q in QCACHE:
        qv, qtok = QCACHE[q], 0
    else:
        v, qtok = embed([q])
        qv = QCACHE[q] = norm(v[0])
        if len(QCACHE) > 500:
            QCACHE.pop(next(iter(QCACHE)))
    t_emb = time.time() - t0
    t1 = time.time()
    qt = toks(q)
    with INDEX_LOCK:
        chunks = INDEX["chunks"]
        cos = [sum(a * b for a, b in zip(qv, c["v"])) for c in chunks]
        bm = [bm25(qt, c) for c in chunks]
    r_cos = {i: r for r, i in enumerate(sorted(range(len(chunks)), key=lambda i: -cos[i]))}
    r_bm = {i: r for r, i in enumerate(sorted(range(len(chunks)), key=lambda i: -bm[i]))}
    if fusion == "rrf":
        fused = sorted(range(len(chunks)), key=lambda i: -(1.0 / (60 + r_cos[i]) + (1.0 / (60 + r_bm[i]) if bm[i] > 0 else 0)))
    else:  # interleave: по очереди лучший по словам и лучший по смыслу — точные формулировки не теряются
        by_bm = [i for i in sorted(range(len(chunks)), key=lambda i: -bm[i]) if bm[i] > 0]
        by_cos = sorted(range(len(chunks)), key=lambda i: -cos[i])
        fused, seen = [], set()
        for pair in zip(by_bm + [None] * len(by_cos), by_cos):
            for i in pair:
                if i is not None and i not in seen:
                    seen.add(i)
                    fused.append(i)
            if len(fused) >= k:
                break
    scored = [(cos[i], chunks[i]) for i in fused[:k]]
    t_search = time.time() - t1
    return scored, t_emb, t_search, qtok


def rerank(q, cands, top_n=4):
    t0 = time.time()
    try:
        j = ait("/rerank", {"model": RERANK_MODEL, "query": q, "documents": [c["text"] for _, c in cands], "top_n": top_n}, timeout=30)
        res = j.get("results") or j.get("data") or []
        picked = [(r.get("relevance_score", r.get("score", 0)), cands[r["index"]][1]) for r in res][:top_n]
        ok = bool(picked)
    except Exception as e:  # реранк не обязателен — откатываемся на косинус
        picked, ok = [], False
        err = str(e)[:120]
    if not ok:
        picked = [(s, c) for s, c in cands[:top_n]]
    return picked, time.time() - t0, ok


# ---------------------------------------------------------------- каталоги моделей провайдеров
SKIP = re.compile(r"(embed|rerank|whisper|ocr|bge|e5-|tts|speech|audio|image|video|vision|moderat|guard|transcri|dall|flux|sora|veo|kandinsky|imagen|midjourney|search)", re.I)


def catalog(prov, force=False):
    """Текстовые модели провайдера (кэш на час). GigaChat — первыми."""
    P, C = PROV[prov], CAT[prov]
    if not P["key"] or (C["models"] and not force and time.time() - C["ts"] < 3600):
        return C["models"]
    req = urllib.request.Request(P["base"] + "/models", headers={"Authorization": "Bearer " + P["key"], "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            j = json.loads(r.read())
        items = j.get("data") or j.get("models") or (j if isinstance(j, list) else [])
        C["raw"] = {m.get("id"): m for m in items if isinstance(m, dict) and m.get("id")}
        ids = [i for i in C["raw"] if not SKIP.search(i)] or list(C["raw"])
        ids.sort(key=lambda i: (0 if "gigachat" in i.lower() else 1, i.lower()))
        C.update(models=ids, ts=time.time(), err="")
    except urllib.error.HTTPError as e:
        C["err"] = "%s /models %s: %s" % (P["label"], e.code, e.read()[:200].decode("utf-8", "ignore"))
    except Exception as e:
        C["err"] = "%s /models: %s" % (P["label"], str(e)[:200])
    return C["models"]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def price_of(prov, model_id):
    """(вход, выход) за 1 млн токенов из карточки модели, если провайдер их отдаёт. Best effort."""
    m = CAT[prov]["raw"].get(model_id) or {}
    found = {}

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + [str(k).lower()])
        elif _num(o) is not None and any(w in "".join(path) for w in ("price", "pricing", "cost", "tariff")):
            key = path[-1]
            if "cache" in key:
                return
            if any(w in key for w in ("input", "prompt", "in_")):
                found.setdefault("in", _num(o))
            elif any(w in key for w in ("output", "completion", "generated", "out_")):
                found.setdefault("out", _num(o))
    walk(m, [])
    return (found["in"], found["out"]) if "in" in found and "out" in found else None


def llm(provider, model, msgs, nothink, timeout=90):
    """Один вызов модели у выбранного провайдера. Возвращает (ответ API, id модели)."""
    body = {"messages": msgs, "temperature": 0.2, "max_tokens": 700}
    if provider == "gigachat":
        m = model if model in GIGA_MODELS else GIGA_MODELS[0]
        body["model"] = m
        return giga_chat(body), m
    if provider not in PROV:
        provider = "aitunnel"
    P = PROV[provider]
    if not P["key"]:
        raise RuntimeError("На сервере не задан %s" % P["env"])
    ids = catalog(provider) or (MODELS if provider == "aitunnel" else [])
    m = model if model in ids else (ids[0] if ids else None)
    if not m:
        raise RuntimeError(CAT[provider]["err"] or "Каталог моделей %s пуст" % P["label"])
    body["model"] = m
    if nothink and provider == "aitunnel" and not m.startswith("gemini") and "/gemini" not in m:
        body["reasoning"] = {"effort": "none"}  # у Gemini рассуждение отключить нельзя (API вернёт 400)
    j = post_json(P["base"] + "/chat/completions", body, {"Authorization": "Bearer " + P["key"]}, timeout)
    u = j.get("usage") or {}
    pr = price_of(provider, m)
    if pr and u.get("prompt_tokens") is not None and u.get("cost_rub") is None:
        u["cost_rub"] = round((u.get("prompt_tokens", 0) * pr[0] + u.get("completion_tokens", 0) * pr[1]) / 1e6, 4)
        u["cost_note"] = "по прайсу, без скидки на кэш"
        j["usage"] = u
    return j, m


def cached_of(u):
    u = u or {}
    return (u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("precached_prompt_tokens") or 0


def answer_ait(q, model, nothink, use_rerank=False, top_k=6, fusion="interleave", provider="aitunnel"):
    if not INDEX.get("chunks"):
        raise RuntimeError("Индекс пуст: загрузите документы в /admin и нажмите «Переиндексировать»")
    T0 = time.time()
    cands, t_emb, t_search, qtok = search(q, fusion=fusion)
    if use_rerank:
        top, t_rr, rr_ok = rerank(q, cands, top_n=top_k)
    else:
        top, t_rr, rr_ok = [(s, c) for s, c in cands[:top_k]], 0.0, False
    ctx = "\n\n".join("--- Фрагмент %d. Документ: «%s»\n%s" % (i + 1, c["doc"], c["text"]) for i, (_, c) in enumerate(top))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Фрагменты базы знаний:\n" + ctx + "\n\nВопрос администратора: " + q}]
    t1 = time.time()
    j, model = llm(provider, model, msgs, nothink)
    t_llm = time.time() - t1
    msg = (j.get("choices") or [{}])[0].get("message") or {}
    return {
        "answer": msg.get("content") or "",
        "usage": j.get("usage"),
        "cached": cached_of(j.get("usage")),
        "model": j.get("model") or model,
        "timing": {"embed": round(t_emb, 2), "search": round(t_search, 3), "rerank": round(t_rr, 2),
                   "llm": round(t_llm, 2), "total": round(time.time() - T0, 2)},
        "rerank_ok": rr_ok,
        "sources": ["%s (%.2f)" % (c["doc"], s) for s, c in top],
    }


# ---------------------------------------------------------------- full context (вся база в промпте)
FULL = {"text": "", "chars": 0}

SYSTEM_FULL = SYSTEM.replace("по фрагментам базы знаний ниже", "по базе знаний ниже").replace("Цифры, сроки, телефоны и фамилии бери точно из фрагментов", "Цифры, сроки, телефоны и фамилии бери точно из базы").replace("Если во фрагментах нет ответа", "Если в базе нет ответа").replace("Если фрагменты противоречат", "Если документы противоречат")


def build_full():
    parts = []
    for fn in sorted(os.listdir(DOCS_DIR)):
        if fn.endswith((".md", ".txt")):
            with open(os.path.join(DOCS_DIR, fn), encoding="utf-8") as f:
                t = f.read()
            t = re.sub(r"https?://\S+", "", t)            # ссылки модели не нужны
            t = re.sub(r"\|\s*:-:\s*", "|", t)
            t = re.sub(r"[ \t]+", " ", t)
            t = re.sub(r"\n{2,}", "\n", t)
            parts.append("=== ДОКУМЕНТ: %s ===\n%s" % (fn, t.strip()))
    FULL["text"] = "\n\n".join(parts)
    FULL["chars"] = len(FULL["text"])


GIGA = {"token": None, "exp": 0}


def giga_ctx():
    import ssl
    ctx = ssl.create_default_context()
    if os.path.exists(GIGA_CA):
        ctx.load_verify_locations(GIGA_CA)
    return ctx


def giga_token():
    if GIGA["token"] and time.time() < GIGA["exp"] - 60:
        return GIGA["token"]
    if not GIGA_KEY:
        raise RuntimeError("На сервере не задан GIGACHAT_AUTH_KEY")
    req = urllib.request.Request("https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                                 data=urllib.parse.urlencode({"scope": GIGA_SCOPE}).encode(), method="POST",
                                 headers={"Authorization": "Basic " + GIGA_KEY, "RqUID": str(uuid.uuid4()),
                                          "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=giga_ctx()) as r:
            j = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError("GigaChat OAuth %s: %s" % (e.code, e.read()[:200].decode("utf-8", "ignore")))
    GIGA["token"] = j["access_token"]
    GIGA["exp"] = j.get("expires_at", 0) / 1000 or time.time() + 1500
    return GIGA["token"]


def giga_chat(body):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request("https://gigachat.devices.sberbank.ru/api/v1/chat/completions", data=data, method="POST",
                                 headers={"Authorization": "Bearer " + giga_token(), "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120, context=giga_ctx()) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError("GigaChat %s: %s" % (e.code, e.read()[:300].decode("utf-8", "ignore")))


def answer_full(q, provider, model, nothink):
    if not FULL["text"]:
        build_full()
    if not FULL["text"]:
        raise RuntimeError("Нет документов: загрузите их в /admin")
    msgs = [{"role": "system", "content": SYSTEM_FULL + "\n\nБАЗА ЗНАНИЙ:\n" + FULL["text"]},
            {"role": "user", "content": "Вопрос администратора: " + q}]
    t0 = time.time()
    j, m = llm(provider, model, msgs, nothink, timeout=120)
    t = time.time() - t0
    msg = (j.get("choices") or [{}])[0].get("message") or {}
    u = j.get("usage") or {}
    cached = cached_of(u)
    return {"answer": msg.get("content") or "", "usage": u, "cached": cached, "model": j.get("model") or m,
            "timing": {"llm": round(t, 2), "total": round(t, 2)}, "sources": ["вся база: %d тыс. символов" % (FULL["chars"] // 1000)]}


def answer_tw(q, nothink):
    if not TW_TOKEN:
        raise RuntimeError("На сервере не задан TW_TOKEN")
    body = {"messages": [{"role": "user", "content": q}], "stream": False}
    if nothink:
        body["thinking"] = {"type": "disabled"}
    t0 = time.time()
    j = post_json(TW_URL, body, {"Authorization": "Bearer " + TW_TOKEN})
    t = time.time() - t0
    msg = (j.get("choices") or [{}])[0].get("message") or {}
    return {"answer": msg.get("content") or "", "usage": j.get("usage"), "model": j.get("model") or "timeweb-agent",
            "timing": {"llm": round(t, 2), "total": round(t, 2)}, "sources": []}


# ---------------------------------------------------------------- HTML
STYLE = """
:root{--bg:#f6f7f9;--card:#fff;--text:#1c2230;--muted:#5d6678;--accent:#5b5ef0;--border:#e3e6ec;--q:#eef0ff;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#141821;--card:#1c2230;--text:#e8ebf2;--muted:#a2aabb;--accent:#8f92ff;--border:#2c3444;--q:#262b45;--bad:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,"Segoe UI",Roboto,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 190px}h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 16px;font-size:14px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 16px;margin-bottom:12px}.q{background:var(--q);border-color:transparent;font-weight:600}
.meta{color:var(--muted);font-size:12px;margin-top:8px;display:flex;gap:10px;flex-wrap:wrap}.fast{color:#1e8e3e}.slow{color:#c26a00}.err{color:var(--bad)}
.src{color:var(--muted);font-size:12px;margin-top:4px}
.bar{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--border);padding:12px 16px}.bar .in{max-width:860px;margin:0 auto}
.row{display:flex;gap:8px}textarea{flex:1;resize:none;height:52px;padding:10px 12px;border-radius:10px;border:1px solid var(--border);background:var(--bg);color:var(--text);font:inherit}
button{border:0;border-radius:10px;padding:0 18px;background:var(--accent);color:#fff;font:inherit;font-weight:600;cursor:pointer}button:disabled{opacity:.5}
select{padding:4px 8px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font:inherit;font-size:13px}
.opts{display:flex;gap:14px;align-items:center;margin-top:8px;font-size:13px;color:var(--muted);flex-wrap:wrap}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px}.chip{background:transparent;color:var(--text);border:1px solid var(--border);border-radius:999px;padding:5px 11px;font-size:13px;font-weight:400}
.login{max-width:340px;margin:80px auto}.login input{width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--border);background:var(--bg);color:var(--text);font:inherit;margin:10px 0}.login button{width:100%;height:42px}
.ans p{margin:0 0 6px}.ans ul{margin:0 0 6px;padding-left:20px}a{color:var(--accent)}table{border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid var(--border);padding:4px 8px;text-align:left}
"""

MAIN = """
<main>
<h1>Помощник КЦ — тест</h1>
<p class="sub">Сравнение: агент Timeweb, свой RAG (поиск по кускам) и «вся база в промпте»; модели — AITunnel, Cloud.ru, Timeweb AI Gateway или GigaChat. Под ответом — время по этапам и токены. Не вводите данные пациентов. <a href="admin">Документы и индекс</a> · <a href="logout">Выйти</a></p>
<div class="chips" id="chips">
<button class="chip">пациент с отеком щеки и температурой что делать</button><button class="chip">бабушка привела ребенка 10 лет на первый прием</button>
<button class="chip">какая гарантия на коронку</button><button class="chip">сколько стоит профосмотр</button><button class="chip">сколько стоит имплантация под ключ</button></div>
<div id="log"></div><div class="card" id="stats" style="display:none"></div></main>
<div class="bar"><div class="in"><div class="row"><textarea id="q" placeholder="Вопрос… (Enter — отправить)"></textarea><button id="send">Спросить</button></div>
<div class="opts"><label>Движок <select id="engine"><option value="aitunnel">Свой RAG (поиск по кускам)</option><option value="full">Вся база в промпте</option><option value="timeweb">Timeweb агент</option></select></label>
<label id="provwrap">Провайдер <select id="provider"><option value="aitunnel">AITunnel</option><option value="cloudru">Cloud.ru</option><option value="twgw">Timeweb AI Gateway</option><option value="gigachat">GigaChat (Сбер)</option></select></label>
<label>Модель <select id="model">__MODELS__</select><select id="gmodel" style="display:none">__GMODELS__</select><select id="cmodel" style="display:none">__CMODELS__</select><select id="tmodel" style="display:none">__TMODELS__</select></label>
<label><input type="checkbox" id="nothink" checked> без рассуждения</label><label><input type="checkbox" id="rerank"> реранк (+0,45 ₽)</label><span id="status"></span></div></div></div>
<script>
(function(){
var log=document.getElementById('log'),q=document.getElementById('q'),send=document.getElementById('send'),st=document.getElementById('status'),stats=document.getElementById('stats'),T={};
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]})}
function md(s){var L=esc(s).split('\\n'),h='',ul=false;L.forEach(function(l){var m=l.match(/^\\s*(?:[-*•]|\\d+[.)])\\s+(.*)/);l=(m?m[1]:l).replace(/\\*\\*(.+?)\\*\\*/g,'<b>$1</b>');if(m){if(!ul){h+='<ul>';ul=true}h+='<li>'+l+'</li>'}else{if(ul){h+='</ul>';ul=false}if(l.trim())h+='<p>'+l+'</p>'}});return h+(ul?'</ul>':'')}
function med(a){var b=a.slice().sort(function(x,y){return x-y}),m=Math.floor(b.length/2);return b.length%2?b[m]:(b[m-1]+b[m])/2}
function upd(){var p=[];for(var k in T){p.push(k+': '+T[k].length+' вопр., медиана '+med(T[k]).toFixed(1)+' с')}stats.style.display=p.length?'':'none';stats.textContent='Итого — '+p.join(' · ')}
window.kcAsk=function(text,opts){opts=opts||{};text=String(text).trim();if(!text)return Promise.resolve();
var engine=opts.engine||document.getElementById('engine').value,provider=opts.provider||document.getElementById('provider').value,model=opts.model||document.getElementById(engine==='timeweb'?'model':({gigachat:'gmodel',cloudru:'cmodel',twgw:'tmodel'}[provider]||'model')).value,nothink=opts.nothink!=null?opts.nothink:document.getElementById('nothink').checked,rr=opts.rerank!=null?opts.rerank:document.getElementById('rerank').checked;
var pn={aitunnel:'AITunnel',cloudru:'Cloud.ru',twgw:'Timeweb GW',gigachat:'GigaChat'}[provider]||provider;
var label=engine==='timeweb'?'Timeweb агент':(engine==='full'?'Вся база · '+pn+' · '+model:'RAG · '+pn+' · '+model+(nothink?'':' · с рассуждением')+(rr?' · реранк':''));
var qd=document.createElement('div');qd.className='card q';qd.textContent=text;log.appendChild(qd);var ad=document.createElement('div');ad.className='card';ad.innerHTML='<span class="meta">Жду ответ…</span>';log.appendChild(ad);ad.scrollIntoView({block:'end'});
send.disabled=true;q.value='';var t0=performance.now(),tm=setInterval(function(){st.textContent=((performance.now()-t0)/1000).toFixed(1)+' с…'},100);
return fetch('api',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q:text,engine:engine,provider:provider,model:model,nothink:nothink,rerank:rr})}).then(function(r){return r.json()}).then(function(j){
var sec=(performance.now()-t0)/1000;if(j.error){ad.innerHTML='<div class="err">'+esc(j.error)+'</div>';return {error:j.error}}
(T[label]=T[label]||[]).push(sec);upd();var u=j.usage||{},t=j.timing||{};
var parts=[];if(t.embed!=null)parts.push('эмбеддинг '+t.embed+' с');if(t.search!=null)parts.push('поиск '+t.search+' с');if(t.rerank)parts.push('реранк '+t.rerank+' с'+(j.rerank_ok?'':' (не сработал)'));parts.push('модель '+t.llm+' с');
ad.innerHTML='<div class="ans">'+md(j.answer||'(пусто)')+'</div>'+(j.sources&&j.sources.length?'<div class="src">Фрагменты: '+esc(j.sources.join('; '))+'</div>':'')+
'<div class="meta"><span class="'+(sec<=5?'fast':'slow')+'">⏱ '+sec.toFixed(1)+' с всего</span><span>'+parts.join(' · ')+'</span>'+(u.prompt_tokens?'<span>токены: '+u.prompt_tokens+(j.cached?' (кэш '+j.cached+')':'')+' / '+u.completion_tokens+'</span>':'')+(u.cost_rub!=null?'<span>'+(u.cost_note?'≈ ':'')+u.cost_rub+' ₽</span>':'')+'<span>'+esc(label)+'</span></div>';
return {sec:sec,j:j,label:label}}).catch(function(e){ad.innerHTML='<div class="err">'+esc(e)+'</div>'}).then(function(x){clearInterval(tm);st.textContent='';send.disabled=false;return x})};
send.onclick=function(){kcAsk(q.value)};q.onkeydown=function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();if(!send.disabled)kcAsk(q.value)}};
document.getElementById('chips').onclick=function(e){var b=e.target.closest('.chip');if(b&&!send.disabled)kcAsk(b.textContent)};
function sync(){var e=document.getElementById('engine').value,p=e==='timeweb'?'aitunnel':document.getElementById('provider').value;document.getElementById('provwrap').style.display=e==='timeweb'?'none':'';document.getElementById('model').style.display=p==='aitunnel'?'':'none';document.getElementById('gmodel').style.display=p==='gigachat'?'':'none';document.getElementById('cmodel').style.display=p==='cloudru'?'':'none';document.getElementById('tmodel').style.display=p==='twgw'?'':'none';document.getElementById('model').disabled=e==='timeweb';document.getElementById('rerank').parentNode.style.display=e==='aitunnel'?'':'none'}
document.getElementById('engine').onchange=sync;document.getElementById('provider').onchange=sync;sync();
})();
</script>
"""

ADMIN = """
<main><h1>Документы и индекс</h1><p class="sub"><a href="./">← к чату</a></p>
<div class="card" id="state">…</div>
<div class="card"><b>Загрузить документы (.md / .txt)</b><p class="sub">Файлы с тем же именем заменяются. После загрузки нажмите «Переиндексировать».</p>
<input type="file" id="files" multiple accept=".md,.txt"> <button id="up">Загрузить</button> <button id="rx">Переиндексировать</button> <span id="msg"></span></div></main>
<script>
function esc(s){return String(s).replace(/[&<>]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;'}[c]})}
function load(){fetch('admin/state').then(function(r){return r.json()}).then(function(s){var h='<b>Индекс:</b> '+(s.built?('собран '+s.built+', кусков: '+s.chunks+', токенов эмбеддинга: '+s.emb_tokens):'не собран')+'<br><b>Документы ('+s.docs.length+'):</b><table>';s.docs.forEach(function(d){h+='<tr><td>'+esc(d.name)+'</td><td>'+d.kb+' КБ</td><td>'+d.chunks+' кусков</td></tr>'});document.getElementById('state').innerHTML=h+'</table>'})}
document.getElementById('up').onclick=async function(){var fs=document.getElementById('files').files,m=document.getElementById('msg');for(var i=0;i<fs.length;i++){m.textContent='Загружаю '+fs[i].name+'…';var t=await fs[i].text();var r=await fetch('admin/upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:fs[i].name,content:t})});if(!r.ok){m.textContent='Ошибка: '+(await r.text());return}}m.textContent='Загружено файлов: '+fs.length;load()};
document.getElementById('rx').onclick=function(){var m=document.getElementById('msg');m.textContent='Индексирую… (до минуты)';var t0=Date.now();fetch('admin/reindex',{method:'POST'}).then(function(r){return r.json()}).then(function(j){m.textContent=j.error?('Ошибка: '+j.error):('Готово за '+((Date.now()-t0)/1000).toFixed(1)+' с: кусков '+j.chunks);load()})};
load();
</script>
"""


def page(body):
    return ("<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<meta name='robots' content='noindex, nofollow'><title>Помощник КЦ — тест</title><style>" + STYLE +
            "</style></head><body>" + body + "</body></html>").encode("utf-8")


def login_page(err=""):
    e = "<div class='err'>%s</div>" % err if err else ""
    return page("<form class='card login' method='post' action='login'><h1>Помощник КЦ</h1><p class='sub'>Тестовая страница. Введите пароль.</p>"
                "<input type='password' name='password' autofocus placeholder='Пароль'>" + e + "<button type='submit'>Войти</button></form>")



EMBED = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow"><title>Помощник КЦ</title><style>
:root{--bg:#f6f7f9;--card:#fff;--text:#1c2230;--muted:#5d6678;--accent:#0f6f73;--border:#dfe4ea;--q:#e8f2f1;--bad:#b3261e}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 "Segoe UI",Roboto,Arial,sans-serif;display:flex;flex-direction:column}
#log{flex:1;overflow-y:auto;padding:12px 14px}
.hint{color:var(--muted);font-size:13px;margin:4px 0 10px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:999px;padding:4px 10px;font:inherit;font-size:13px;cursor:pointer}
.chip:hover{border-color:var(--accent);color:var(--accent)}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin:0 0 8px}
.q{background:var(--q);border-color:transparent;font-weight:600}
.ans p{margin:0 0 5px}.ans ul{margin:0 0 5px;padding-left:18px}.ans li{margin:0 0 3px}
.meta{color:var(--muted);font-size:12px;margin-top:6px}.err{color:var(--bad)}
.bar{border-top:1px solid var(--border);background:var(--card);padding:10px 14px;display:flex;gap:8px}
textarea{flex:1;resize:none;height:44px;padding:9px 11px;border-radius:9px;border:1px solid var(--border);background:var(--bg);color:var(--text);font:inherit}
button.send{border:0;border-radius:9px;padding:0 16px;background:var(--accent);color:#fff;font:inherit;font-weight:600;cursor:pointer}
button.send:disabled{opacity:.5}
</style></head><body>
<div id="log"><div class="hint">Задайте вопрос по регламенту КЦ. Ответ строится только по базе знаний. Не вводите ФИО, телефоны и диагнозы пациентов.</div>
<div class="chips" id="chips"><button class="chip">пациент с отеком щеки и температурой что делать</button><button class="chip">бабушка привела ребенка 10 лет</button><button class="chip">какая гарантия на коронку</button></div></div>
<div class="bar"><textarea id="q" placeholder="Вопрос… (Enter — отправить)"></textarea><button class="send" id="send">Спросить</button></div>
<script>
(function(){
var API="__API__", KEY="__KEY__";
var log=document.getElementById('log'),q=document.getElementById('q'),send=document.getElementById('send');
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]})}
function md(s){var L=esc(s).split('\\n'),h='',ul=false;L.forEach(function(l){var m=l.match(/^\\s*(?:[-*•]|\\d+[.)])\\s+(.*)/);l=(m?m[1]:l).replace(/\\*\\*(.+?)\\*\\*/g,'<b>$1</b>');if(m){if(!ul){h+='<ul>';ul=true}h+='<li>'+l+'</li>'}else{if(ul){h+='</ul>';ul=false}if(l.trim())h+='<p>'+l+'</p>'}});return h+(ul?'</ul>':'')}
function ask(text){text=String(text||'').trim();if(!text||send.disabled)return;
var qd=document.createElement('div');qd.className='card q';qd.textContent=text;log.appendChild(qd);
var ad=document.createElement('div');ad.className='card';ad.innerHTML='<span class="meta">Ищу в базе…</span>';log.appendChild(ad);log.scrollTop=log.scrollHeight;
send.disabled=true;q.value='';var t0=Date.now();
var x=new XMLHttpRequest();x.open('POST',API,true);x.setRequestHeader('Content-Type','text/plain;charset=UTF-8');x.timeout=60000;
x.onload=function(){var j={};try{j=JSON.parse(x.responseText)}catch(e){j={error:'Сервер ответил не JSON ('+x.status+')'}}
 if(j.error){ad.innerHTML='<div class="err">'+esc(j.error)+'</div>'}else{ad.innerHTML='<div class="ans">'+md(j.answer||'(пусто)')+'</div><div class="meta">'+((Date.now()-t0)/1000).toFixed(1)+' с</div>'}done()};
x.onerror=function(){ad.innerHTML='<div class="err">Нет связи с сервером помощника ('+esc(API)+')</div>';done()};
x.ontimeout=function(){ad.innerHTML='<div class="err">Сервер не ответил за 60 с</div>';done()};
x.send(JSON.stringify({q:text,k:KEY}))}
function done(){send.disabled=false;log.scrollTop=log.scrollHeight;q.focus()}
send.onclick=function(){ask(q.value)};
q.onkeydown=function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask(q.value)}};
document.getElementById('chips').onclick=function(e){var b=e.target.closest?e.target.closest('.chip'):null;if(b)ask(b.textContent)};
})();
</script></body></html>"""


def embed_ok(k):
    return bool(EMBED_KEY) and hmac.compare_digest(str(k or "").encode(), EMBED_KEY.encode())


def embed_answer(q):
    day = time.strftime("%Y-%m-%d")
    if EMBED_COUNT["day"] != day:
        EMBED_COUNT.update(day=day, n=0)
    EMBED_COUNT["n"] += 1
    if EMBED_COUNT["n"] > EMBED_DAY_LIMIT:
        raise RuntimeError("Дневной лимит вопросов исчерпан")
    try:
        return answer_ait(q, EMBED_MODEL, True, provider=EMBED_PROVIDER)
    except Exception:
        if not EMBED_FALLBACK:
            raise
        return answer_ait(q, EMBED_FALLBACK, True, provider=EMBED_PROVIDER)


def public_base(h):
    """Внешний адрес сервера по заголовкам запроса (для абсолютной ссылки на API в виджете)."""
    host = h.headers.get("X-Forwarded-Host") or h.headers.get("Host") or ("localhost:%d" % PORT)
    proto = h.headers.get("X-Forwarded-Proto") or "http"
    return "%s://%s" % (proto, re.sub(r"[^\w.:\-\[\]]", "", host))


class H(BaseHTTPRequestHandler):
    server_version = "kc-test"

    def sid(self):
        c = http.cookies.SimpleCookie(self.headers.get("Cookie", ""))
        v = c.get("kc_sid")
        return v.value if v and v.value in SESSIONS else None

    def send(self, code, body, ctype="text/html; charset=utf-8", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("X-Robots-Tag", "noindex, nofollow")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def js(self, code, obj, headers=None):
        self.send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", headers)

    CORS = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Max-Age": "86400"}

    def do_OPTIONS(self):
        if self.route().endswith("/embed/api"):
            return self.send(204, b"", headers=self.CORS)
        self.send(404, b"not found")

    def route(self):
        p = urllib.parse.urlparse(self.path).path
        return p.rstrip("/")

    def do_GET(self):
        p = self.route()
        if p.endswith("/logout"):
            SESSIONS.pop(self.sid(), None)
            save_sessions()
            return self.send(302, b"", headers={"Location": "./", "Set-Cookie": "kc_sid=; Max-Age=0; Path=/"})
        if p.endswith("/embed"):
            k = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("k", [""])[0]
            if not embed_ok(k):
                return self.send(403, "<p style='font:14px sans-serif'>Нет доступа: неверный ключ виджета (параметр k).</p>".encode("utf-8"))
            api = public_base(self) + "/embed/api"
            return self.send(200, EMBED.replace("__API__", api).replace("__KEY__", k).encode("utf-8"))
        if not self.sid():
            return self.send(200, login_page())
        if p.endswith("/admin/state"):
            counts = {}
            for c in INDEX.get("chunks", []):
                counts[c["file"]] = counts.get(c["file"], 0) + 1
            docs = [{"name": fn, "kb": round(os.path.getsize(os.path.join(DOCS_DIR, fn)) / 1024, 1), "chunks": counts.get(fn, 0)}
                    for fn in sorted(os.listdir(DOCS_DIR))]
            return self.js(200, {"version": VERSION, "built": INDEX.get("built"), "chunks": len(INDEX.get("chunks", [])),
                                 "emb_tokens": INDEX.get("emb_tokens"), "docs": docs})
        if p.endswith("/admin"):
            return self.send(200, page(ADMIN))
        mm = re.search(r"/(aitunnel|cloudru|twgw)/models$", p)
        if mm:
            pv = mm.group(1)
            ids = catalog(pv, force=True)
            return self.js(200, {"models": ids, "err": CAT[pv]["err"], "prices": {i: price_of(pv, i) for i in ids},
                                 "raw": {i: CAT[pv]["raw"].get(i) for i in ids} if "raw=1" in self.path else None})
        opts = "".join("<option>%s</option>" % m for m in MODELS)
        gopts = "".join("<option>%s</option>" % m for m in GIGA_MODELS)
        def opts_for(pv, first=()):
            ids = catalog(pv)
            ids = [m for m in first if m in ids] + [m for m in ids if m not in first]
            return "".join("<option>%s</option>" % re.sub(r"[<>&\"]", "", m) for m in ids) or \
                "<option value=''>%s</option>" % ("нет " + PROV[pv]["env"] if not PROV[pv]["key"] else "каталог недоступен")
        if catalog("aitunnel"):
            opts = opts_for("aitunnel", MODELS)
        return self.send(200, page(MAIN.replace("__MODELS__", opts).replace("__GMODELS__", gopts)
                                   .replace("__CMODELS__", opts_for("cloudru")).replace("__TMODELS__", opts_for("twgw"))))

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(min(n, 5_000_000))
        p = self.route()
        if p.endswith("/login"):
            pw = urllib.parse.parse_qs(raw.decode("utf-8", "ignore")).get("password", [""])[0]
            if hmac.compare_digest(pw.encode(), PASSWORD.encode()):
                s = secrets.token_urlsafe(24)
                SESSIONS[s] = 0
                save_sessions()
                return self.send(302, b"", headers={"Location": "./", "Set-Cookie": "kc_sid=%s; HttpOnly; SameSite=Strict; Path=/; Max-Age=604800" % s})
            time.sleep(1)
            return self.send(200, login_page("Неверный пароль"))
        if p.endswith("/embed/api"):
            try:
                data = json.loads(raw or b"{}")
            except ValueError:
                data = {}
            if not embed_ok(data.get("k")):
                return self.js(403, {"error": "Неверный ключ виджета"}, self.CORS)
            q = str(data.get("q", "")).strip()
            if not q or len(q) > 1000:
                return self.js(400, {"error": "Пустой или слишком длинный вопрос"}, self.CORS)
            try:
                r = embed_answer(q)
                return self.js(200, {"answer": r["answer"], "model": r.get("model"), "timing": r.get("timing")}, self.CORS)
            except Exception as e:
                return self.js(502, {"error": "Помощник недоступен: " + str(e)[:200]}, self.CORS)
        s = self.sid()
        if not s:
            return self.js(401, {"error": "Нужно войти заново"})
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            data = {}
        try:
            if p.endswith("/admin/upload"):
                name = os.path.basename(str(data.get("name", "")))
                if not re.fullmatch(r"[\w.\- ]{1,120}\.(md|txt)", name):
                    return self.js(400, {"error": "Имя файла: только .md или .txt"})
                with open(os.path.join(DOCS_DIR, name), "w", encoding="utf-8") as f:
                    f.write(str(data.get("content", "")))
                FULL["text"] = ""
                return self.js(200, {"ok": True})
            if p.endswith("/admin/reindex"):
                idx = build_index()
                return self.js(200, {"chunks": len(idx["chunks"]), "emb_tokens": idx["emb_tokens"]})
            if p.endswith("/api"):
                SESSIONS[s] += 1
                if SESSIONS[s] > 500:
                    return self.js(429, {"error": "Лимит вопросов в этой сессии исчерпан"})
                q = str(data.get("q", "")).strip()
                if not q or len(q) > 1000:
                    return self.js(400, {"error": "Пустой или слишком длинный вопрос"})
                if data.get("engine") == "timeweb":
                    return self.js(200, answer_tw(q, bool(data.get("nothink"))))
                if data.get("engine") == "full":
                    return self.js(200, answer_full(q, data.get("provider") or "aitunnel", data.get("model"), bool(data.get("nothink"))))
                prov = data.get("provider") if data.get("provider") in ("aitunnel", "cloudru", "twgw", "gigachat") else "aitunnel"
                return self.js(200, answer_ait(q, data.get("model"), bool(data.get("nothink")), data.get("rerank") is True, min(int(data.get("top_k") or 6), 12), "rrf" if data.get("fusion") == "rrf" else "interleave", prov))
        except Exception as e:
            return self.js(502, {"error": str(e)[:400]})
        self.send(404, b"not found")

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    load_sessions()
    load_index()
    print("kc-test on port", PORT, "chunks:", len(INDEX.get("chunks", [])))
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
PYEOF
ENVF=/opt/kc-test/env
umask 077
touch "$ENVF"
getv() { grep -E "^$1=" "$ENVF" | tail -1 | cut -d= -f2- || true; }
if [ -z "$(getv PAGE_PASSWORD)" ]; then
  read -r -s -p "Придумайте пароль для страницы: " PW </dev/tty; echo
  echo "PAGE_PASSWORD=$PW" >> "$ENVF"
fi
if [ -z "$(getv TW_TOKEN)" ]; then
  read -r -s -p "Токен агента Timeweb (Enter — пропустить): " TOK </dev/tty; echo
  if [ -n "$TOK" ]; then echo "TW_TOKEN=$TOK" >> "$ENVF"; fi
fi
if [ -z "$(getv AITUNNEL_KEY)" ]; then
  read -r -s -p "API-ключ AITunnel (Enter — пропустить): " AK </dev/tty; echo
  if [ -n "$AK" ]; then echo "AITUNNEL_KEY=$AK" >> "$ENVF"; fi
fi
# Ключ Cloud.ru, по ошибке сохранённый как ключ GigaChat (у Сбера ключ — base64 без дефисов), переносим
if [ -z "$(getv CLOUDRU_KEY)" ] && getv GIGACHAT_AUTH_KEY | grep -q -- "-"; then
  sed -i 's/^GIGACHAT_AUTH_KEY=/CLOUDRU_KEY=/; /^GIGACHAT_SCOPE=/d' "$ENVF"
  echo "Ключ с дефисами перенесён из GIGACHAT_AUTH_KEY в CLOUDRU_KEY (это ключ Cloud.ru)"
fi
if [ -z "$(getv CLOUDRU_KEY)" ]; then
  read -r -s -p "API-ключ Cloud.ru Foundation Models (Enter — пропустить): " CK </dev/tty; echo
  if [ -n "$CK" ]; then echo "CLOUDRU_KEY=$CK" >> "$ENVF"; fi
fi
if [ -z "$(getv TWGW_KEY)" ]; then
  read -r -s -p "API-ключ Timeweb AI Gateway (панель Timeweb → AI Gateway → API-ключи; Enter — пропустить): " TK </dev/tty; echo
  if [ -n "$TK" ]; then echo "TWGW_KEY=$TK" >> "$ENVF"; fi
fi
if [ -z "$(getv GIGACHAT_AUTH_KEY)" ]; then
  read -r -s -p "Ключ авторизации GigaChat (Authorization key из кабинета Сбера; Enter — пропустить): " GK </dev/tty; echo
  if [ -n "$GK" ]; then
    echo "GIGACHAT_AUTH_KEY=$GK" >> "$ENVF"
    read -r -p "Scope GigaChat (Enter = GIGACHAT_API_CORP; для физлица — GIGACHAT_API_PERS): " GS </dev/tty
    echo "GIGACHAT_SCOPE=${GS:-GIGACHAT_API_CORP}" >> "$ENVF"
  fi
fi
# Корневой сертификат НУЦ Минцифры — нужен для API GigaChat
if ! grep -q "BEGIN CERTIFICATE" /opt/kc-test/russian_trusted_root_ca.pem 2>/dev/null; then
  TMPCA=$(mktemp)
  curl -fsSL https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt >> "$TMPCA" 2>/dev/null || true
  echo >> "$TMPCA"
  curl -fsSL https://gu-st.ru/content/lending/russian_trusted_sub_ca_pem.crt >> "$TMPCA" 2>/dev/null || true
  if grep -q "BEGIN CERTIFICATE" "$TMPCA"; then install -m 644 "$TMPCA" /opt/kc-test/russian_trusted_root_ca.pem; else echo "Внимание: не удалось скачать сертификат Минцифры (нужен только для GigaChat)"; fi
  rm -f "$TMPCA"
fi
# Ключ виджета /embed (для 1С и iframe) — генерируется один раз
if [ -z "$(getv EMBED_KEY)" ]; then
  echo "EMBED_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(18))')" >> "$ENVF"
fi
[ -n "$(getv TW_AGENT_ID)" ] || echo "TW_AGENT_ID=d94b4d73-2ce5-4006-a6f5-4039b60c2d9e" >> "$ENVF"
[ -n "$(getv PORT)" ] || echo "PORT=8790" >> "$ENVF"
PORT=$(getv PORT)
cat > /etc/systemd/system/kc-test.service <<UNIT
[Unit]
Description=KC test page (Timeweb agent + AITunnel RAG)
After=network-online.target

[Service]
EnvironmentFile=/opt/kc-test/env
Environment=DATA_DIR=/var/lib/kc-test
ExecStart=/usr/bin/python3 /opt/kc-test/app.py
DynamicUser=yes
StateDirectory=kc-test
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable kc-test >/dev/null 2>&1 || true
systemctl reset-failed kc-test 2>/dev/null || true
systemctl restart kc-test
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then ufw allow ${PORT}/tcp >/dev/null; fi
sleep 1.5
if systemctl is-active --quiet kc-test; then
  IP=$(hostname -I | awk '{print $1}')
  echo "Готово: http://${IP}:${PORT}/"
  echo "Виджет для 1С / iframe: http://${IP}:${PORT}/embed?k=$(getv EMBED_KEY)"
  echo "  (если сервер виден снаружи под другим IP — подставьте его вместо ${IP})"
  grep -q '^AITUNNEL_KEY=' "$ENVF" && echo "Ключ AITunnel: задан" || echo "Ключ AITunnel: НЕ задан"
  grep -q '^CLOUDRU_KEY=' "$ENVF" && echo "Ключ Cloud.ru: задан" || echo "Ключ Cloud.ru: не задан"
  grep -q '^TWGW_KEY=' "$ENVF" && echo "Ключ Timeweb AI Gateway: задан" || echo "Ключ Timeweb AI Gateway: не задан"
  grep -q '^GIGACHAT_AUTH_KEY=' "$ENVF" && echo "Ключ GigaChat: задан" || echo "Ключ GigaChat: не задан (можно добавить позже, запустив установку ещё раз)"
else
  echo "Сервис не запустился:"; journalctl -u kc-test -n 20 --no-pager
fi
