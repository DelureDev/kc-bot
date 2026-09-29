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
AIT_BASE, EMB_MODEL, RERANK_MODEL, DATA_DIR.
"""
import hmac, http.cookies, json, math, os, re, secrets, threading, time
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

MODELS = ["gemini-3.5-flash-lite", "deepseek-v4.1-flash", "gpt-5.4-mini", "qwen3.8-flash", "claude-haiku-4.5"]
SESSIONS = {}
SESS_FILE = os.path.join(DATA_DIR, "sessions.json")
VERSION = "3"


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


def search(q, k=12):
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
    fused = sorted(range(len(chunks)), key=lambda i: -(1.0 / (60 + r_cos[i]) + (1.0 / (60 + r_bm[i]) if bm[i] > 0 else 0)))
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


def answer_ait(q, model, nothink, use_rerank=False, top_k=6):
    if not INDEX.get("chunks"):
        raise RuntimeError("Индекс пуст: загрузите документы в /admin и нажмите «Переиндексировать»")
    T0 = time.time()
    cands, t_emb, t_search, qtok = search(q)
    if use_rerank:
        top, t_rr, rr_ok = rerank(q, cands, top_n=top_k)
    else:
        top, t_rr, rr_ok = [(s, c) for s, c in cands[:top_k]], 0.0, False
    ctx = "\n\n".join("--- Фрагмент %d. Документ: «%s»\n%s" % (i + 1, c["doc"], c["text"]) for i, (_, c) in enumerate(top))
    body = {"model": model, "temperature": 0.2, "max_tokens": 700,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Фрагменты базы знаний:\n" + ctx + "\n\nВопрос администратора: " + q}]}
    if nothink and not model.startswith("gemini"):  # у Gemini рассуждение отключить нельзя (API вернёт 400)
        body["reasoning"] = {"effort": "none"}
    t1 = time.time()
    j = ait("/chat/completions", body)
    t_llm = time.time() - t1
    msg = (j.get("choices") or [{}])[0].get("message") or {}
    return {
        "answer": msg.get("content") or "",
        "usage": j.get("usage"),
        "model": j.get("model") or model,
        "timing": {"embed": round(t_emb, 2), "search": round(t_search, 3), "rerank": round(t_rr, 2),
                   "llm": round(t_llm, 2), "total": round(time.time() - T0, 2)},
        "rerank_ok": rr_ok,
        "sources": ["%s (%.2f)" % (c["doc"], s) for s, c in top],
    }


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
<p class="sub">Сравнение: агент Timeweb (их база знаний) и свой RAG на AITunnel. Под ответом — время по этапам и токены. Не вводите данные пациентов. <a href="admin">Документы и индекс</a> · <a href="logout">Выйти</a></p>
<div class="chips" id="chips">
<button class="chip">пациент с отеком щеки и температурой что делать</button><button class="chip">бабушка привела ребенка 10 лет на первый прием</button>
<button class="chip">какая гарантия на коронку</button><button class="chip">сколько стоит профосмотр</button><button class="chip">сколько стоит имплантация под ключ</button></div>
<div id="log"></div><div class="card" id="stats" style="display:none"></div></main>
<div class="bar"><div class="in"><div class="row"><textarea id="q" placeholder="Вопрос… (Enter — отправить)"></textarea><button id="send">Спросить</button></div>
<div class="opts"><label>Движок <select id="engine"><option value="aitunnel">AITunnel + свой RAG</option><option value="timeweb">Timeweb агент</option></select></label>
<label>Модель <select id="model">__MODELS__</select></label>
<label><input type="checkbox" id="nothink" checked> без рассуждения</label><label><input type="checkbox" id="rerank"> реранк (+0,45 ₽)</label><span id="status"></span></div></div></div>
<script>
(function(){
var log=document.getElementById('log'),q=document.getElementById('q'),send=document.getElementById('send'),st=document.getElementById('status'),stats=document.getElementById('stats'),T={};
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]})}
function md(s){var L=esc(s).split('\\n'),h='',ul=false;L.forEach(function(l){var m=l.match(/^\\s*(?:[-*•]|\\d+[.)])\\s+(.*)/);l=(m?m[1]:l).replace(/\\*\\*(.+?)\\*\\*/g,'<b>$1</b>');if(m){if(!ul){h+='<ul>';ul=true}h+='<li>'+l+'</li>'}else{if(ul){h+='</ul>';ul=false}if(l.trim())h+='<p>'+l+'</p>'}});return h+(ul?'</ul>':'')}
function med(a){var b=a.slice().sort(function(x,y){return x-y}),m=Math.floor(b.length/2);return b.length%2?b[m]:(b[m-1]+b[m])/2}
function upd(){var p=[];for(var k in T){p.push(k+': '+T[k].length+' вопр., медиана '+med(T[k]).toFixed(1)+' с')}stats.style.display=p.length?'':'none';stats.textContent='Итого — '+p.join(' · ')}
window.kcAsk=function(text,opts){opts=opts||{};text=String(text).trim();if(!text)return Promise.resolve();
var engine=opts.engine||document.getElementById('engine').value,model=opts.model||document.getElementById('model').value,nothink=opts.nothink!=null?opts.nothink:document.getElementById('nothink').checked,rr=opts.rerank!=null?opts.rerank:document.getElementById('rerank').checked;
var label=engine==='timeweb'?'Timeweb агент':'AITunnel · '+model+(nothink?'':' · с рассуждением')+(rr?' · реранк':'');
var qd=document.createElement('div');qd.className='card q';qd.textContent=text;log.appendChild(qd);var ad=document.createElement('div');ad.className='card';ad.innerHTML='<span class="meta">Жду ответ…</span>';log.appendChild(ad);ad.scrollIntoView({block:'end'});
send.disabled=true;q.value='';var t0=performance.now(),tm=setInterval(function(){st.textContent=((performance.now()-t0)/1000).toFixed(1)+' с…'},100);
return fetch('api',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q:text,engine:engine,model:model,nothink:nothink,rerank:rr})}).then(function(r){return r.json()}).then(function(j){
var sec=(performance.now()-t0)/1000;if(j.error){ad.innerHTML='<div class="err">'+esc(j.error)+'</div>';return {error:j.error}}
(T[label]=T[label]||[]).push(sec);upd();var u=j.usage||{},t=j.timing||{};
var parts=[];if(t.embed!=null)parts.push('эмбеддинг '+t.embed+' с');if(t.search!=null)parts.push('поиск '+t.search+' с');if(t.rerank)parts.push('реранк '+t.rerank+' с'+(j.rerank_ok?'':' (не сработал)'));parts.push('модель '+t.llm+' с');
ad.innerHTML='<div class="ans">'+md(j.answer||'(пусто)')+'</div>'+(j.sources&&j.sources.length?'<div class="src">Фрагменты: '+esc(j.sources.join('; '))+'</div>':'')+
'<div class="meta"><span class="'+(sec<=5?'fast':'slow')+'">⏱ '+sec.toFixed(1)+' с всего</span><span>'+parts.join(' · ')+'</span>'+(u.prompt_tokens?'<span>токены: '+u.prompt_tokens+' / '+u.completion_tokens+'</span>':'')+'<span>'+esc(label)+'</span></div>';
return {sec:sec,j:j,label:label}}).catch(function(e){ad.innerHTML='<div class="err">'+esc(e)+'</div>'}).then(function(x){clearInterval(tm);st.textContent='';send.disabled=false;return x})};
send.onclick=function(){kcAsk(q.value)};q.onkeydown=function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();if(!send.disabled)kcAsk(q.value)}};
document.getElementById('chips').onclick=function(e){var b=e.target.closest('.chip');if(b&&!send.disabled)kcAsk(b.textContent)};
document.getElementById('engine').onchange=function(){document.getElementById('model').disabled=this.value==='timeweb'};
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

    def js(self, code, obj):
        self.send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def route(self):
        p = urllib.parse.urlparse(self.path).path
        return p.rstrip("/")

    def do_GET(self):
        p = self.route()
        if p.endswith("/logout"):
            SESSIONS.pop(self.sid(), None)
            save_sessions()
            return self.send(302, b"", headers={"Location": "./", "Set-Cookie": "kc_sid=; Max-Age=0; Path=/"})
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
        opts = "".join("<option>%s</option>" % m for m in MODELS)
        return self.send(200, page(MAIN.replace("__MODELS__", opts)))

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
                model = data.get("model") if data.get("model") in MODELS else MODELS[0]
                return self.js(200, answer_ait(q, model, bool(data.get("nothink")), data.get("rerank") is True, int(data.get("top_k") or 6)))
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
  echo "Готово: http://$(hostname -I | awk '{print $1}'):${PORT}/"
  grep -q '^AITUNNEL_KEY=' "$ENVF" && echo "Ключ AITunnel: задан" || echo "Ключ AITunnel: НЕ задан"
else
  echo "Сервис не запустился:"; journalctl -u kc-test -n 20 --no-pager
fi
