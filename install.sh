#!/usr/bin/env bash
# Установка тестовой страницы помощника КЦ на сервер (порт 8787).
# Запуск:  curl -fsSL https://raw.githubusercontent.com/DelureDev/kc-bot/main/install.sh | sudo bash
# Удаление: systemctl disable --now kc-test && rm -rf /opt/kc-test /etc/systemd/system/kc-test.service
set -euo pipefail
PORT=8787
[ "$(id -u)" -eq 0 ] || { echo "Запустите через sudo"; exit 1; }
command -v python3 >/dev/null || { apt-get update -qq && apt-get install -y -qq python3; }
mkdir -p /opt/kc-test
cat > /opt/kc-test/app.py <<'PYEOF'
#!/usr/bin/env python3
"""Тестовая страница помощника КЦ: вход по паролю + прослойка к API агента Timeweb.
Только стандартная библиотека Python 3. Настройки — из переменных окружения:
TW_TOKEN, TW_AGENT_ID, PAGE_PASSWORD, PORT (по умолчанию 8787)."""
import hmac, http.cookies, json, os, secrets, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = os.environ["TW_TOKEN"]
AGENT = os.environ.get("TW_AGENT_ID", "d94b4d73-2ce5-4006-a6f5-4039b60c2d9e")
PASSWORD = os.environ["PAGE_PASSWORD"]
PORT = int(os.environ.get("PORT", "8787"))
URL = "https://agent.timeweb.cloud/api/v1/cloud-ai/agents/%s/v1/chat/completions" % AGENT
SESSIONS = {}  # sid -> число вопросов

STYLE = '\n  :root { --bg:#f6f7f9; --card:#fff; --text:#1c2230; --muted:#5d6678; --accent:#5b5ef0; --border:#e3e6ec; --q:#eef0ff; --bad:#b3261e; }\n  @media (prefers-color-scheme: dark) { :root { --bg:#141821; --card:#1c2230; --text:#e8ebf2; --muted:#a2aabb; --accent:#8f92ff; --border:#2c3444; --q:#262b45; --bad:#ff8a80; } }\n  * { box-sizing:border-box; }\n  body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }\n  main { max-width:820px; margin:0 auto; padding:24px 16px 160px; }\n  h1 { font-size:22px; margin:0 0 4px; }\n  .sub { color:var(--muted); margin:0 0 16px; font-size:14px; }\n  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px 16px; margin-bottom:12px; }\n  .q { background:var(--q); border-color:transparent; font-weight:600; }\n  .meta { color:var(--muted); font-size:12px; margin-top:8px; display:flex; gap:12px; flex-wrap:wrap; }\n  .fast { color:#1e8e3e; } .slow { color:#c26a00; }\n  .err { color:var(--bad); }\n  .bar { position:fixed; left:0; right:0; bottom:0; background:var(--card); border-top:1px solid var(--border); padding:12px 16px; }\n  .bar .in { max-width:820px; margin:0 auto; }\n  .row { display:flex; gap:8px; }\n  textarea { flex:1; resize:none; height:52px; padding:10px 12px; border-radius:10px; border:1px solid var(--border); background:var(--bg); color:var(--text); font:inherit; }\n  button { border:0; border-radius:10px; padding:0 18px; background:var(--accent); color:#fff; font:inherit; font-weight:600; cursor:pointer; }\n  button:disabled { opacity:.5; cursor:default; }\n  .opts { display:flex; gap:16px; align-items:center; margin-top:8px; font-size:13px; color:var(--muted); flex-wrap:wrap; }\n  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:16px; }\n  .chip { background:transparent; color:var(--text); border:1px solid var(--border); border-radius:999px; padding:5px 11px; font-size:13px; font-weight:400; }\n  .chip:hover { border-color:var(--accent); color:var(--accent); }\n  .login { max-width:340px; margin:80px auto; }\n  .login input { width:100%; padding:10px 12px; border-radius:10px; border:1px solid var(--border); background:var(--bg); color:var(--text); font:inherit; margin:10px 0; }\n  .login button { width:100%; height:42px; }\n  .ans p { margin:0 0 6px; } .ans ul { margin:0 0 6px; padding-left:20px; }\n  a { color:var(--accent); }\n'
MAIN = '\n<main>\n  <h1>Помощник КЦ — тест API</h1>\n  <p class="sub">Запросы идут напрямую в API агента Timeweb (без виджета). Под каждым ответом — время. Не вводите данные пациентов. <a href="logout">Выйти</a></p>\n  <div class="chips" id="chips">\n    <button class="chip">пациент с отеком щеки и температурой что делать</button>\n    <button class="chip">бабушка привела ребенка 10 лет на первый прием</button>\n    <button class="chip">какая гарантия на коронку</button>\n    <button class="chip">сколько стоит профосмотр</button>\n    <button class="chip">не дозвонились накануне подтвердить визит</button>\n    <button class="chip">сколько стоит имплантация под ключ</button>\n  </div>\n  <div id="log"></div>\n  <div class="card" id="stats" style="display:none"></div>\n</main>\n<div class="bar"><div class="in">\n  <div class="row">\n    <textarea id="q" placeholder="Вопрос… (Enter — отправить, Shift+Enter — новая строка)"></textarea>\n    <button id="send">Спросить</button>\n  </div>\n  <div class="opts">\n    <label><input type="checkbox" id="nothink"> отключить рассуждение (thinking: disabled)</label>\n    <span id="status"></span>\n  </div>\n</div></div>\n<script>\n(function () {\n  var log = document.getElementById(\'log\'), q = document.getElementById(\'q\'), send = document.getElementById(\'send\');\n  var statusEl = document.getElementById(\'status\'), stats = document.getElementById(\'stats\'), times = { on: [], off: [] };\n\n  function esc(s) { return s.replace(/[&<>"]/g, function (c) { return ({\'&\':\'&amp;\',\'<\':\'&lt;\',\'>\':\'&gt;\',\'"\':\'&quot;\'})[c]; }); }\n  function md(s) {\n    var lines = esc(s).split(\'\\n\'), html = \'\', inList = false;\n    lines.forEach(function (l) {\n      var m = l.match(/^\\s*(?:[-*•]|\\d+[.)])\\s+(.*)/);\n      l = (m ? m[1] : l).replace(/\\*\\*(.+?)\\*\\*/g, \'<b>$1</b>\');\n      if (m) { if (!inList) { html += \'<ul>\'; inList = true; } html += \'<li>\' + l + \'</li>\'; }\n      else { if (inList) { html += \'</ul>\'; inList = false; } if (l.trim()) html += \'<p>\' + l + \'</p>\'; }\n    });\n    return html + (inList ? \'</ul>\' : \'\');\n  }\n  function median(a) { if (!a.length) return null; var b = a.slice().sort(function (x, y) { return x - y; }), m = Math.floor(b.length / 2); return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; }\n  function renderStats() {\n    var parts = [];\n    [[\'on\', \'с рассуждением\'], [\'off\', \'без рассуждения\']].forEach(function (p) {\n      var a = times[p[0]]; if (a.length) parts.push(p[1] + \': \' + a.length + \' вопр., медиана \' + median(a).toFixed(1) + \' с, макс \' + Math.max.apply(null, a).toFixed(1) + \' с\');\n    });\n    stats.style.display = parts.length ? \'\' : \'none\'; stats.textContent = \'Итого — \' + parts.join(\' · \');\n  }\n\n  function ask(text) {\n    text = text.trim(); if (!text) return;\n    var nothink = document.getElementById(\'nothink\').checked;\n    var qd = document.createElement(\'div\'); qd.className = \'card q\'; qd.textContent = text; log.appendChild(qd);\n    var ad = document.createElement(\'div\'); ad.className = \'card\'; ad.innerHTML = \'<span class="meta">Жду ответ…</span>\'; log.appendChild(ad);\n    ad.scrollIntoView({ behavior: \'smooth\', block: \'end\' });\n    send.disabled = true; q.value = \'\';\n    var t0 = performance.now(), timer = setInterval(function () { statusEl.textContent = ((performance.now() - t0) / 1000).toFixed(1) + \' с…\'; }, 100);\n    fetch(\'api\', { method: \'POST\', headers: { \'Content-Type\': \'application/json\' }, body: JSON.stringify({ q: text, nothink: nothink }) })\n      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })\n      .then(function (res) {\n        var sec = (performance.now() - t0) / 1000, j = res.j, cls = sec <= 5 ? \'fast\' : \'slow\';\n        if (!res.ok || j.error) { ad.innerHTML = \'<div class="err">\' + esc(j.error || \'Ошибка\') + \'</div>\'; return; }\n        (nothink ? times.off : times.on).push(sec); renderStats();\n        var u = j.usage || {};\n        ad.innerHTML = \'<div class="ans">\' + md(j.answer || \'(пустой ответ)\') + \'</div>\' +\n          \'<div class="meta"><span class="\' + cls + \'">⏱ \' + sec.toFixed(1) + \' с всего</span>\' +\n          \'<span>Timeweb: \' + (j.upstream_ms / 1000).toFixed(1) + \' с</span>\' +\n          (u.total_tokens ? \'<span>токены: \' + u.prompt_tokens + \' / \' + u.completion_tokens + \'</span>\' : \'\') +\n          \'<span>\' + (nothink ? \'без рассуждения\' : \'с рассуждением\') + \'</span></div>\';\n      })\n      .catch(function (e) { ad.innerHTML = \'<div class="err">Сеть: \' + esc(String(e)) + \'</div>\'; })\n      .then(function () { clearInterval(timer); statusEl.textContent = \'\'; send.disabled = false; q.focus(); ad.scrollIntoView({ behavior: \'smooth\', block: \'end\' }); });\n  }\n\n  send.addEventListener(\'click\', function () { ask(q.value); });\n  q.addEventListener(\'keydown\', function (e) { if (e.key === \'Enter\' && !e.shiftKey) { e.preventDefault(); if (!send.disabled) ask(q.value); } });\n  document.getElementById(\'chips\').addEventListener(\'click\', function (e) { var b = e.target.closest(\'.chip\'); if (b && !send.disabled) ask(b.textContent); });\n  q.focus();\n})();\n</script>\n'

def page(body):
    return ("<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<meta name='robots' content='noindex, nofollow'><title>Помощник КЦ — тест API</title>"
            "<style>" + STYLE + "</style></head><body>" + body + "</body></html>").encode("utf-8")

def login_page(err=""):
    e = "<div class='err'>%s</div>" % err if err else ""
    return page("<form class='card login' method='post' action='login'><h1>Помощник КЦ</h1>"
                "<p class='sub'>Тестовая страница. Введите пароль.</p>"
                "<input type='password' name='password' autofocus placeholder='Пароль'>" + e +
                "<button type='submit'>Войти</button></form>")

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

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
        if path == "logout":
            s = self.sid()
            SESSIONS.pop(s, None)
            return self.send(302, b"", headers={"Location": "./", "Set-Cookie": "kc_sid=; Max-Age=0; Path=/"})
        if not self.sid():
            return self.send(200, login_page())
        return self.send(200, page(MAIN))

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(min(n, 20000))
        path = self.path.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
        if path == "login":
            pw = urllib.parse.parse_qs(raw.decode("utf-8", "ignore")).get("password", [""])[0]
            if hmac.compare_digest(pw.encode(), PASSWORD.encode()):
                s = secrets.token_urlsafe(24)
                SESSIONS[s] = 0
                return self.send(302, b"", headers={"Location": "./", "Set-Cookie": "kc_sid=%s; HttpOnly; SameSite=Strict; Path=/; Max-Age=604800" % s})
            time.sleep(1)
            return self.send(200, login_page("Неверный пароль"))
        if path == "api":
            s = self.sid()
            if not s:
                return self.js(401, {"error": "Нужно войти заново"})
            SESSIONS[s] += 1
            if SESSIONS[s] > 200:
                return self.js(429, {"error": "Лимит вопросов в этой сессии исчерпан"})
            try:
                data = json.loads(raw or b"{}")
            except ValueError:
                return self.js(400, {"error": "Плохой запрос"})
            q = str(data.get("q", "")).strip()
            if not q or len(q) > 1000:
                return self.js(400, {"error": "Пустой или слишком длинный вопрос"})
            body = {"messages": [{"role": "user", "content": q}], "stream": False}
            if data.get("nothink"):
                body["thinking"] = {"type": "disabled"}
            req = urllib.request.Request(URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
                                         headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    j = json.loads(r.read())
            except urllib.error.HTTPError as e:
                return self.js(502, {"error": "Timeweb ответил %s: %s" % (e.code, e.read()[:300].decode("utf-8", "ignore")),
                                     "upstream_ms": int((time.time() - t0) * 1000)})
            except Exception as e:
                return self.js(502, {"error": "Нет связи с Timeweb: %r" % e, "upstream_ms": int((time.time() - t0) * 1000)})
            ms = int((time.time() - t0) * 1000)
            ans = (((j.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
            return self.js(200, {"answer": ans, "upstream_ms": ms, "usage": j.get("usage"), "model": j.get("model")})
        self.send(404, b"not found")

    def log_message(self, fmt, *args):
        pass

import urllib.parse  # noqa: E402

if __name__ == "__main__":
    print("kc-test on port", PORT)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
PYEOF
if [ ! -f /opt/kc-test/env ]; then
  echo
  read -r -s -p "Вставьте токен агента Timeweb (Управление → Доступ по API): " TOK </dev/tty; echo
  read -r -s -p "Придумайте пароль для страницы: " PW </dev/tty; echo
  umask 077
  printf 'TW_TOKEN=%s\nPAGE_PASSWORD=%s\nTW_AGENT_ID=d94b4d73-2ce5-4006-a6f5-4039b60c2d9e\nPORT=%s\n' "$TOK" "$PW" "$PORT" > /opt/kc-test/env
fi
cat > /etc/systemd/system/kc-test.service <<UNIT
[Unit]
Description=KC test page (Timeweb agent proxy)
After=network-online.target

[Service]
EnvironmentFile=/opt/kc-test/env
ExecStart=/usr/bin/python3 /opt/kc-test/app.py
DynamicUser=yes
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now kc-test >/dev/null
systemctl restart kc-test
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then ufw allow ${PORT}/tcp >/dev/null; fi
sleep 1
systemctl is-active --quiet kc-test && echo "Готово: http://$(hostname -I | awk '{print $1}'):${PORT}/  (или http://anarchychess.ru:${PORT}/)" || { echo "Сервис не запустился:"; journalctl -u kc-test -n 20 --no-pager; }
