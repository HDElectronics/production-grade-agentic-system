/* ============================================================
   Production-Grade Agentic System — Interactive Handbook
   Shared runtime: nav, theme, code highlighting, copy buttons
   No dependencies. Works from file:// as well as http://.
   ============================================================ */

const PAGES = [
  { file: "index.html",               num: "00", title: "Start here",              group: "Overview" },
  { file: "01-modular-codebase.html", num: "01", title: "Modular codebase",        group: "Foundations" },
  { file: "02-data-persistence.html", num: "02", title: "Data persistence",        group: "Foundations" },
  { file: "03-security.html",         num: "03", title: "Security & safeguards",   group: "Foundations" },
  { file: "04-service-layer.html",    num: "04", title: "Service layer",           group: "Core" },
  { file: "05-agent-architecture.html", num: "05", title: "Agentic architecture",  group: "Core" },
  { file: "06-api-gateway.html",      num: "06", title: "API gateway",             group: "Core" },
  { file: "07-observability.html",    num: "07", title: "Observability",           group: "Operations" },
  { file: "08-devops.html",           num: "08", title: "DevOps automation",       group: "Operations" },
  { file: "09-evaluation.html",       num: "09", title: "Evaluation framework",    group: "Operations" },
  { file: "10-stress-testing.html",   num: "10", title: "Stress testing",          group: "Operations" },
];

/* ---------------- Theme ---------------- */
// localStorage can throw on opaque origins (some file:// contexts, sandboxed
// iframes). The theme is a nicety; never let it break the page.
function readTheme()  { try { return localStorage.getItem("pgas-theme"); } catch { return null; } }
function writeTheme(v) { try { localStorage.setItem("pgas-theme", v); } catch { /* ignore */ } }

function initTheme() {
  const prefers = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", readTheme() || prefers);
}
function toggleTheme() {
  const now = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", now);
  writeTheme(now);
  document.querySelectorAll("[data-theme-btn]").forEach(b => b.textContent = now === "dark" ? "☾ Dark" : "☀ Light");
}
initTheme();

/* ---------------- Navigation ---------------- */
function currentFile() {
  const p = location.pathname.split("/").pop();
  return p && p.length ? p : "index.html";
}

function buildChrome() {
  if (document.getElementById("sidebar")) return;   // idempotent
  const here = currentFile();

  // Sidebar
  const side = document.createElement("aside");
  side.className = "sidebar";
  side.id = "sidebar";
  let html = `<a class="brand" href="index.html">
      <div class="brand-title">Production-Grade<br>Agentic AI System</div>
      <div class="brand-sub">Interactive handbook</div>
    </a><nav class="nav">`;
  let group = null;
  for (const pg of PAGES) {
    if (pg.group !== group) { group = pg.group; html += `<div class="nav-group-label">${group}</div>`; }
    html += `<a href="${pg.file}"${pg.file === here ? ' class="active"' : ""}><span class="num">${pg.num}</span><span>${pg.title}</span></a>`;
  }
  html += `</nav>
    <div class="nav-group-label">Reference</div>
    <nav class="nav">
      <a href="../README.md"><span class="num">↗</span><span>Full write-up (README)</span></a>
      <a href="https://github.com/FareedKhan-dev/production-grade-agentic-system"><span class="num">↗</span><span>Source repository</span></a>
    </nav>
    <div style="margin-top:26px"><button class="icon-btn" data-theme-btn onclick="toggleTheme()">☾ Dark</button></div>`;
  side.innerHTML = html;

  // Topbar for narrow screens
  const top = document.createElement("div");
  top.className = "topbar";
  top.innerHTML = `<button class="icon-btn" onclick="document.getElementById('sidebar').classList.toggle('open')">☰ Menu</button>
    <span style="font-size:13px;color:var(--text-dim)">Agentic System Handbook</span>
    <span style="margin-left:auto"><button class="icon-btn" data-theme-btn onclick="toggleTheme()">☾ Dark</button></span>`;

  const content = document.querySelector(".content");
  const layout = document.createElement("div");
  layout.className = "layout";
  document.body.insertBefore(top, document.body.firstChild);
  document.body.insertBefore(layout, content);
  layout.appendChild(side);
  layout.appendChild(content);

  document.querySelectorAll("[data-theme-btn]").forEach(
    b => b.textContent = document.documentElement.getAttribute("data-theme") === "dark" ? "☾ Dark" : "☀ Light");

  // Prev / next pager
  const idx = PAGES.findIndex(p => p.file === here);
  if (idx >= 0) {
    const prev = PAGES[idx - 1], next = PAGES[idx + 1];
    const pager = document.createElement("div");
    pager.className = "pager";
    pager.innerHTML =
      (prev ? `<a href="${prev.file}"><span class="dir">← Previous</span>${prev.num} · ${prev.title}</a>` : "<span></span>") +
      (next ? `<a href="${next.file}" style="text-align:right"><span class="dir">Next →</span>${next.num} · ${next.title}</a>` : "<span></span>");
    content.appendChild(pager);
  }
}

/* ---------------- Syntax highlighting ---------------- */
const KEYWORDS = {
  python: ["False","None","True","and","as","assert","async","await","break","class","continue","def","del","elif","else","except","finally","for","from","global","if","import","in","is","lambda","match","case","nonlocal","not","or","pass","raise","return","try","while","with","yield","self","cls"],
  js: ["const","let","var","function","return","if","else","for","while","await","async","new","class","try","catch","finally","throw","import","export","from","of","in","typeof","null","undefined","true","false"],
  bash: ["if","then","fi","else","elif","for","in","do","done","while","case","esac","function","echo","exit","export","source","set","local","return","cd","curl","docker","make","sudo","apt-get","pip","uv","python"],
  yaml: ["true","false","null","on","off"],
  sql: ["CREATE","TABLE","SELECT","FROM","WHERE","INSERT","INTO","VALUES","UPDATE","DELETE","PRIMARY","KEY","FOREIGN","REFERENCES","NOT","NULL","INDEX","EXTENSION","IF","EXISTS","TEXT","INTEGER","SERIAL","TIMESTAMP","DEFAULT","UNIQUE","ON","VARCHAR"],
};

function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

function highlight(src, lang) {
  const kw = KEYWORDS[lang] || KEYWORDS.python;
  const kwRe = kw.map(k => k.replace(/[-]/g, "\\-")).join("|");
  const commentRe = lang === "sql" ? "--[^\\n]*" : "#[^\\n]*";
  const parts = [
    "(" + (lang === "js" ? "\\/\\/[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/" : commentRe) + ")",           // 1 comment
    "(\"\"\"[\\s\\S]*?\"\"\"|'''[\\s\\S]*?'''|\"(?:[^\"\\\\\\n]|\\\\.)*\"|'(?:[^'\\\\\\n]|\\\\.)*'|`(?:[^`\\\\]|\\\\.)*`)", // 2 string
    "(@[A-Za-z_][\\w.]*)",                                                                       // 3 decorator
    "\\b(def|class|function)\\s+([A-Za-z_]\\w*)",                                                // 4,5 def name
    "\\b(" + kwRe + ")\\b",                                                                      // 6 keyword
    "\\b(\\d+\\.?\\d*)\\b",                                                                      // 7 number
    "([A-Za-z_]\\w*)(?=\\()",                                                                    // 8 call
  ];
  const re = new RegExp(parts.join("|"), "g");
  let out = "", last = 0, m;
  const text = esc(src);
  while ((m = re.exec(text)) !== null) {
    out += text.slice(last, m.index);
    if (m[1])       out += `<span class="tok-com">${m[1]}</span>`;
    else if (m[2])  out += `<span class="tok-str">${m[2]}</span>`;
    else if (m[3])  out += `<span class="tok-dec">${m[3]}</span>`;
    else if (m[4])  out += `<span class="tok-kw">${m[4]}</span> <span class="tok-cls">${m[5]}</span>`;
    else if (m[6])  out += `<span class="tok-kw">${m[6]}</span>`;
    else if (m[7])  out += `<span class="tok-num">${m[7]}</span>`;
    else if (m[8])  out += `<span class="tok-fn">${m[8]}</span>`;
    last = re.lastIndex;
  }
  out += text.slice(last);
  return out;
}

function initCode() {
  document.querySelectorAll(".code").forEach(block => {
    const pre = block.querySelector("pre");
    if (!pre || block.querySelector(".code-head")) return;   // idempotent
    const lang = block.dataset.lang || "python";
    const file = block.dataset.file || "";
    // de-indent
    let raw = pre.textContent.replace(/^\n/, "").replace(/\s+$/, "");
    const indents = raw.split("\n").filter(l => l.trim()).map(l => l.match(/^ */)[0].length);
    const min = indents.length ? Math.min(...indents) : 0;
    if (min > 0) raw = raw.split("\n").map(l => l.slice(min)).join("\n");

    const head = document.createElement("div");
    head.className = "code-head";
    head.innerHTML = `<span class="code-file">${file || `<span class="code-lang">${lang}</span>`}</span>
      <span style="display:flex;gap:8px;align-items:center">
        ${file ? `<span class="code-lang">${lang}</span>` : ""}
        <button class="copy-btn">copy</button>
      </span>`;
    block.insertBefore(head, pre);
    pre.innerHTML = highlight(raw, lang);
    head.querySelector(".copy-btn").addEventListener("click", e => {
      navigator.clipboard.writeText(raw).then(() => {
        e.target.textContent = "copied"; e.target.classList.add("done");
        setTimeout(() => { e.target.textContent = "copy"; e.target.classList.remove("done"); }, 1400);
      });
    });
  });
}

/* ---------------- On-this-page TOC ---------------- */
function initToc() {
  const host = document.querySelector("[data-toc]");
  if (!host) return;
  const hs = [...document.querySelectorAll(".content h2")];
  if (!hs.length) { host.remove(); return; }
  host.innerHTML = `<div class="toc-title">On this page</div><ol>` +
    hs.map((h, i) => {
      if (!h.id) h.id = "s" + i;
      return `<li><a href="#${h.id}">${h.textContent}</a></li>`;
    }).join("") + "</ol>";
}

/* ---------------- Small shared helpers for widgets ----------------
   Attached to `window` explicitly so per-page inline scripts can use them
   regardless of how this file was loaded. */
window.$ = (sel, root = document) => root.querySelector(sel);
window.$$ = (sel, root = document) => [...root.querySelectorAll(sel)];
window.fmt = (n, d = 2) => Number(n).toFixed(d);
window.clamp = (v, a, b) => Math.min(b, Math.max(a, v));
function bindRange(id, cb) {
  const el = document.getElementById(id);
  if (!el) return;
  const out = document.getElementById(id + "-val");
  const fire = () => { if (out) out.textContent = el.dataset.suffix ? el.value + el.dataset.suffix : el.value; cb(Number(el.value)); };
  el.addEventListener("input", fire);
  fire();
}
function bar(label, value, max, color) {
  const pct = max > 0 ? clamp((value / max) * 100, 0, 100) : 0;
  return `<div class="bar-row"><span class="bar-label">${label}</span>
    <span class="bar-track"><span class="bar-fill" style="width:${pct}%${color ? `;background:${color}` : ""}"></span></span>
    <span class="bar-val">${value}</span></div>`;
}

document.addEventListener("DOMContentLoaded", () => {
  buildChrome();
  initCode();
  initToc();
  if (window.pageInit) window.pageInit();
});
