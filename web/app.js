/* Regime Radar front-end — vanilla JS, hand-rolled SVG charts, no build step. */
"use strict";

const $ = (s) => document.querySelector(s);
const state = { snap: null, themeFilter: null, sources: {}, history: [] };
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtPct = (v) => (v == null ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(1)}%`);
const sv = (n) => `var(--series-${n})`;

/* ---------------- theme toggle ---------------- */
(() => {
  let saved = null;
  try { saved = localStorage.getItem("rr-theme"); } catch (e) {}
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").onclick = () => {
    const cur = document.documentElement.dataset.theme ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("rr-theme", next); } catch (e) {}
  };
})();

/* ---------------- tooltip + popover ---------------- */
const tip = $("#tip");
function showTip(e, html) { tip.innerHTML = html; tip.style.display = "block"; moveTip(e); }
function moveTip(e) {
  const x = Math.min(e.clientX + 12, innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + "px"; tip.style.top = (e.clientY + 14) + "px";
}
function hideTip() { tip.style.display = "none"; }
document.addEventListener("mousemove", (e) => { if (tip.style.display === "block") moveTip(e); });

const pop = $("#pop");
function showSource(el, id) {
  const d = state.sources[id];
  if (!d) return;
  let body = "";
  if (d.kind === "tool") body = `<pre class="mono small" style="white-space:pre-wrap;margin:0">${esc(JSON.stringify(d.data, null, 1)).slice(0, 900)}</pre>`;
  else body = `<div>${esc(d.summary || d.text || "")}</div>`;
  const meta = [d.publisher, d.published && d.published.slice(0, 10), d.ticker && `${d.ticker} FY${d.fiscal_year} 10-K`]
    .filter(Boolean).map(esc).join(" · ");
  pop.innerHTML = `<div class="pt">${esc(d.title)}</div><div class="pm">${meta}</div>${body}` +
    (d.url ? `<div style="margin-top:6px"><a href="${esc(d.url)}" target="_blank" rel="noopener">Open source ↗</a></div>` : "");
  const r = el.getBoundingClientRect();
  pop.style.display = "block";
  pop.style.left = Math.max(8, Math.min(r.left, innerWidth - pop.offsetWidth - 8)) + "px";
  const top = r.bottom + 6;
  pop.style.top = (top + pop.offsetHeight > innerHeight ? r.top - pop.offsetHeight - 6 : top) + "px";
}
document.addEventListener("click", (e) => {
  const c = e.target.closest("[data-src]");
  if (c) { e.stopPropagation(); showSource(c, c.dataset.src.split(",")[0]); return; }
  if (!e.target.closest("#pop")) pop.style.display = "none";
});

/* ---------------- citation-aware markdown ---------------- */
function renderCited(text, citations, sources) {
  Object.assign(state.sources, sources || {});
  const order = {};
  let n = 0;
  const cites = [...(citations || [])].filter((c) => c.start != null).sort((a, b) => a.start - b.start);
  let html = "", pos = 0;
  for (const c of cites) {
    if (c.start < pos) continue;
    html += esc(text.slice(pos, c.start));
    const nums = c.sources.map((s) => (order[s] ??= ++n));
    html += `<span class="cite" data-src="${esc(c.sources.join(","))}">${esc(text.slice(c.start, c.end))}<sup>${nums.join(",")}</sup></span>`;
    pos = c.end;
  }
  html += esc(text.slice(pos));
  return markdown(html);
}
function markdown(html) {
  const lines = html.split("\n");
  let out = "", inList = false;
  for (let line of lines) {
    line = line.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<i>$2</i>");
    const li = line.match(/^\s*[-•]\s+(.*)/);
    if (li) { if (!inList) { out += "<ul>"; inList = true; } out += `<li>${li[1]}</li>`; continue; }
    if (inList) { out += "</ul>"; inList = false; }
    const h = line.match(/^(#{1,4})\s+(.*)/);
    if (h) out += `<h3>${h[2]}</h3>`;
    else if (line.trim()) out += `<p>${line}</p>`;
  }
  if (inList) out += "</ul>";
  return out;
}

/* ---------------- charts (SVG) ---------------- */
function sparkline(values, color) {
  if (!values || values.length < 2) return "";
  const w = 140, h = 34, min = Math.min(...values), max = Math.max(...values), rng = max - min || 1;
  const pts = values.map((v, i) => [(i / (values.length - 1)) * w, h - 3 - ((v - min) / rng) * (h - 6)]);
  const d = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" data-spark='${JSON.stringify(values)}'>
    <path d="${d}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>
    <line class="xh" x1="0" x2="0" y1="0" y2="${h}" stroke="var(--text-muted)" stroke-width="1" vector-effect="non-scaling-stroke" opacity="0"/>
  </svg>`;
}
function columns(values, labels, color, h = 40) {
  const w = 210, n = values.length, max = Math.max(1, ...values), bw = w / n;
  let s = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">`;
  s += `<line x1="0" x2="${w}" y1="${h - 0.5}" y2="${h - 0.5}" stroke="var(--grid)"/>`;
  values.forEach((v, i) => {
    const bh = v ? Math.max(2, (v / max) * (h - 4)) : 0;
    s += `<rect x="${i * bw}" y="0" width="${bw}" height="${h}" fill="transparent" data-tip="${esc(labels[i])}: ${v} materiality pts"/>`;
    if (bh) s += `<rect x="${i * bw + 1}" y="${h - bh}" width="${Math.max(1, bw - 2)}" height="${bh}" rx="1.5" fill="${color}" pointer-events="none"/>`;
  });
  return s + "</svg>";
}
document.addEventListener("mouseover", (e) => {
  const t = e.target.closest("[data-tip]");
  if (t) showTip(e, t.dataset.tip);
});
document.addEventListener("mouseout", (e) => { if (e.target.closest("[data-tip]")) hideTip(); });

/* ---------------- renderers ---------------- */
function renderHeader(s) {
  const m = s.models;
  $("#model-chips").innerHTML =
    `<span class="chip mode ${s.mode}">${s.mode === "demo" ? "OFFLINE DEMO" : "LIVE"}</span>` +
    `<span class="chip">news index <b>${esc(m.embed_fast)}</b></span>` +
    `<span class="chip">filings index <b>${esc(m.embed_pro)}</b></span>` +
    `<span class="chip">reasoning <b>${esc(m.chat)}</b></span>`;
  $("#refreshed").textContent = s.last_refresh ? `Updated ${s.last_refresh.replace("T", " ")}` : "Indexing…";
}

function renderTiles(s) {
  const demo = s.mode === "demo";
  $("#price-note").innerHTML = demo
    ? `<div class="banner">Offline demo: price series are <b>synthetic random walks indexed to 100</b>, not market data. Add a Cohere key and run without <code>--demo</code> to load real prices.</div>` : "";
  $("#tiles").innerHTML = s.tiles.map((t) => {
    if (t.error) return `<div class="tile"><div class="tk">${esc(t.ticker)}</div><div class="muted small">${esc(t.error)}</div></div>`;
    const up = (t.chg_1m_pct ?? 0) >= 0;
    return `<div class="tile">
      <div class="tk">${esc(t.ticker)}</div>
      <div class="px num">${t.last?.toFixed(2) ?? "–"}</div>
      <div class="chg num"><span class="${(t.chg_1d_pct ?? 0) >= 0 ? "up" : "down"}">${fmtPct(t.chg_1d_pct)}</span>
        <span class="muted">1d</span> · <span class="${up ? "up" : "down"}">${fmtPct(t.chg_1m_pct)}</span> <span class="muted">1m</span></div>
      ${sparkline(t.spark, "var(--text-secondary)")}
    </div>`;
  }).join("");
  document.querySelectorAll("[data-spark]").forEach((svg) => {
    const vals = JSON.parse(svg.dataset.spark), xh = svg.querySelector(".xh");
    svg.addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect(), i = Math.round(((e.clientX - r.left) / r.width) * (vals.length - 1));
      const x = (i / (vals.length - 1)) * 140;
      xh.setAttribute("x1", x); xh.setAttribute("x2", x); xh.setAttribute("opacity", 1);
      showTip(e, `${vals.length - 1 - i} sessions ago · <b>${vals[i].toFixed(2)}</b>`);
    });
    svg.addEventListener("mouseleave", () => { xh.setAttribute("opacity", 0); hideTip(); });
  });
}

const MOMENTUM = { heating: ["▲", "heating", "var(--critical)"], cooling: ["▼", "cooling", "var(--pos)"], steady: ["■", "steady", "var(--text-muted)"] };
function renderThemes(s) {
  const max = Math.max(1, ...s.themes.map((t) => t.heat));
  const sorted = [...s.themes].sort((a, b) => b.heat - a.heat);
  $("#theme-bars").innerHTML = sorted.map((t) => {
    const [ic, lbl, col] = MOMENTUM[t.momentum];
    return `<div class="tb-row ${state.themeFilter === t.id ? "active" : ""}" data-theme="${t.id}"
        data-tip="${esc(t.name)} · heat ${t.heat} · ${t.count} headlines · avg sentiment ${t.sentiment >= 0 ? "+" : ""}${t.sentiment}">
      <div class="tb-name"><span class="dot" style="background:${sv(t.slot)}"></span><span>${esc(t.name)}</span></div>
      <div class="tb-track"><div class="tb-bar" style="width:${(t.heat / max) * 100}%;background:${sv(t.slot)}"></div></div>
      <div class="tb-val num">${t.heat.toFixed(1)} <span class="momentum" style="color:${col}">${ic} ${lbl}</span></div>
    </div>`;
  }).join("");
  $("#theme-grid").innerHTML = sorted.map((t) => `
    <div class="tcard ${state.themeFilter === t.id ? "active" : ""}" data-theme="${t.id}">
      <div class="tn"><span class="dot" style="background:${sv(t.slot)}"></span>${esc(t.name)}</div>
      ${columns(t.timeline, t.days, sv(t.slot))}
      <div class="meta"><span>${t.days[0].slice(5)} → ${t.days.at(-1).slice(5)}</span>
        <span>sentiment <b class="${t.sentiment >= 0 ? "up" : "down"} num">${t.sentiment >= 0 ? "+" : ""}${t.sentiment.toFixed(2)}</b></span></div>
    </div>`).join("");
  document.querySelectorAll("[data-theme]").forEach((el) => el.onclick = () => {
    state.themeFilter = state.themeFilter === el.dataset.theme ? null : el.dataset.theme;
    renderThemes(state.snap); renderFeed(state.snap.feed);
  });
}

function sentChip(v) {
  if (v == null) return "";
  const a = Math.min(1, Math.abs(v));
  const col = v > 0.05 ? "var(--pos)" : v < -0.05 ? "var(--neg)" : "var(--text-muted)";
  return `<span class="sent num" style="color:${col};background:color-mix(in srgb, ${col} ${8 + a * 18}%, transparent)">${v > 0 ? "+" : ""}${v.toFixed(2)}</span>`;
}
function pips(n) { return `<span class="pips" title="materiality ${n}/5">${[1, 2, 3, 4, 5].map((i) => `<i class="${i <= n ? "on" : ""}"></i>`).join("")}</span>`; }

function renderFeed(items, label) {
  const themes = Object.fromEntries((state.snap?.themes || []).map((t) => [t.id, t]));
  let list = items;
  if (state.themeFilter && !label) list = items.filter((d) => d.theme === state.themeFilter);
  const f = state.themeFilter && !label ? themes[state.themeFilter] : null;
  $("#filter-pill").innerHTML = label ? `${esc(label)} · <a href="#" id="clear-f">clear</a>`
    : f ? `Showing <b>${esc(f.name)}</b> · <a href="#" id="clear-f">clear filter</a>` : "";
  const cf = $("#clear-f");
  if (cf) cf.onclick = (e) => { e.preventDefault(); state.themeFilter = null; $("#search-input").value = ""; renderThemes(state.snap); renderFeed(state.snap.feed); };
  list.forEach((d) => (state.sources[d.id] = d));
  $("#feed").innerHTML = list.map((d) => {
    const th = themes[d.theme], ai = d.ai || {};
    const color = d.kind === "filing" ? "var(--text-muted)" : th ? sv(th.slot) : "var(--neutral)";
    return `<div class="item">
      <span class="dot" style="background:${color}" title="${esc(th?.name || (d.kind === "filing" ? "10-K filing" : "no dominant theme"))}"></span>
      <div><div class="t"><a href="${esc(d.url || "#")}" target="_blank" rel="noopener" data-src="${esc(d.id)}">${esc(d.title)}</a></div>
        <div class="m">${d.kind === "filing" ? `<span class="tag">10-K Item 1A</span>` : ""}
          ${d.publisher ? `<span>${esc(d.publisher)}</span>` : ""}${d.published ? `<span class="num">${d.published.slice(0, 10)}</span>` : ""}
          ${ai.event_type ? `<span class="tag">${esc(ai.event_type.replace(/_/g, " "))}</span>` : ""}
          ${(ai.tickers || d.tickers || []).slice(0, 4).map((t) => `<span class="tag mono">${esc(t)}</span>`).join("")}
          ${d.score != null ? `<span class="num">sim ${d.score.toFixed(3)}</span>` : ""}</div></div>
      <div style="text-align:right">${sentChip(ai.sentiment)}<div style="margin-top:4px">${ai.materiality ? pips(ai.materiality) : ""}</div></div>
    </div>`;
  }).join("") || `<div class="skeleton">Nothing here yet.</div>`;
}

async function loadBriefing(force) {
  $("#briefing").innerHTML = `<div class="skeleton">Retrieving evidence and reasoning… (Command A+ with a 2,048-token thinking budget)</div>`;
  try {
    const b = await api(`/api/briefing${force ? "?force=1" : ""}`);
    $("#briefing").innerHTML = renderCited(b.markdown, b.citations, b.sources) +
      `<p class="muted small">${b.n_documents} documents in context · ${b.citations.length} citations · generated ${esc(b.generated?.replace("T", " "))}</p>` +
      (b.thinking ? `<details><summary>Show model reasoning</summary><div class="thinking">${esc(b.thinking)}</div></details>` : "");
  } catch (e) { $("#briefing").innerHTML = `<div class="skeleton">Briefing failed: ${esc(e.message)}</div>`; }
}

const SUGGESTIONS = [
  "Which watchlist names are most exposed to 10-year yields above 5%?",
  "Is the oil spike showing up in corporate risk disclosures?",
  "Who is financing the AI build-out, and what could go wrong?",
  "Compare XOM and SPY over the last 3 months",
  "What new risks did NWND add this year?",
];
function renderSuggestions() {
  $("#suggest").innerHTML = SUGGESTIONS.map((q) => `<button type="button">${esc(q)}</button>`).join("");
  $("#suggest").querySelectorAll("button").forEach((b) => b.onclick = () => { $("#ask-input").value = b.textContent; ask(); });
}
async function ask() {
  const q = $("#ask-input").value.trim();
  if (!q) return;
  $("#ask-input").value = "";
  const box = document.createElement("div");
  box.className = "qa";
  box.innerHTML = `<div class="q">${esc(q)}</div><div class="skeleton">The agent is planning and calling tools…</div>`;
  $("#answers").prepend(box);
  try {
    const a = await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, history: state.history.slice(-6) }) });
    state.history.push({ role: "user", content: q }, { role: "assistant", content: a.answer });
    const trace = a.trace.map((t) => `<li><code>${esc(t.tool)}(${esc(Object.entries(t.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", "))})</code>
      ${t.plan ? `<div class="muted">${esc(t.plan)}</div>` : ""}</li>`).join("");
    box.innerHTML = `<div class="q">${esc(q)}</div><div class="prose">${renderCited(a.answer, a.citations, a.sources)}</div>
      <details class="trace"><summary>${a.trace.length} tool calls over ${a.steps} steps · ${a.citations.length} citations</summary><ol>${trace}</ol>
      ${a.thinking ? `<div class="thinking">${esc(a.thinking)}</div>` : ""}</details>`;
    loadTelemetry();
  } catch (e) { box.querySelector(".skeleton").textContent = "Error: " + e.message; }
}

function histogram(counts) {
  const w = 300, h = 70, max = Math.max(1, ...counts), bw = w / counts.length;
  let s = `<svg viewBox="0 0 ${w} ${h + 16}" style="width:100%;max-width:320px;height:auto">`;
  counts.forEach((c, i) => {
    const bh = c ? Math.max(2, (c / max) * h) : 0, lo = (0.4 + i * 0.06).toFixed(2);
    s += `<rect x="${i * bw}" y="0" width="${bw}" height="${h}" fill="transparent" data-tip="best-match similarity ${lo}–${(+lo + 0.06).toFixed(2)}: ${c} paragraphs"/>`;
    if (bh) s += `<rect x="${i * bw + 1}" y="${h - bh}" width="${bw - 2}" height="${bh}" rx="2" fill="var(--series-1)" pointer-events="none"/>`;
  });
  s += `<line x1="0" x2="${w}" y1="${h + .5}" y2="${h + .5}" stroke="var(--grid)"/>`;
  s += `<text x="0" y="${h + 13}" font-size="10" fill="var(--text-muted)">0.40 · novel</text><text x="${w}" y="${h + 13}" font-size="10" fill="var(--text-muted)" text-anchor="end">unchanged · 1.00</text>`;
  return s + "</svg>";
}
async function loadDrift(ticker) {
  if (!ticker) { $("#drift").innerHTML = `<div class="skeleton">No filings indexed. Set SEC_USER_AGENT to pull 10-Ks.</div>`; return; }
  $("#drift").innerHTML = `<div class="skeleton">Comparing ${esc(ticker)}'s two latest 10-Ks…</div>`;
  try {
    const d = await api(`/api/drift?ticker=${encodeURIComponent(ticker)}`);
    if (d.error) { $("#drift").innerHTML = `<div class="skeleton">${esc(d.error)}</div>`; return; }
    [...d.added, ...d.dropped].forEach((x) => (state.sources[x.id] = x));
    const risk = (x, cls) => `<div class="risk ${cls}" data-src="${esc(x.id)}" style="cursor:pointer">
      ${esc(x.title.split("· ")[1] || x.title)} <span class="nv num">novelty ${x.novelty.toFixed(2)}</span></div>`;
    $("#drift").innerHTML = `
      <div class="drift-top">
        <div class="stat"><div class="v num">${(d.stability * 100).toFixed(0)}%</div><div class="l">median match to FY${esc(d.old_year)}</div></div>
        <div class="stat"><div class="v num">${d.n_new_chunks} / ${d.n_old_chunks}</div><div class="l">paragraphs FY${esc(d.new_year)} / FY${esc(d.old_year)}</div></div>
        <div style="flex:1;min-width:220px">${histogram(d.similarity_histogram)}</div>
      </div>
      <div class="prose">${d.summary ? renderCited(d.summary.markdown, d.summary.citations, {}) : ""}</div>
      <div class="cols"><div><h3 class="small" style="margin:8px 0 2px">New or rewritten in FY${esc(d.new_year)}</h3>${d.added.map((x) => risk(x, "added")).join("")}</div>
        <div><h3 class="small" style="margin:8px 0 2px">Dropped since FY${esc(d.old_year)}</h3>${d.dropped.map((x) => risk(x, "dropped")).join("") || `<div class="muted small">None below threshold</div>`}</div></div>`;
  } catch (e) { $("#drift").innerHTML = `<div class="skeleton">Error: ${esc(e.message)}</div>`; }
}

async function loadTelemetry() {
  const t = await api("/api/telemetry");
  const rows = Object.entries(t.models).map(([m, v]) => `<tr><td class="mono">${esc(m)}</td><td class="r num">${v.calls}</td>
    <td class="r num">${v.items || "–"}</td><td class="r num">${(v.input_tokens / 1000).toFixed(1)}k/${(v.output_tokens / 1000).toFixed(1)}k</td>
    <td class="r num">${v.p50_ms}ms</td></tr><tr><td colspan="5" class="muted small" style="padding-top:0">${v.features.map(esc).join(" · ")}</td></tr>`).join("");
  $("#telemetry").innerHTML = `<table><thead><tr><th>model</th><th class="r">calls</th><th class="r">items</th><th class="r">tokens in/out</th><th class="r">p50</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function renderIndexStats(s) {
  const kb = (b) => (b / 1024).toFixed(0) + " KB";
  $("#index-stats").innerHTML = `<table><thead><tr><th>index</th><th class="r">docs</th><th class="r">float32</th><th class="r">int8</th><th class="r">${s.index_stats[0].coarse_dim}-bit</th></tr></thead><tbody>` +
    s.index_stats.map((x) => `<tr><td>${esc(x.name)}</td><td class="r num">${x.docs}</td><td class="r num">${kb(x.bytes_float32_equiv)}</td>
      <td class="r num">${kb(x.bytes_int8)}</td><td class="r num">${kb(x.bytes_binary_prefix)}</td></tr>`).join("") + "</tbody></table>";
  $("#log").textContent = (s.status || []).join("\n");
}

/* ---------------- boot ---------------- */
async function loadSnapshot() {
  const s = await api("/api/snapshot");
  state.snap = s;
  renderHeader(s); renderTiles(s); renderThemes(s); renderFeed(s.feed); renderIndexStats(s);
  const sel = $("#drift-ticker"), cur = sel.value;
  sel.innerHTML = s.filing_tickers.map((t) => `<option ${t === cur ? "selected" : ""}>${esc(t)}</option>`).join("");
  return s;
}
async function boot() {
  renderSuggestions();
  let s = await loadSnapshot();
  while (!s.last_refresh) { await new Promise((r) => setTimeout(r, 1500)); s = await loadSnapshot(); }
  loadBriefing(false);
  loadDrift($("#drift-ticker").value);
  loadTelemetry();
}
$("#drift-ticker").onchange = (e) => loadDrift(e.target.value);
$("#regen").onclick = () => loadBriefing(true);
$("#ask-form").onsubmit = (e) => { e.preventDefault(); ask(); };
$("#search-form").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("#search-input").value.trim();
  if (!q) return renderFeed(state.snap.feed);
  const hits = await api(`/api/search?q=${encodeURIComponent(q)}&k=15`);
  const themeOf = Object.fromEntries(state.snap.feed.map((d) => [d.id, d.theme]));
  renderFeed(hits.map((h) => ({ ...h, theme: themeOf[h.id] })), `Top ${hits.length} semantic matches for “${q}”`);
  loadTelemetry();
};
$("#refresh").onclick = async () => {
  const b = $("#refresh");
  b.disabled = true; b.textContent = "Refreshing…";
  try { await api("/api/refresh", { method: "POST" }); await loadSnapshot(); loadBriefing(true); loadDrift($("#drift-ticker").value); loadTelemetry(); }
  catch (e) { $("#refreshed").textContent = "Refresh failed: " + e.message; }
  finally { b.disabled = false; b.textContent = "Refresh data"; }
};
boot().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<div class="banner">Failed to load: ${esc(e.message)}</div>`); });
