"use strict";
const $ = s => document.querySelector(s);
const fmt = (v, d = 3) => v === null || v === undefined ? "—" : Number(v).toFixed(d);
const MODE_COL = { stuck: "#e0554f", tip_over: "#f2a03d", collision: "#c76bff", nav_miss: "#37b6ff", timeout: "#7d8899", success: "#3a4656" };
let D = null, META = null;

function ctx(cv) {
  const dpr = window.devicePixelRatio || 1; cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, cv.clientWidth, cv.clientHeight);
  return [g, cv.clientWidth, cv.clientHeight];
}
function hist(cv, counts, edges, col, overlay) {
  const [g, W, H] = overlay ? [cv.getContext("2d"), cv.clientWidth, cv.clientHeight] : ctx(cv);
  const max = Math.max(...counts, 1), n = counts.length, bw = (W - 8) / n;
  g.fillStyle = col;
  for (let i = 0; i < n; i++) { const h = counts[i] / max * (H - 18); g.fillRect(4 + i * bw, H - 14 - h, Math.max(bw - 1, 1), h); }
  g.fillStyle = "#7d8899"; g.font = "9px monospace";
  g.fillText(fmt(edges[0], 2), 4, H - 3); g.fillText(fmt(edges[edges.length - 1], 2), W - 40, H - 3);
}
function fillSel(sel, opts, val) { sel.innerHTML = ""; for (const o of opts) { const e = document.createElement("option"); e.value = o; e.textContent = o; sel.appendChild(e); } if (val) sel.value = val; }

function render() {
  $("#b-n").textContent = `n = ${D.n} (${D.n_mc} MC, ${D.n_cem} CEM)`;
  $("#min-ess").textContent = D.min_ess;
  const tb = $("#rep tbody"); tb.innerHTML = "";
  for (const p of META.priors) {
    const r = D.priors[p.name] ? D.priors[p.name].report : null;
    const cov = META.support_coverage[p.name];
    const tr = document.createElement("tr");
    tr.innerHTML = r ? `<td title="${p.label}">${p.name}</td><td class="num" style="${cov < 0.5 ? "color:var(--red)" : ""}">${fmt(cov, 3)}</td><td class="num">${fmt(r.ess, 1)}</td><td class="num">${fmt(r.ess_frac, 3)}</td>
      <td class="num" style="${r.max_weight_frac > 0.1 ? "color:var(--amber)" : ""}">${fmt(r.max_weight_frac, 4)}</td><td class="num">${fmt(r.p_fail, 4)}</td>
      <td class="num">${r.ci[0] === null ? "—" : `[${fmt(r.ci[0], 3)}, ${fmt(r.ci[1], 3)}]`}</td><td>${r.reliable ? "<span style='color:var(--green)'>yes</span>" : "<span style='color:var(--red)'>no</span>"}</td>`
      : `<td>${p.name}</td><td class="num">${fmt(cov, 3)}</td><td colspan="6" class="muted">no MC episodes yet</td>`;
    tb.appendChild(tr);
  }
  drawW(); drawMarg(); drawScatter(); drawCem();
  $("#log").textContent = D.log.map(l => `[${fmt(l.t, 1).padStart(7)}s] ${l.msg}`).join("\n");
}
function drawW() {
  const p = D.priors[$("#w-prior").value];
  const cv = $("#w-hist"); if (!p) { ctx(cv); return; }
  hist(cv, p.log10w_hist, p.log10w_edges, "#37b6ff");
}
function drawMarg() {
  const box = $("#marg"); box.innerHTML = "";
  const p = $("#m-prior").value;
  for (const d of META.dims) {
    const w = document.createElement("div"); w.innerHTML = `<div class="small mono muted">${d} <span style="float:right">${META.units[d]}</span></div><canvas class="chart" style="height:90px"></canvas>`;
    box.appendChild(w);
    const cv = w.querySelector("canvas");
    hist(cv, D.marginals.Q[d], D.edges[d], "#3a4656");
    hist(cv, D.marginals[p][d], D.edges[d], "rgba(242,160,61,.75)", true);
  }
}
function drawScatter() {
  const cv = $("#scatter"), [g, W, H] = ctx(cv);
  if (!D.sample) { g.fillStyle = "#7d8899"; g.fillText("run a campaign first", 10, 20); return; }
  const ix = META.dims.indexOf($("#sx").value), iy = META.dims.indexOf($("#sy").value);
  const [x0, x1] = META.bounds[META.dims[ix]], [y0, y1] = META.bounds[META.dims[iy]];
  const S = D.sample;
  const order = S.fail.map((f, i) => i).sort((a, b) => S.fail[a] - S.fail[b]);
  for (const i of order) {
    const x = 30 + (S.X[i][ix] - x0) / (x1 - x0) * (W - 40), y = H - 18 - (S.X[i][iy] - y0) / (y1 - y0) * (H - 28);
    g.fillStyle = MODE_COL[S.outcome[i]]; g.globalAlpha = S.fail[i] ? 0.85 : 0.5;
    g.beginPath(); g.arc(x, y, S.fail[i] ? 2.4 : 1.6, 0, 6.283); g.fill();
  }
  g.globalAlpha = 1; g.fillStyle = "#7d8899"; g.font = "10px monospace";
  g.fillText(META.dims[ix], W / 2 - 20, H - 4); g.save(); g.translate(10, H / 2); g.rotate(-Math.PI / 2); g.fillText(META.dims[iy], -20, 0); g.restore();
}
function drawCem() {
  const box = $("#cem"); box.innerHTML = "";
  if (!D.cem.length) { box.innerHTML = "<div class='small muted'>no CEM runs yet</div>"; return; }
  for (const run of D.cem) {
    const last = run.iters[run.iters.length - 1];
    const el = document.createElement("div"); el.className = "small"; el.style.marginBottom = "8px";
    el.innerHTML = `<div class="mono">under ${run.prior}: ${run.iters.length} iters, fail rate ${run.iters.map(i => fmt(i.fail_rate, 2)).join(" → ")}</div>
      <div class="mono muted">best J=${fmt(last.best.J, 3)} ${last.best.outcome} #${last.best.index} · σ shrink: ${META.dims.map(d => `${d} ${fmt(run.iters[0].sigma[d], 2)}→${fmt(last.sigma[d], 2)}`).join(", ")}</div>`;
    box.appendChild(el);
  }
}
async function load() {
  D = await (await fetch("/api/internals")).json();
  render();
}
(async function boot() {
  META = await (await fetch("/api/priors")).json();
  const names = META.priors.map(p => p.name);
  fillSel($("#w-prior"), names, "P2"); fillSel($("#m-prior"), names, "P2");
  fillSel($("#sx"), META.dims, "slope_deg"); fillSel($("#sy"), META.dims, "k_soil");
  $("#w-prior").addEventListener("change", drawW); $("#m-prior").addEventListener("change", drawMarg);
  $("#sx").addEventListener("change", drawScatter); $("#sy").addEventListener("change", drawScatter);
  $("#btn-refresh").addEventListener("click", load);
  await load();
})();
