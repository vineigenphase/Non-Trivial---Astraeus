/* Astraeus UI controller. Vanilla JS; talks to astraeus/app.py. */
"use strict";
const $ = s => document.querySelector(s);
const fmt = (v, d = 3) => v === null || v === undefined ? "—" : Number(v).toFixed(d);
const MODES = ["stuck", "tip_over", "collision", "nav_miss", "timeout"];
const MODE_COL = { stuck: "#e0554f", tip_over: "#f2a03d", collision: "#c76bff", nav_miss: "#37b6ff", timeout: "#7d8899", success: "#4fd97a" };

const S = {
  meta: null, world: null, current: { kind: "whatif", index: null }, x: {}, seed: 0,
  playing: null, cem: [], busy: null, gl: null, campaign: null,
};

async function api(path, body, method) {
  const r = await fetch(path, body ? { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!r.ok) { const t = await r.text(); throw new Error(`${r.status} ${t}`); }
  return r.json();
}
function log(msg) {
  const el = $("#log"), d = document.createElement("div");
  d.textContent = `${new Date().toLocaleTimeString()}  ${msg}`; el.appendChild(d); el.scrollTop = el.scrollHeight;
}

/* ---------------------------------------------------------------- dims */
function buildDims() {
  const box = $("#dims"); box.innerHTML = "";
  for (const d of S.meta.dims) {
    const [lo, hi] = S.meta.bounds[d];
    const row = document.createElement("div"); row.className = "dim";
    row.innerHTML = `<label>${d}</label><input type="range" min="${lo}" max="${hi}" step="${(hi - lo) / 400}" data-dim="${d}">
      <div class="val"><span id="v-${d}"></span><span class="unit">${S.meta.units[d]}</span></div>`;
    box.appendChild(row);
  }
  box.querySelectorAll("input[type=range]").forEach(inp => {
    inp.addEventListener("input", () => { S.x[inp.dataset.dim] = parseFloat(inp.value); showX(); scheduleSim(); });
  });
  for (const sel of ["#sample-prior", "#cem-prior", "#elite-prior", "#sens-prior"]) {
    const el = $(sel); el.innerHTML = "";
    for (const p of S.meta.priors) { const o = document.createElement("option"); o.value = p.name; o.textContent = `${p.name} · ${p.label}`; el.appendChild(o); }
  }
  $("#cem-prior").value = "P2"; $("#elite-prior").value = "P2"; $("#sens-prior").value = "P2";
}
function setX(x, source) {
  S.x = { ...x };
  for (const d of S.meta.dims) { const inp = document.querySelector(`input[data-dim=${d}]`); if (inp) inp.value = x[d]; }
  showX(); $("#x-source").textContent = source || "what-if";
}
function showX() { for (const d of S.meta.dims) $(`#v-${d}`).textContent = fmt(S.x[d], d === "sun_psi" ? 1 : 3); }

let simTimer = null;
function scheduleSim() { clearTimeout(simTimer); simTimer = setTimeout(simulateWhatIf, 120); }
async function simulateWhatIf() {
  S.seed = parseInt($("#seed").value || "0", 10);
  S.current = { kind: "whatif", index: null };
  try {
    const w = await api("/api/simulate", { x: S.x, seed: S.seed });
    loadWorld(w, "what-if");
  } catch (e) { log("simulate failed: " + e.message); }
}

/* --------------------------------------------------------------- world */
function loadWorld(w, source) {
  S.world = w; S.gl.setWorld(w);
  $("#frame").max = w.n_frames - 1; $("#frame").value = w.n_frames - 1;
  $("#x-source").textContent = source;
  updateHud(w.n_frames - 1);
  $("#hud-ep").textContent = (w.index !== null && w.index !== undefined ? `episode #${w.index} · ` : "what-if · ") + `seed ${w.seed} · ${w.rocks.length} rocks`;
  refreshFrame();
}
function updateHud(k) {
  const w = S.world; if (!w) return;
  const tr = w.trace, last = k === w.n_frames - 1;
  const oc = last ? w.outcome : "driving";
  const el = $("#hud-outcome"); el.textContent = oc.toUpperCase(); el.className = "outcome " + oc;
  $("#frame-label").textContent = `${k + 1} / ${w.n_frames}  t=${fmt(tr.t[k], 1)}s`;
  const dist = Math.hypot(w.goal.x - tr.x[k], tr.y[k]);
  const voe = Math.hypot(tr.ex[k] - tr.x[k], tr.ey[k] - tr.y[k]);
  const tilt = Math.max(Math.abs(tr.pitch[k]), Math.abs(tr.roll[k])) * 180 / Math.PI;
  const c = (v, bad) => bad ? `<span style="color:var(--red)">${v}</span>` : v;
  $("#hud-tele").innerHTML = [
    `dist→goal ${fmt(dist, 2)} m`, `speed ${fmt(tr.v[k], 2)} m/s`, `slip ${c(fmt(tr.slip[k], 2), tr.slip[k] > 0.5)}`,
    `sinkage ${fmt(tr.sinkage[k] * 100, 1)} cm`, `pitch/roll ${c(fmt(tr.pitch[k] * 180 / Math.PI, 1) + "/" + fmt(tr.roll[k] * 180 / Math.PI, 1) + "°", tilt > 20)}`,
    `visibility ${c(fmt(tr.visibility[k], 2), tr.visibility[k] < 0.4)}`, `VO error ${c(fmt(voe, 2) + " m", voe > 1)}`,
    `sun el ${fmt(w.x.sun_e, 1)}° az ${fmt(w.x.sun_psi, 0)}°  τ ${fmt(w.x.tau_dust, 2)}`,
  ].join("<br>");
}
$("#frame").addEventListener("input", e => { const k = parseInt(e.target.value, 10); S.gl.setFrame(k); updateHud(k); });
$("#btn-play").addEventListener("click", () => {
  if (S.playing) { clearInterval(S.playing); S.playing = null; $("#btn-play").textContent = "▶"; return; }
  let k = parseInt($("#frame").value, 10); if (k >= S.world.n_frames - 1) k = 0;
  $("#btn-play").textContent = "❚❚";
  S.playing = setInterval(() => {
    k++; if (k >= S.world.n_frames) { clearInterval(S.playing); S.playing = null; $("#btn-play").textContent = "▶"; refreshFrame(); return; }
    $("#frame").value = k; S.gl.setFrame(k); updateHud(k);
  }, 80);
});
$("#cam-follow").addEventListener("change", e => { S.gl.cam.follow = e.target.checked; S.gl.draw(); });

/* --------------------------------------------------------------- frame */
function frameUrl(sheet) {
  const cam = $("#camera").value, q = $("#quality").value, k = parseInt($("#frame").value, 10);
  if (S.current.kind === "episode") {
    return sheet ? `/api/episode/${S.current.index}/sheet.png?camera=${cam}`
                 : `/api/episode/${S.current.index}/frame.png?camera=${cam}&frame=${k}&w=960&h=540&q=${q}`;
  }
  const qs = S.meta.dims.map(d => `${d}=${S.x[d]}`).join("&") + `&seed=${S.seed}`;
  return sheet ? `/api/whatif/sheet.png?${qs}&camera=${cam}` : `/api/whatif/frame.png?${qs}&camera=${cam}&frame=${k}&w=960&h=540&q=${q}`;
}
let frameTimer = null;
function refreshFrame() { clearTimeout(frameTimer); frameTimer = setTimeout(() => setFrame(frameUrl(false)), 150); }
function setFrame(url) {
  const img = $("#frame-img"), t0 = performance.now();
  $("#frame-meta").textContent = "rendering …";
  img.onload = () => { $("#frame-meta").textContent = `${url.split("?")[0].split("/").pop()} · ${(performance.now() - t0).toFixed(0)} ms · software render (synthetic, deterministic)`; };
  img.onerror = () => { $("#frame-meta").textContent = "render failed"; };
  img.src = url;
}
$("#btn-render").addEventListener("click", () => setFrame(frameUrl(false)));
$("#btn-sheet").addEventListener("click", () => setFrame(frameUrl(true)));
$("#camera").addEventListener("change", refreshFrame);
$("#quality").addEventListener("change", refreshFrame);

/* ------------------------------------------------------------ campaign */
function renderCampaign(summ) {
  S.campaign = summ;
  $("#b-n").textContent = `n = ${summ.n_total}` + (summ.n_cem ? ` (${summ.n_cem} CEM)` : "");
  $("#c-status").textContent = summ.n_mc ? `${summ.n_mc} MC · ${summ.episodes_per_s || "—"} ep/s · raw q fail ${fmt(summ.raw_fail_rate_q, 3)}` : "empty";
  const tb = $("#prior-table tbody"); tb.innerHTML = "";
  for (const r of summ.priors) {
    const tr = document.createElement("tr");
    const cov = S.meta.support_coverage[r.prior];
    const dim = !r.reliable;
    tr.style.opacity = dim ? 0.55 : 1;
    let note = "";
    if (cov < 0.5) note = ` <span class="chip" title="${((1 - cov) * 100).toFixed(0)}% of this prior's mass lies outside the simulated support">out of support</span>`;
    else if (r.ess < S.meta.min_ess) note = ` <span class="chip">ESS&lt;${S.meta.min_ess}</span>`;
    tr.innerHTML = `<td title="${r.label}">${r.prior}${note}</td><td class="num">${r.p_fail === null ? "—" : fmt(r.p_fail, 3)}</td>
      <td class="num">${r.ci[0] === null ? "—" : `[${fmt(r.ci[0], 3)}, ${fmt(r.ci[1], 3)}]`}</td>
      <td class="num">${fmt(r.ess, 0)}<span class="muted">/${r.n}</span></td>
      <td>${r.top_mode ? `<span class="chip" style="border-color:${MODE_COL[r.top_mode]};color:${MODE_COL[r.top_mode]}">${r.top_mode}</span>` : "—"}</td>`;
    tb.appendChild(tr);
  }
  renderShift(summ);
  loadElites(); loadSensitivity();
}
function renderShift(summ) {
  const box = $("#shift"); box.innerHTML = "";
  for (const r of summ.shift.rows) {
    const cov = S.meta.support_coverage[r.prior];
    const row = document.createElement("div"); row.className = "modes";
    const total = r.p_fail || 0;
    let bars = "";
    for (const m of MODES) { const v = r.modes[m] || 0; bars += `<i class="m-${m}" style="width:${(v * 100).toFixed(2)}%" title="${m}: ${fmt(v, 3)}"></i>`; }
    row.innerHTML = `<span class="mono" style="opacity:${r.reliable ? 1 : .5}">${r.prior}</span><div class="bar" title="${r.label}">${bars}</div>
      <span class="num mono" style="opacity:${r.reliable ? 1 : .5}">${cov < 0.5 ? "n/a" : fmt(total, 3)}</span>`;
    box.appendChild(row);
  }
  const leg = document.createElement("div"); leg.className = "small muted"; leg.style.marginTop = "6px";
  leg.innerHTML = MODES.map(m => `<span class="legend"><span style="background:${MODE_COL[m]}"></span>${m}</span>`).join(" &nbsp;");
  box.appendChild(leg);
  const tops = summ.shift.top_modes;
  $("#shift-note").innerHTML = summ.n_mc === 0 ? "" : summ.shift.ranking_shifts
    ? `<span style="color:var(--amber)">The dominant failure mode changes with the prior:</span> ${Object.entries(tops).map(([p, m]) => `${p}→${m}`).join(", ")}. Whichever prior you commit to decides what you should have hardened against.`
    : `Same dominant failure mode under every estimable prior: ${Object.entries(tops).map(([p, m]) => `${p}→${m}`).join(", ")}.`;
}
async function loadElites() {
  if (!S.campaign || !S.campaign.n_total) { $("#elite-table tbody").innerHTML = ""; return; }
  const p = $("#elite-prior").value;
  const d = await api(`/api/episodes?prior=${p}&k=12`);
  const tb = $("#elite-table tbody"); tb.innerHTML = "";
  for (const e of d.elites) {
    const tr = document.createElement("tr"); tr.className = "clickable" + (S.current.kind === "episode" && S.current.index === e.index ? " sel" : "");
    tr.innerHTML = `<td class="num">${e.index}${e.source === "cem" ? "<sup>c</sup>" : ""}</td><td><span class="chip" style="color:${MODE_COL[e.outcome]};border-color:${MODE_COL[e.outcome]}">${e.outcome}</span></td>
      <td class="num">${fmt(e.severity, 2)}</td><td class="num">${e.source === "cem" ? "<span class='muted'>cem</span>" : e.weight === undefined ? "—" : e.weight.toExponential(1)}</td>
      <td class="num">${fmt(e.max_tilt_deg, 1)}</td><td class="num">${fmt(e.max_slip, 2)}</td><td class="num">${fmt(e.min_visibility, 2)}</td>`;
    tr.addEventListener("click", () => openEpisode(e.index));
    tb.appendChild(tr);
  }
}
$("#elite-prior").addEventListener("change", loadElites);
async function openEpisode(i) {
  try {
    const w = await api(`/api/episode/${i}`);
    S.current = { kind: "episode", index: i };
    setX(w.x, `episode #${i}`); $("#seed").value = w.seed; S.seed = w.seed;
    loadWorld(w, `episode #${i}`); loadElites();
  } catch (e) { log("episode load failed: " + e.message); }
}
async function loadSensitivity() {
  const box = $("#sens"); box.innerHTML = "";
  if (!S.campaign || !S.campaign.n_mc) return;
  const d = await api(`/api/sensitivity?prior=${$("#sens-prior").value}`);
  for (const dim of S.meta.dims) {
    const wrap = document.createElement("div"); wrap.innerHTML = `<div class="small mono muted">${dim}</div><canvas class="chart" style="height:64px"></canvas>`;
    box.appendChild(wrap);
    drawBars(wrap.querySelector("canvas"), d[dim].rate, d[dim].count, 1.0);
  }
}
$("#sens-prior").addEventListener("change", loadSensitivity);
function drawBars(cv, vals, counts, vmax) {
  const dpr = window.devicePixelRatio || 1; cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr); const W = cv.clientWidth, H = cv.clientHeight;
  g.clearRect(0, 0, W, H); const n = vals.length, bw = W / n;
  for (let i = 0; i < n; i++) {
    const v = vals[i]; if (v === null) continue;
    const h = (v / vmax) * (H - 4);
    g.fillStyle = counts && counts[i] < 15 ? "#3a4656" : "#e0554f"; g.fillRect(i * bw + 1, H - h, bw - 2, h);
  }
  g.fillStyle = "#7d8899"; g.font = "9px monospace"; g.fillText("0", 2, H - 2); g.fillText("1", 2, 9);
}

/* ----------------------------------------------------------------- CEM */
function drawCem() {
  const cv = $("#cem-chart"), dpr = window.devicePixelRatio || 1; cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr); const W = cv.clientWidth, H = cv.clientHeight; g.clearRect(0, 0, W, H);
  const h = S.cem; if (!h.length) { g.fillStyle = "#7d8899"; g.font = "11px monospace"; g.fillText("no search yet — pick a prior and press search", 8, H / 2); return; }
  const n = h[0].iters, xs = i => 24 + (W - 32) * (i / Math.max(n - 1, 1));
  g.strokeStyle = "#232c39"; g.beginPath(); for (const f of [0, .5, 1]) { g.moveTo(24, 6 + (H - 22) * (1 - f)); g.lineTo(W - 8, 6 + (H - 22) * (1 - f)); } g.stroke();
  const line = (key, col, scale) => { g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath();
    h.forEach((it, i) => { const y = 6 + (H - 22) * (1 - Math.min(1, it[key] / scale)); i ? g.lineTo(xs(i), y) : g.moveTo(xs(i), y); }); g.stroke(); };
  line("fail_rate", "#e0554f", 1); line("elite_mean_severity", "#37b6ff", 3);
  g.fillStyle = "#7d8899"; g.font = "9px monospace"; g.fillText("fail rate (red) · elite severity/3 (blue)", 24, H - 4);
  for (let i = 0; i < n; i++) g.fillText(String(i + 1), xs(i) - 3, H - 12);
  const b = h[h.length - 1].best;
  $("#cem-best").innerHTML = `best J=${fmt(b.J, 2)} · ${b.outcome} · episode #${b.index} · <a href="#" id="cem-open">replay</a><br>` +
    S.meta.dims.map(d => `${d}=${fmt(b.x[d], 2)}`).join("  ");
  $("#cem-open").addEventListener("click", ev => { ev.preventDefault(); openEpisode(b.index); });
}
$("#btn-cem").addEventListener("click", async () => {
  S.cem = []; drawCem();
  try {
    await api("/api/cem", { prior: $("#cem-prior").value, iters: parseInt($("#cem-iters").value, 10), batch: parseInt($("#cem-batch").value, 10) });
    $("#cem-status").textContent = "running…";
  } catch (e) { log("CEM: " + e.message); }
});

/* ------------------------------------------------------------- buttons */
$("#btn-run").addEventListener("click", async () => {
  const n = parseInt($("#run-n").value, 10);
  setBusy("campaign");
  try { const s = await api("/api/campaign/run", { n }); renderCampaign(s); }
  catch (e) { log("run: " + e.message); }
  finally { setBusy(null); $("#run-prog").style.width = "0"; }
});
$("#btn-reset").addEventListener("click", async () => { const s = await api("/api/campaign/reset", { seed: parseInt($("#seed").value || "0", 10) }); renderCampaign(s); S.cem = []; drawCem(); });
$("#btn-save").addEventListener("click", async () => { const r = await api("/api/save", { name: "campaign" }); log(`saved ${r.saved}`); });
$("#btn-sample").addEventListener("click", async () => {
  const p = $("#sample-prior").value;
  try {
    const d = await api(`/api/priors/${p}/sample`);
    $("#seed").value = d.seed;
    setX(d.x, `x ~ ${p}` + (d.clipped ? " (clipped to support)" : "")); simulateWhatIf();
  } catch (e) { log("sample: " + e.message); }
});
$("#btn-resim").addEventListener("click", simulateWhatIf);
$("#seed").addEventListener("change", simulateWhatIf);
function setBusy(b) { S.busy = b; $("#btn-run").disabled = $("#btn-cem").disabled = $("#btn-reset").disabled = !!b; }

/* ----------------------------------------------------------- websocket */
function connectWs() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => { $("#b-ws").textContent = "ws live"; $("#b-ws").className = "badge ok"; };
  ws.onclose = () => { $("#b-ws").textContent = "ws closed"; $("#b-ws").className = "badge bad"; setTimeout(connectWs, 2000); };
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.type === "progress") { $("#run-prog").style.width = (100 * m.done / m.total).toFixed(1) + "%"; }
    else if (m.type === "log") log(m.msg);
    else if (m.type === "campaign_done") { renderCampaign(m.summary); }
    else if (m.type === "cem_iter") { S.cem.push(m); drawCem(); $("#cem-status").textContent = `iter ${m.iter}/${m.iters} · fail ${fmt(m.fail_rate, 2)}`; setBusy("cem"); }
    else if (m.type === "cem_done") { $("#cem-status").textContent = `done under ${m.prior}`; setBusy(null); renderCampaign(m.summary); }
  };
}

/* ---------------------------------------------------------------- boot */
(async function boot() {
  S.gl = new AstraeusWorld($("#world"));
  S.meta = await api("/api/priors");
  const h = await api("/api/health");
  const pb = $("#b-provider"); pb.textContent = `provider: ${h.provider.provider}` + (h.provider.live ? " (live)" : "");
  pb.className = "badge " + (h.provider.provider === "local" ? "ok" : h.provider.live ? "ok" : "warn");
  pb.title = h.provider.status || "local software renderer — no network";
  buildDims();
  setX(S.meta.nominal, "P2 nominal");
  connectWs();
  drawCem();
  await simulateWhatIf();
  renderCampaign(await api("/api/campaign"));
  log(`Astraeus ${h.version} ready — provider ${h.provider.provider}, ${h.n_episodes} episodes in memory`);
})();
