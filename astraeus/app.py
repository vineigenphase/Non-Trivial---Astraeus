"""Astraeus web application: the live interactive world model.

    uvicorn astraeus.app:app --port 8000

Everything runs locally with the software renderer by default. The browser gets
a WebGL view of the exact terrain the simulator used (rasterised from the same
deterministic height field), the true and estimated trajectories, and PNG frames
from the renderer or the Reactor provider. The eight disturbance dimensions are
live sliders: moving one re-simulates the episode (~50 ms) and the world updates.

Endpoints
    GET  /                          world model UI
    GET  /internals                 engine room: weights, ESS, marginals, CEM traces
    GET  /api/health
    GET  /api/priors                the candidate priors, proposal, bounds, units
    GET  /api/campaign              summary: per-prior P(fail), CI, ESS, shift matrix
    POST /api/campaign/run          {n}         simulate n more proposal episodes
    POST /api/campaign/reset        {seed}
    GET  /api/episodes              ?prior=P2&k=12   elites (most severe plausible failures)
    GET  /api/episode/{i}           full replay payload (world + trace) for the WebGL view
    GET  /api/episode/{i}/frame.png ?camera=chase&frame=-1&w=960&h=540&q=1
    GET  /api/episode/{i}/sheet.png
    POST /api/simulate              {x: {...8 dims}, seed}  what-if episode, same payload
    GET  /api/sensitivity           ?prior=P2
    GET  /api/internals
    POST /api/cem                   {prior, iters, batch}   adversarial search (streams over /ws)
    POST /api/save  /api/load       campaign persistence under runs/
    WS   /ws                        progress, cem_iter, campaign_done, log
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from . import campaign as C
from . import priors as PR
from . import providers as PV
from . import render as R
from . import rover as RV
from .priors import BOUNDS, DIMS, UNITS

log = logging.getLogger("astraeus.app")
STATIC = Path(__file__).parent / "static"
RUNS = Path(os.getenv("ASTRAEUS_RUNS", "runs"))
DEFAULT_SEED = int(os.getenv("ASTRAEUS_SEED", "0"))
WORLD_RES = 0.5                    # m, DEM cell sent to the browser
WORLD_X = (-6.0, 38.0)
WORLD_Y = (-12.0, 12.0)
# Mission-nominal point: P2 (VIPER-spec south pole) centre, the natural what-if start.
NOMINAL_X: Dict[str, float] = {"k_soil": 2.0, "theta_r": 0.724, "r_terrain": 1.0, "rho_rock": 2.0,
                               "slope_deg": 4.0, "sun_e": 1.5, "sun_psi": 200.0, "tau_dust": 0.05}


def _j(o: Any) -> Any:
    """numpy -> plain JSON-serialisable."""
    if isinstance(o, dict):
        return {k: _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_j(v) for v in o.tolist()]
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def world_payload(res: RV.EpisodeResult, index: Optional[int] = None) -> Dict[str, Any]:
    """Everything the browser needs to draw and scrub one recorded episode."""
    assert res.trace is not None and res.terrain is not None
    tr = res.trace
    dem = res.terrain.dem(0, WORLD_RES, WORLD_X, WORLD_Y)
    x = res.X[0]
    e, psi = np.radians(x[DIMS.index("sun_e")]), np.radians(x[DIMS.index("sun_psi")])
    sun = [float(np.cos(psi) * np.cos(e)), float(np.sin(psi) * np.cos(e)), float(np.sin(e))]
    keys = ("t", "x", "y", "z", "heading", "pitch", "roll", "v", "slip", "sinkage", "visibility", "ex", "ey")
    return _j({
        "index": index, "seed": int(res.seed[0]), "x": PR.to_dict(x), "row": res.row(0),
        "outcome": RV.OUTCOMES[int(res.outcome[0])], "sun": sun,
        "goal": {"x": RV.GOAL_X, "y": 0.0, "tol": RV.GOAL_TOL},
        "dem": {"x0": float(dem["x"][0]), "y0": float(dem["y"][0]), "res": WORLD_RES,
                "nx": int(len(dem["x"])), "ny": int(len(dem["y"])),
                "z": np.round(dem["z"], 3).ravel()},
        "rocks": res.terrain.rocks_of(0), "craters": res.terrain.craters_of(0),
        "trace": {k: np.round(tr[k][0], 4) for k in keys},
        "n_frames": int(tr["x"].shape[1]),
        "rover": {"half_len": RV.HALF_LEN, "half_wid": RV.HALF_WID, "wheel_r": R.WHEEL_R},
    })


class State:
    """Process-wide state: one campaign, one provider, one event bus."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.campaign = C.Campaign(seed=DEFAULT_SEED)
        self.provider = PV.make_provider()
        self.busy: Optional[str] = None
        self.cem_history: List[Dict[str, Any]] = []
        self.log: List[Dict[str, Any]] = []
        self.clients: List[WebSocket] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._replays: "OrderedDict[int, RV.EpisodeResult]" = OrderedDict()
        self.t0 = time.time()

    # events ---------------------------------------------------------------
    def emit(self, kind: str, **data: Any) -> None:
        msg = {"type": kind, "t": round(time.time() - self.t0, 3), **_j(data)}
        if kind == "log":
            self.log = (self.log + [msg])[-200:]
        loop = self.loop
        if loop is None or not self.clients:
            return
        text = json.dumps(msg, separators=(",", ":"))
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(self._send(ws, text), loop)

    async def _send(self, ws: WebSocket, text: str) -> None:
        try:
            await ws.send_text(text)
        except Exception:                          # noqa: BLE001 — client went away
            if ws in self.clients:
                self.clients.remove(ws)

    # campaign -------------------------------------------------------------
    def run_chunk(self, n: int) -> Dict[str, Any]:
        with self.lock:
            self.busy = "campaign"
            try:
                t0 = time.perf_counter()
                self.emit("log", msg=f"simulating {n} proposal episodes")

                def prog(step, steps, done, total):
                    self.emit("progress", phase="campaign", step=step, steps=steps, done=done, total=total)
                self.campaign.run_chunk(n, progress=prog)
                self._replays.clear()
                summ = self.campaign.summary()
                self.emit("log", msg=f"{n} episodes in {time.perf_counter() - t0:.1f}s; "
                                     f"n={self.campaign.n}, raw fail rate under q={summ['raw_fail_rate_q']}")
                self.emit("campaign_done", summary=summ)
                return summ
            finally:
                self.busy = None

    def reset(self, seed: int) -> None:
        with self.lock:
            self.campaign = C.Campaign(seed=seed)
            self.cem_history = []
            self._replays.clear()
            self.emit("log", msg=f"campaign reset, seed={seed}")
            self.emit("campaign_done", summary=self.campaign.summary())

    def run_cem(self, prior: PR.Prior, iters: int, batch: int) -> None:
        with self.lock:
            self.busy = "cem"
            try:
                self.emit("log", msg=f"CEM adversarial search under {prior.name}: {iters}x{batch}")
                hist: List[Dict[str, Any]] = []
                for it in self.campaign.cem_search(prior, iters=iters, batch=batch):
                    hist.append(it)
                    self.emit("cem_iter", **it)
                self.cem_history.append({"prior": prior.name, "iters": hist})
                self._replays.clear()
                self.emit("log", msg=f"CEM done: best J={hist[-1]['best']['J']:.3f} "
                                     f"({hist[-1]['best']['outcome']}) at episode {hist[-1]['best']['index']}")
                self.emit("cem_done", prior=prior.name, best=hist[-1]["best"], summary=self.campaign.summary())
            finally:
                self.busy = None

    # episodes -------------------------------------------------------------
    def replay(self, i: int) -> RV.EpisodeResult:
        if i < 0 or i >= self.campaign.n:
            raise HTTPException(404, f"episode {i} not in campaign (n={self.campaign.n})")
        if i in self._replays:
            self._replays.move_to_end(i)
            return self._replays[i]
        res = self.campaign.replay(i)
        self._replays[i] = res
        while len(self._replays) > 32:
            self._replays.popitem(last=False)
        return res


STATE = State()
app = FastAPI(title="Astraeus", version=__version__)


def _prior(name: str) -> PR.Prior:
    if name not in PR.PRIORS:
        raise HTTPException(400, f"unknown prior {name!r}; choose from {list(PR.PRIORS)}")
    return PR.PRIORS[name]


def _not_busy() -> None:
    if STATE.busy:
        raise HTTPException(409, f"engine busy: {STATE.busy}")


# ------------------------------------------------------------------- pages
@app.on_event("startup")
async def _startup() -> None:
    STATE.loop = asyncio.get_running_loop()
    log.info("Astraeus %s provider=%s seed=%d", __version__, STATE.provider.name, DEFAULT_SEED)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/internals")
async def internals_page() -> FileResponse:
    return FileResponse(STATIC / "internals.html")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# --------------------------------------------------------------------- api
@app.get("/api/health")
async def api_health() -> JSONResponse:
    return JSONResponse({"ok": True, "version": __version__, "provider": STATE.provider.health(),
                         "busy": STATE.busy, "n_episodes": STATE.campaign.n,
                         "deterministic": True, "external_calls": STATE.provider.name != "local"})


@app.get("/api/priors")
async def api_priors() -> JSONResponse:
    return JSONResponse(_j({
        "dims": DIMS, "units": UNITS, "bounds": BOUNDS, "sim_defaults": PR.SIM_DEFAULTS,
        "priors": PR.describe_all(), "proposal": PR.PROPOSAL.describe(),
        "support_coverage": {p.name: C.support_coverage(p) for p in PR.PRIORS.values()},
        "min_ess": C.MIN_ESS, "nominal": NOMINAL_X,
    }))


@app.get("/api/priors/{name}/sample")
async def api_prior_sample(name: str, seed: Optional[int] = None) -> JSONResponse:
    """One draw x ~ p (clipped to the simulator box; the flag says if that bit)."""
    p = _prior(name)
    seed = int(np.random.SeedSequence().entropy % (2**31)) if seed is None else seed
    x = p.sample(np.random.default_rng(seed), 1)[0]
    xc = PR.clip_to_bounds(x)
    return JSONResponse(_j({"prior": name, "seed": seed, "x": PR.to_dict(xc),
                            "clipped": bool(np.any(np.abs(xc - x) > 1e-12))}))


@app.get("/api/campaign")
async def api_campaign() -> JSONResponse:
    return JSONResponse(_j(STATE.campaign.summary()))


class RunBody(BaseModel):
    n: int = Field(500, ge=1, le=20000)


@app.post("/api/campaign/run")
async def api_run(body: RunBody) -> JSONResponse:
    _not_busy()
    summ = await asyncio.to_thread(STATE.run_chunk, body.n)
    return JSONResponse(_j(summ))


class ResetBody(BaseModel):
    seed: int = 0


@app.post("/api/campaign/reset")
async def api_reset(body: ResetBody) -> JSONResponse:
    _not_busy()
    await asyncio.to_thread(STATE.reset, body.seed)
    return JSONResponse(_j(STATE.campaign.summary()))


@app.get("/api/episodes")
async def api_episodes(prior: Optional[str] = None, k: int = 12, mode: Optional[str] = None) -> JSONResponse:
    c = STATE.campaign
    if c.n == 0:
        return JSONResponse({"elites": [], "n": 0})
    p = _prior(prior) if prior else None
    idx = c.elites(k=max(k * 4, 40), prior=p)
    rows = [c.episode(i) for i in idx]
    if mode:
        rows = [r for r in rows if r["outcome"] == mode]
    if p is not None:
        w = c.weights(p)
        for r in rows:
            r["weight"] = float(w[r["index"]])
    return JSONResponse(_j({"elites": rows[:k], "n": c.n, "prior": prior}))


@app.get("/api/episode/{i}")
async def api_episode(i: int) -> JSONResponse:
    res = await asyncio.to_thread(STATE.replay, i)
    return JSONResponse(world_payload(res, i))


def _png(im) -> Response:
    return Response(R.to_png_bytes(im), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/episode/{i}/frame.png")
async def api_frame(i: int, camera: str = "chase", frame: int = -1, w: int = 960, h: int = 540,
                    q: int = 1, hud: int = 1) -> Response:
    res = await asyncio.to_thread(STATE.replay, i)
    w, h, q = min(max(w, 160), 1920), min(max(h, 90), 1080), 1 if q < 2 else 2
    im = await asyncio.to_thread(R.render_episode, res, camera, frame, (w, h), q, bool(hud))
    return _png(im)


@app.get("/api/episode/{i}/frame.json")
async def api_frame_json(i: int, camera: str = "chase", frame: int = -1, w: int = 960, h: int = 540,
                         q: int = 1) -> JSONResponse:
    """Provider-routed frame (local or Reactor) with provenance metadata."""
    res = await asyncio.to_thread(STATE.replay, i)
    fr = await asyncio.to_thread(STATE.provider.render, res, frame, camera, (w, h), 1 if q < 2 else 2)
    return JSONResponse(fr.to_dict())


@app.get("/api/episode/{i}/sheet.png")
async def api_sheet(i: int, n: int = 6, camera: str = "chase") -> Response:
    res = await asyncio.to_thread(STATE.replay, i)
    im = await asyncio.to_thread(R.render_contact_sheet, res, n, (640, 360), camera)
    return _png(im)


class SimBody(BaseModel):
    x: Dict[str, float]
    seed: int = 0
    gt_pose: bool = False


def _validate_x(x: Dict[str, float]) -> np.ndarray:
    missing = [d for d in DIMS if d not in x]
    if missing:
        raise HTTPException(400, f"missing dimensions: {missing}")
    return PR.from_dict(x)


@app.post("/api/simulate")
async def api_simulate(body: SimBody) -> JSONResponse:
    """What-if episode: any point in the 8-D box, deterministic in (x, seed)."""
    x = _validate_x(body.x)
    res = await asyncio.to_thread(RV.simulate_one, x, body.seed, body.gt_pose)
    return JSONResponse(world_payload(res))


def _whatif_x(k_soil: float, theta_r: float, r_terrain: float, rho_rock: float, slope_deg: float,
              sun_e: float, sun_psi: float, tau_dust: float) -> np.ndarray:
    return PR.clip_to_bounds(np.array([k_soil, theta_r, r_terrain, rho_rock, slope_deg, sun_e, sun_psi, tau_dust]))


@app.get("/api/whatif/frame.png")
async def api_whatif_frame(k_soil: float, theta_r: float, r_terrain: float, rho_rock: float, slope_deg: float,
                           sun_e: float, sun_psi: float, tau_dust: float, seed: int = 0, camera: str = "chase",
                           frame: int = -1, w: int = 960, h: int = 540, q: int = 1, hud: int = 1) -> Response:
    """GET form of the what-if render so <img src> can address it directly."""
    x = _whatif_x(k_soil, theta_r, r_terrain, rho_rock, slope_deg, sun_e, sun_psi, tau_dust)
    res = await asyncio.to_thread(RV.simulate_one, x, seed, False)
    w, h = min(max(w, 160), 1920), min(max(h, 90), 1080)
    im = await asyncio.to_thread(R.render_episode, res, camera, frame, (w, h), 1 if q < 2 else 2, bool(hud))
    return Response(R.to_png_bytes(im), media_type="image/png")


@app.get("/api/whatif/sheet.png")
async def api_whatif_sheet(k_soil: float, theta_r: float, r_terrain: float, rho_rock: float, slope_deg: float,
                           sun_e: float, sun_psi: float, tau_dust: float, seed: int = 0, camera: str = "chase",
                           n: int = 6) -> Response:
    x = _whatif_x(k_soil, theta_r, r_terrain, rho_rock, slope_deg, sun_e, sun_psi, tau_dust)
    res = await asyncio.to_thread(RV.simulate_one, x, seed, False)
    im = await asyncio.to_thread(R.render_contact_sheet, res, min(max(n, 2), 12), (640, 360), camera)
    return Response(R.to_png_bytes(im), media_type="image/png")


@app.get("/api/sensitivity")
async def api_sensitivity(prior: str = "P1") -> JSONResponse:
    if STATE.campaign.n == 0:
        return JSONResponse({})
    return JSONResponse(_j(STATE.campaign.sensitivity(_prior(prior))))


class CemBody(BaseModel):
    prior: str = "P2"
    iters: int = Field(6, ge=1, le=40)
    batch: int = Field(200, ge=20, le=2000)


@app.post("/api/cem")
async def api_cem(body: CemBody) -> JSONResponse:
    _not_busy()
    p = _prior(body.prior)
    threading.Thread(target=STATE.run_cem, args=(p, body.iters, body.batch), daemon=True).start()
    return JSONResponse({"started": True, "prior": p.name, "iters": body.iters, "batch": body.batch})


@app.get("/api/internals")
async def api_internals() -> JSONResponse:
    c = STATE.campaign
    mc = c.source == 0
    out: Dict[str, Any] = {"n": c.n, "n_mc": int(mc.sum()), "n_cem": int((c.source == 1).sum()),
                           "n_external": int((c.source == 2).sum()),
                           "busy": STATE.busy, "log": STATE.log[-60:], "cem": STATE.cem_history,
                           "min_ess": C.MIN_ESS, "priors": {}, "marginals": {}, "sample": None}
    rng = np.random.default_rng(7)
    edges = {d: np.linspace(*BOUNDS[d], 25) for d in DIMS}
    for name, p in list(PR.PRIORS.items()) + [("Q", PR.PROPOSAL)]:
        Xp = p.sample(rng, 6000)
        out["marginals"][name] = {d: np.histogram(np.clip(Xp[:, j], *BOUNDS[d]), edges[d])[0] for j, d in enumerate(DIMS)}
    out["edges"] = {d: e for d, e in edges.items()}
    if mc.any():
        fail = c.fail[mc]
        Xm = c.X[mc]
        keep = rng.choice(mc.sum(), size=min(2500, int(mc.sum())), replace=False)
        out["sample"] = {"X": Xm[keep], "fail": fail[keep].astype(int),
                         "outcome": [RV.OUTCOMES[int(o)] for o in c.outcome[mc][keep]],
                         "severity": c.metrics["severity"][mc][keep]}
        for name, p in PR.PRIORS.items():
            w = c.weights(p)[mc]
            lw = np.log10(np.maximum(w, 1e-4))
            hist, hedges = np.histogram(lw, bins=np.linspace(-4, 2, 49))
            hist = hist.astype(float) / max(int(hist.sum()), 1)
            rep = c.prior_report(p)
            out["priors"][name] = {"report": rep, "log10w_hist": hist, "log10w_edges": hedges,
                                   "ess": PR.ess(w), "max_w_frac": float(w.max() / w.sum()) if w.sum() > 0 else None,
                                   "sensitivity": c.sensitivity(p),
                                   "weighted_modes": rep["modes"]}
    return JSONResponse(_j(out))


class SaveBody(BaseModel):
    name: str = "campaign"


def _run_path(name: str) -> Path:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "campaign"
    return RUNS / f"{safe}.npz"


@app.post("/api/save")
async def api_save(body: SaveBody) -> JSONResponse:
    _not_busy()
    path = _run_path(body.name)
    safe = path.stem
    await asyncio.to_thread(STATE.campaign.save, path)
    C.write_summary_txt(STATE.campaign, RUNS / f"{safe}.txt")
    STATE.emit("log", msg=f"saved {path} ({STATE.campaign.n} episodes)")
    return JSONResponse({"saved": str(path), "n": STATE.campaign.n})


@app.post("/api/load")
async def api_load(body: SaveBody) -> JSONResponse:
    _not_busy()
    path = _run_path(body.name)
    if not path.exists():
        raise HTTPException(404, f"{path} not found")
    STATE.campaign = await asyncio.to_thread(C.Campaign.load, path)
    STATE._replays.clear()
    STATE.emit("campaign_done", summary=STATE.campaign.summary())
    return JSONResponse({"loaded": str(path), "n": STATE.campaign.n})


@app.get("/api/runs")
async def api_runs() -> JSONResponse:
    RUNS.mkdir(parents=True, exist_ok=True)
    return JSONResponse({"runs": sorted(p.stem for p in RUNS.glob("*.npz"))})


# ---------------------------------------------------------------- websocket
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    STATE.clients.append(ws)
    await ws.send_text(json.dumps({"type": "hello", "version": __version__, "n": STATE.campaign.n,
                                   "provider": STATE.provider.health(), "busy": STATE.busy}))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong", "busy": STATE.busy, "n": STATE.campaign.n}))
    except WebSocketDisconnect:
        pass
    finally:
        if ws in STATE.clients:
            STATE.clients.remove(ws)
