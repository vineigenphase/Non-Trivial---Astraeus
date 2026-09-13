# Architecture

Astraeus is a single Python package (`astraeus/`) plus an optional Isaac Sim
layer (`isaac/`). Everything in `astraeus/` runs on a laptop with NumPy,
Pillow, FastAPI and no GPU. Everything in `isaac/` is optional and talks to the
core only through a CSV schema.

```
                         disturbance vector x  (8 dims, priors.DIMS order)
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          ▼                            ▼                            ▼
   priors.py                     rover.py                     isaac/…
   P1..P6, proposal q,           simulate(X, seed)            standalone_env.py
   log-density, weights,           ├─ terrain.py              omnilrs_bridge.py
   ESS, support checks             ├─ terramechanics.py         (same x, same seed,
          │                        └─ perception.py              same CSV schema)
          │                            │                            │
          ▼                            ▼                            ▼
   campaign.py  ◄──────────── EpisodeResult ◄──────────── ingest_csv(source=2)
   MC chunks, CEM, prior_report(P_k), failure-mode shift, save/load .npz
          │
          ├───────────► cli.py           python -m astraeus run|cem|report|render|whatif|ingest|serve|selftest
          ├───────────► app.py           FastAPI + WebSocket  ──►  static/  (WebGL world model, internals)
          └───────────► render.py        NumPy ray-marched frames  ──►  providers.py (Local | Reactor)
```

## Modules

| Module | Responsibility | Depends on |
|---|---|---|
| `priors.py` | The eight dimensions, bounds, six candidate priors, the defensive mixture proposal `q`, log-pdfs, importance weights, Kish ESS, support coverage. | numpy |
| `terrain.py` | Deterministic procedural lunar surface per `(x, seed)`: multi-octave fractal relief scaled by `r_terrain`, regional slope, 4 craters, Poisson rock field with density `rho_rock`. Height, gradient, and rock queries are vectorised over episodes. | numpy |
| `terramechanics.py` | Bekker pressure–sinkage, Janosi–Hanamoto shear, compaction resistance, slip–sinkage excavation with rut relaxation, immobilisation and entrapment thresholds. `k_soil` scales `k_c, k_phi`; `theta_r` is the friction angle. | numpy |
| `perception.py` | Composite visibility `V ∈ [0,1]` from self-shadowing (`sun_e`, `r_terrain`), glare (`sun_psi` vs. heading, `sun_e`) and dust (`tau_dust`); rock detection rate and visual-odometry drift as functions of `V`. | numpy |
| `rover.py` | Episode loop (0.25 s steps, 180 s budget, 30 m waypoint) for `n` episodes in lockstep: perception → proportional heading control with reactive rock avoidance → wheel-soil mobility → attitude → outcome classification. Optional trajectory recording for replay. | terrain, terramechanics, perception |
| `campaign.py` | Accumulates episodes with a `source` tag (0 = proposal MC, 1 = CEM, 2 = external). Importance re-weights onto each `P_k`, reports failure rate with CI, ESS, max-weight share, support coverage, failure-mode distribution shift; cross-entropy method adversarial search; `.npz` save/load; CSV ingestion. | priors, rover |
| `render.py` | Deterministic software renderer: ray-marched DEM, rocks, rover body/wheels, sun-direction shading, cast shadows, dust haze, HUD; four cameras; contact sheets. | rover, perception, Pillow |
| `providers.py` | `LocalProvider` (default, credential-free) and opt-in `ReactorProvider` (reactor-sdk). Every frame carries provenance (`provider`, `live`, `reason`). | render |
| `app.py` | FastAPI REST + WebSocket. Owns one in-memory `Campaign`, streams progress, serves frames and episode JSON to the browser. | campaign, providers, fastapi |
| `static/` | Browser UI: 8 sliders, prior sampling, what-if, campaign/CEM controls, failure-mode shift tables, episode timeline, WebGL world model (`gl.js`), software-render panel, `/internals` diagnostics page. | — |
| `cli.py`, `__main__.py` | `python -m astraeus …`. | all |
| `selftest.py` | 19 invariant checks (determinism, monotonicity, weight normalisation, persistence, renderer shape). Also collected by pytest. | all |

## Data flow of one campaign

1. `Campaign.run_chunk(n)` draws `X ~ q` and seeds, calls `rover.simulate`, and stores
   `X, seed, outcome, metrics, source=0`.
2. `Campaign.prior_report(P_k)` computes `w = p_k(X)/q(X)` on `source == 0` rows
   only, self-normalises, and returns failure probability, weighted-bootstrap
   CI, ESS, ESS fraction, max normalised weight, support coverage,
   and the failure-mode distribution under `P_k` versus under `q`.
3. `Campaign.cem_search(P_k)` runs cross-entropy adversarial search seeded from the
   prior, appending elites with `source=1`. CEM rows are **never** used in
   weighted estimates; they exist for worst-case replay and failure geometry.
4. External rows from Isaac Sim / OmniLRS (`ingest_csv`) keep the `source` tag
   written by the harness (blank = `2`, external) unless the caller overrides it
   (`--source proposal` asserts i.i.d. draws from `q`; `--source external` forces 2).
5. `save()` writes one `.npz`; `load()` restores it. The web app and CLI share
   this format.

## Determinism contract

`simulate(X, seed)` is a pure function of its arguments. Terrain, rock
placement, detection events, and VO noise are all derived from `(seed, step,
slot)` hashes, so an episode found by CEM or seen in a campaign can be
re-simulated alone with `record=True` and rendered frame-by-frame, and the
Isaac exporter (`isaac/export_terrain.py`) regenerates the identical DEM and
rock list for the same `(x, seed)`.

## Web application

```
GET  /                          dashboard          GET  /internals                 diagnostics page
GET  /api/health                                   GET  /api/priors
GET  /api/priors/{name}/sample                     GET  /api/campaign
POST /api/campaign/run          {n, seed}          POST /api/campaign/reset
GET  /api/episodes              ranked list        GET  /api/episode/{i}           trajectory JSON
GET  /api/episode/{i}/frame.png                    GET  /api/episode/{i}/frame.json  (provenance)
GET  /api/episode/{i}/sheet.png                    POST /api/simulate              {x, seed, gt_pose}
POST /api/simulate/frame.png                       GET  /api/whatif/frame.png      ?k_soil=..&…
GET  /api/whatif/sheet.png                         GET  /api/sensitivity           one-at-a-time sweeps
POST /api/cem                   {prior, iters}     GET  /api/internals             weights, ESS, coverage
POST /api/save  POST /api/load  GET /api/runs      WS   /ws                        hello|log|progress|campaign_done|cem_iter|cem_done|pong
```

The browser keeps its own copy of the selected episode's trajectory and drives
the WebGL scene locally; only the ray-marched "photo" panel is fetched from the
server.
