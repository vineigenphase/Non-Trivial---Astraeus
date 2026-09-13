# Validation

What was run, what came out, and — as importantly — what was **not** validated. All numbers below
were produced by the commands shown, on the commit that introduced this document, with the NumPy
simulator (no Isaac Sim, no ROS 2, no Reactor key). Re-running the commands reproduces them exactly:
every episode is deterministic in `(x, seed)` and the campaign seed is fixed.

## 1. Automated checks

| Check | Command | Result |
|-------|---------|--------|
| Byte-compile | `python -m compileall -q astraeus isaac tests legacy` | clean |
| Built-in invariants (19) | `python -m astraeus selftest` | all pass |
| pytest (19 invariants + 11 Isaac-helper tests) | `python -m pytest tests -q` | 30 passed |
| Lint | `ruff check --select E,F,W --line-length 130 astraeus isaac tests` | clean |

The 19 invariants (`astraeus/selftest.py`) check that: every prior's PDF integrates to ≈1 on the
box; the proposal covers every prior with finite weights; P3 is flagged outside the polar support;
ESS is Kish's (uniform weights → n, one-hot → 1); terrain is deterministic in `(seed, x)` and differs
across seeds; regional slope drops height by `tan(slope)` downhill; rock count scales with `rho_rock`;
Bekker sinkage decreases with soil stiffness and is O(cm) at nominal; slip rises with demanded thrust
and immobilises past `H_max`; steep uphill on weak soil immobilises while flat firm soil does not;
visibility falls with dust and rises with sun elevation away from glare; episodes are deterministic in
`(x, seed)`; the nominal polar episode succeeds while extreme slope + soft soil fails; batch and
single-episode paths agree; P5 fails more than P4; campaign re-weighting matches direct sampling from
P2 and P4 within the CI; CEM samples never enter weighted estimates and replay is exact; save/load
round-trips and CSV ingest preserves source provenance; the renderer produces a non-trivial PNG from
every camera, including at a campaign-scale seed (`20260912001568`, the regression case for the
integer-overflow bug fixed in `render._stars`).

The Isaac-helper tests exercise `isaac/astraeus_isaac/` without Isaac Sim: `x → scene` mapping, sun
vector and rotation, each outcome detector in `EpisodeMonitor`, the controller, durable CSV logging
followed by `Campaign.ingest_csv`, terrain export, byte-compilation of every Isaac entry point, and
offline topic discovery.

## 2. Validation campaign

```bash
python -m astraeus run --n 4000 --seed 20260912 --out runs/val.npz
python -m astraeus cem runs/val.npz --prior P2      # 6 x 200
python -m astraeus cem runs/val.npz --prior P5      # 6 x 200
python -m astraeus report runs/val.npz
```

Simulation throughput: **~460–660 episodes/s** on one CPU core (4000 episodes in 13.8 s including
report); raw failure rate under the proposal `q`: **0.443**.

### 2.1 Per-prior estimates (4000 proposal episodes)

| Prior | P(fail) | 95 % bootstrap CI | ESS | Reliable | Mode ranking |
|------:|--------:|------------------:|----:|:--------:|--------------|
| P1 max ignorance         | 0.550 | 0.530 – 0.573 | 1794 | yes | stuck › collision › nav_miss › tip_over |
| P2 VIPER-spec south pole | 0.331 | 0.294 – 0.364 |  687 | yes | stuck › collision › nav_miss › tip_over |
| P3 simulator default     | 0.000 | —             |    1 | **no** — 100 % of mass outside simulated support | (undefined) |
| P4 equatorial mare       | 0.084 | 0.064 – 0.108 |  583 | yes | stuck › collision › nav_miss |
| P5 PSR rim, deep winter  | 0.665 | 0.629 – 0.697 |  675 | yes | stuck › collision › **tip_over** › nav_miss |
| P6 post-landing plume    | 0.379 | 0.347 – 0.417 |  667 | yes | **nav_miss** › collision › stuck › tip_over |

Weighted mode probabilities:

| Prior | stuck | tip_over | collision | nav_miss | timeout |
|------:|------:|---------:|----------:|---------:|--------:|
| P1 | 0.263 | 0.048 | 0.173 | 0.066 | 0.000 |
| P2 | 0.137 | 0.015 | 0.127 | 0.051 | 0.001 |
| P4 | 0.043 | 0.000 | 0.038 | 0.004 | 0.000 |
| P5 | 0.287 | 0.072 | 0.249 | 0.056 | 0.002 |
| P6 | 0.088 | 0.011 | 0.131 | 0.149 | 0.000 |

### 2.2 What the campaign shows

- **The answer depends on the prior by a factor of 8** (P4 0.084 vs P5 0.665) for the same rover,
  the same controller and the same 4000 episodes. This is the point of the framework.
- **The dominant failure mode shifts.** Every terrain-driven prior ranks `stuck` first; the dust
  prior P6 moves `nav_miss` to the top (0.149) because visibility loss degrades detection and VO
  before it degrades mobility. P5 promotes `tip_over` above `nav_miss`. `ranking_shifts: true` in
  the JSON report.
- **P3 is not estimable, by design.** Its 45° sun elevation has no mass on the simulated support
  `[0.5°, 6°]`, so ESS collapses to 1 and the reliability gate fires. The framework refuses to report
  a failure rate for it rather than printing a misleading 0.000.
- **ESS is bounded below by the mixture weight**, as the defensive-mixture argument predicts:
  P2/P4/P5/P6 each sit at roughly `0.15 × 4000 = 600` ± the overlap with P1.

### 2.3 CEM adversarial search

| Prior | Iter 1 → 6 fail rate in batch | Best J | Worst-case elite |
|-------|------------------------------:|-------:|------------------|
| P2 | 0.36 → 0.33 | 0.763 | #5181 `stuck`: k_soil 2.0, theta_r 0.675, slope 6.9°, sun_e 3.6° |
| P5 | — → 0.76    | 1.237 | #6399 `stuck`: k_soil 1.2, theta_r 0.622, r_terrain 1.76, rho_rock 5.3, slope 10.2°, sun_e 1.1° |

Both searches converge on the soft-soil / steep-slope / low-friction corner; under P5 the batch
failure rate reaches 0.76. CEM rows are stored with `source=1` and never enter the tables in §2.1.

### 2.4 Renders

Generated from the same run by `python scripts/make_renders.py runs/val.npz --prior P1 --quality 3`
(the most plausible elite under P1 per outcome; first success by index) and stored in `docs/img/`:

| File | Episode | Seed | Note |
|------|---------|------|------|
| `stuck_chase.png`         | #5523 stuck (CEM/P5 row) | 20260912005523 | slip 1.00, pitch/roll 18°/22°, sinkage 7.4 cm, visibility 0.16 at τ=0.44, sun 0.5° |
| `stuck_sheet.png`         | #5523, 6 frames          | 20260912005523 | contact sheet of the traverse into entrapment |
| `tip_over_chase.png`      | #5277 tip_over           | 20260912005277 | attitude beyond 30° |
| `collision_rover_cam.png` | #5584 collision          | 20260912005584 | rover-camera view at impact |
| `nav_miss_overview.png`   | #758 nav_miss            | 20260912000758 | true path vs VO estimate diverging |
| `success_orbit.png`       | #0 success               | 20260912000000 | orbit camera |

Episode indices ≥ 4000 are CEM rows (`source=1`); they are legitimate replayable failures but are not
part of the estimates in §2.1.

The README showcase assets come from the same run via
`python scripts/make_showcase.py runs/val.npz --hero-episode 111` (2× supersampled, clean caption
instead of the HUD). Selection is photogenic (low but not grazing sun, light dust, visible rocks), not
statistical:

| File | Episode | Seed | Camera |
|------|---------|------|--------|
| `hero.png`, `gallery_stuck.png`, `demo.webp` | #111 stuck (proposal row) | 20260912000111 | cinematic / cinematic / chase, 40 frames |
| `gallery_tip_over.png`  | #675 tip_over (proposal row)   | 20260912000675 | chase |
| `gallery_collision.png` | #4363 collision (CEM row)      | 20260912004363 | rover_cam |
| `gallery_nav_miss.png`  | #4686 nav_miss (CEM row)       | 20260912004686 | overview |
| `gallery_success.png`   | #18 success (proposal row)     | 20260912000018 | cinematic |

## 3. Web application

Manually exercised on the same commit with `uvicorn astraeus.app:app --port 8000`:
`/`, `/internals`, `/api/health`, `/api/priors`, `/api/campaign`, `/api/campaign/run`,
`/api/episodes?prior=P2`, `/api/episode/{i}`, `/api/episode/{i}/frame.png`, `/api/episode/{i}/sheet.png`,
`/api/simulate`, `/api/internals`, `/api/save`, `/api/load`, and `/ws` progress messages during a
campaign run and a CEM search. Local provider only (`ASTRAEUS_PROVIDER` unset); `/api/health` reports
`provider: local`, no Reactor credentials present.

## 4. Not validated — read before relying on anything above

| Item | Status |
|------|--------|
| **NVIDIA Isaac Sim execution** | `isaac/standalone_env.py` compiles and its Isaac-free helpers are tested; the Isaac API calls have **not** been executed. Isaac Sim is not installed in the validation environment. |
| **OmniLRS / ROS 2** | `isaac/omnilrs_bridge.py` compiles; topic names/types in `isaac/configs/ros2_topics.yaml` are marked `verified: false`. Run `isaac/discover_topics.py` against a live sim first. |
| **Terramechanics in Isaac** | Rigid PhysX contact does not reproduce Bekker sinkage. Isaac episodes will under-report `stuck`. Documented gap; hook provided. |
| **Reactor provider** | Written against reactor-sdk 1.5 and probes the model schema at runtime; **no frames were generated** because no API key was used. Fallback to the local renderer was exercised. |
| **Absolute failure rates** | Properties of this reduced-order simulator and its simple controller. Use the *comparison across priors* and the *replayable mechanisms*, not the absolute numbers, for anything mission-related. |
| **Priors P2/P4/P5/P6** | Engineering-judgement marginals, not fitted to site data. |
| **Perception** | Parametric model of visibility, detection and VO drift; no image processing. |
| **Renderer radiometry** | Physically motivated, not calibrated. |
| **Browser UI across browsers** | Exercised in Chromium only. |

## 5. Reproducing

```bash
make selftest test lint
make campaign          # runs/val.npz + report
make renders           # docs/img/*.png
```

Changing `--seed` changes every number; changing a prior's marginals changes only its column
(no re-simulation is needed — `python -m astraeus report runs/val.npz` re-weights the saved run).
