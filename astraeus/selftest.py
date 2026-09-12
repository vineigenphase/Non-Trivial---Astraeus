"""Built-in invariant checks. `python -m astraeus selftest` (or via pytest in
tests/). Every check is a property that must hold for the framework to be
trusted; none of them is a tuning target.
"""
from __future__ import annotations

import io
import sys
import time
from typing import Callable, List, Tuple

import numpy as np

from . import campaign as C
from . import perception as P
from . import priors as PR
from . import render as RD
from . import rover as RV
from . import terramechanics as TM
from . import terrain as TR

CHECKS: List[Tuple[str, Callable[[], None]]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ---------------------------------------------------------------- priors
@check("every prior's pdf integrates to ~1 on the box (MC)")
def _pdf_normalised():
    rng = np.random.default_rng(0)
    vol = float(np.prod([hi - lo for lo, hi in PR.BOUNDS.values()]))
    U = np.column_stack([rng.uniform(lo, hi, 200_000) for lo, hi in PR.BOUNDS.values()])
    for p in PR.PRIORS.values():
        mass = vol * p.pdf(U).mean()
        cov = C.support_coverage(p)
        assert abs(mass - cov) < 0.08, f"{p.name}: mass on box {mass:.3f} vs coverage {cov:.3f}"


@check("proposal covers every prior with mass on the box (finite weights)")
def _proposal_covers():
    rng = np.random.default_rng(1)
    X = PR.sample_proposal(rng, 5000)
    assert np.all(PR.PROPOSAL.pdf(X) > 0)
    for p in PR.PRIORS.values():
        w = PR.importance_weights(X, p)
        assert np.all(np.isfinite(w)) and w.min() >= 0
        if C.support_coverage(p) > 0.5:
            assert abs(w.mean() - 1) < 0.25, f"{p.name}: mean weight {w.mean():.3f}"


@check("P3 (simulator default) is flagged as outside the polar support")
def _p3_out_of_support():
    assert C.support_coverage(PR.P3) < 0.05


@check("ESS is Kish: uniform weights give n, one-hot gives 1")
def _ess():
    assert abs(PR.ess(np.ones(100)) - 100) < 1e-9
    assert abs(PR.ess(np.eye(1, 100)[0]) - 1) < 1e-9


# ---------------------------------------------------------------- terrain
@check("terrain is deterministic in (seed, x) and differs across seeds")
def _terrain_det():
    X = np.tile(np.array([[2.0, 0.72, 1.0, 3.0, 5.0, 1.5, 200.0, 0.05]]), (2, 1))
    a = TR.TerrainBatch.build(np.array([7, 7]), X[:, 2], X[:, 4], X[:, 3])
    b = TR.TerrainBatch.build(np.array([7, 8]), X[:, 2], X[:, 4], X[:, 3])
    xs, ys = np.array([3.0, 3.0]), np.array([1.0, 1.0])
    ha, hb = a.height(xs, ys), b.height(xs, ys)
    assert ha[0] == ha[1] == hb[0] and hb[1] != hb[0]


@check("regional slope drops height along the downhill direction by tan(slope)")
def _slope():
    t = TR.TerrainBatch.build(np.array([3]), np.array([0.0]), np.array([10.0]), np.array([0.0]))
    d = float(t.slope_dir[0])
    p0 = np.array([-50.0, 50.0])                       # far from the crater corridor
    p1 = p0 + 20.0 * np.array([np.cos(d), np.sin(d)])
    h = t.height(np.array([[p0[0], p1[0]]]), np.array([[p0[1], p1[1]]]), rocks=False)[0]
    assert abs((h[0] - h[1]) / 20.0 - np.tan(np.radians(10.0))) < 1e-3, h


@check("rock count scales with rho_rock")
def _rocks():
    n = 400
    for rho in (0.0, 2.0, 10.0):
        t = TR.TerrainBatch.build(np.arange(n), np.ones(n), np.zeros(n), np.full(n, rho))
        m, lam = float(t.rock_count.mean()), rho * TR.ROCK_AREA_M2 / 100.0
        assert abs(m - min(lam, TR.N_ROCK_SLOTS)) <= 0.15 * lam + 0.5, f"rho={rho}: mean rocks {m:.1f} vs {lam:.1f}"


# ---------------------------------------------------------------- terramechanics
@check("Bekker sinkage decreases with soil stiffness and is O(cm) at nominal")
def _sinkage():
    W = np.full(3, RV.MASS_KG * TM.G_MOON / TM.WHEEL.count)
    z = TM.static_sinkage(W, np.array([0.5, 1.0, 5.0]))
    assert z[0] > z[1] > z[2] and 0.003 < z[1] < 0.06, z


@check("slip rises with demanded thrust and immobilises when demand exceeds H_max")
def _slip():
    L = np.full(4, 0.15)
    H = np.ones(4)
    slip, imm = TM.required_slip(np.array([0.1, 0.4, 0.8, 1.2]), H, L)
    assert np.all(np.diff(slip) > 0) and not imm[:3].any() and imm[3] and slip[3] == 1.0


@check("steep uphill with weak soil immobilises; flat firm soil does not")
def _step():
    out = TM.step_wheel_soil(RV.MASS_KG, np.radians(np.array([0.0, 28.0])), np.array([3.0, 0.5]),
                             np.array([0.75, 0.62]), np.zeros(2), np.full(2, 0.5), 0.25, np.zeros(2))
    assert out["slip"][0] < 0.2 and out["v"][0] > 0.4
    assert out["immobilised"][1] or out["slip"][1] > 0.6


# ---------------------------------------------------------------- perception
@check("visibility falls with dust and rises with sun elevation away from glare")
def _vis():
    two = lambda a: np.array([a, a])
    v = P.visibility(two(1.5), two(200.0), np.array([0.0, 0.6]), two(1.0), two(0.0))
    assert v[0] > v[1], v
    glare = P.visibility(two(1.5), np.array([0.0, 180.0]), two(0.0), two(1.0), two(0.0))
    assert glare[0] < glare[1], glare  # sun dead ahead (azimuth 0 = +x = heading) blinds


# ---------------------------------------------------------------- rover
@check("episodes are deterministic in (x, seed)")
def _det():
    x = np.array([2.0, 0.72, 1.0, 3.0, 5.0, 1.5, 200.0, 0.05])
    a, b = RV.simulate_one(x, 11), RV.simulate_one(x, 11)
    assert a.outcome[0] == b.outcome[0] and np.allclose(a.trace["x"], b.trace["x"])


@check("nominal polar episode succeeds; extreme slope+soft soil fails")
def _nominal():
    ok = RV.simulate_one(np.array([2.0, 0.724, 1.0, 2.0, 4.0, 1.5, 200.0, 0.05]), 3)
    assert RV.OUTCOMES[int(ok.outcome[0])] == "success", ok.row(0)
    bad = RV.simulate(np.tile([[0.5, 0.61, 2.0, 12.0, 15.0, 0.5, 200.0, 0.6]], (8, 1)), np.arange(8))
    assert bad.fail.mean() >= 0.75, bad.outcome_names()


@check("batch and single-episode paths agree")
def _batch_agree():
    rng = np.random.default_rng(5)
    X = PR.sample_proposal(rng, 6)
    seeds = np.arange(100, 106)
    batch = RV.simulate(X, seeds)
    for i in range(6):
        one = RV.simulate_one(X[i], int(seeds[i]))
        assert one.outcome[0] == batch.outcome[i], (i, one.outcome_names(), batch.outcome_names())
        assert abs(one.t_end[0] - batch.t_end[i]) < 1e-6


@check("failure monotonicity: harder priors (P5) fail more than easier (P4)")
def _mono():
    rng = np.random.default_rng(9)
    r4 = RV.simulate(PR.clip_to_bounds(PR.P4.sample(rng, 300)), np.arange(300)).fail.mean()
    r5 = RV.simulate(PR.clip_to_bounds(PR.P5.sample(rng, 300)), np.arange(300)).fail.mean()
    assert r5 > r4 + 0.1, (r4, r5)


# ---------------------------------------------------------------- campaign
@check("campaign re-weighting matches direct sampling within CI (P2, P4)")
def _reweight():
    c = C.Campaign(seed=4)
    c.run_chunk(1500)
    rng = np.random.default_rng(77)
    for name in ("P2", "P4"):
        p = PR.PRIORS[name]
        rep = c.prior_report(p)
        direct = RV.simulate(PR.clip_to_bounds(p.sample(rng, 800)), np.arange(5000, 5800)).fail.mean()
        lo, hi = rep["ci"]
        assert rep["reliable"] and lo - 0.08 <= direct <= hi + 0.08, (name, rep["p_fail"], rep["ci"], direct)


@check("CEM samples never enter weighted estimates; replay is exact")
def _cem_replay():
    c = C.Campaign(seed=2)
    c.run_chunk(200)
    before = c.prior_report(PR.P2)["p_fail"]
    list(c.cem_search(PR.P2, iters=2, batch=50))
    assert c.n == 300 and abs(c.prior_report(PR.P2)["p_fail"] - before) < 1e-12
    i = c.elites(k=1, prior=PR.P2)[0]
    r = c.replay(i)
    assert r.outcome[0] == c.outcome[i] and abs(r.t_end[0] - c.metrics["t_end"][i]) < 1e-6


@check("save/load round-trips and ingest accepts the CSV row format")
def _persist():
    import tempfile
    from pathlib import Path
    c = C.Campaign(seed=8)
    c.run_chunk(60)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.npz"
        c.save(p)
        c2 = C.Campaign.load(p)
        assert c2.n == 60 and np.allclose(c2.X, c.X) and np.all(c2.outcome == c.outcome)
        rows = c.to_rows()[:10]
        c3 = C.Campaign(seed=1)
        assert c3.ingest_rows(rows) == 10 and c3.n == 10
        assert np.allclose(c3.metrics["severity"], c.metrics["severity"][:10])


# ---------------------------------------------------------------- render
@check("software renderer produces a non-trivial PNG for every camera")
def _render():
    res = RV.simulate_one(np.array([2.0, 0.72, 1.0, 4.0, 5.0, 1.5, 200.0, 0.1]), 21)
    for cam in RD.CAMERAS:
        im = RD.render_episode(res, camera=cam, size=(320, 180), quality=3)
        a = np.asarray(im)
        assert a.shape == (180, 320, 3) and a.std() > 10, cam
    png = RD.to_png_bytes(im)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 2000


def main(verbose: bool = True) -> int:
    fails = 0
    t_all = time.perf_counter()
    for name, fn in CHECKS:
        t0 = time.perf_counter()
        try:
            fn()
            status = "ok  "
        except AssertionError as e:
            status = "FAIL"
            fails += 1
            name += f"  -> {e}"
        except Exception as e:  # noqa: BLE001 - report, don't hide
            status = "ERR "
            fails += 1
            name += f"  -> {type(e).__name__}: {e}"
        if verbose:
            print(f"[{status}] {name}  ({time.perf_counter() - t0:.1f}s)")
    if verbose:
        print(f"{len(CHECKS) - fails}/{len(CHECKS)} checks passed in {time.perf_counter() - t_all:.1f}s")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
