"""Campaign engine: Monte Carlo under the proposal, re-weighting under every
candidate prior, ESS gating, bootstrap intervals, failure-mode ranking,
per-dimension sensitivity, and a cross-entropy adversarial search (a tractable
stand-in for adaptive stress testing, Lee et al. 2020) that hunts for
rare-but-plausible failures.

Everything here is pure numpy over `rover.simulate`; the FastAPI server and the
CLI both drive campaigns through `Campaign.run_chunk` so the same numbers are
produced whichever front end is used.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from . import priors as PR
from . import rover as RV
from .priors import DIMS, N_DIMS, Prior

MIN_ESS = 30.0            # below this a prior's estimate is reported as unreliable
BOOT = 400                # bootstrap replicates for the interval on P(fail)


# --------------------------------------------------------------------------
# weighted statistics
# --------------------------------------------------------------------------
def weighted_rate(fail: np.ndarray, w: np.ndarray) -> float:
    s = float(w.sum())
    return float((w * fail).sum() / s) if s > 0 else float("nan")


def bootstrap_ci(fail: np.ndarray, w: np.ndarray, rng: np.random.Generator,
                 reps: int = BOOT, alpha: float = 0.05) -> tuple:
    """Percentile bootstrap of the self-normalised weighted rate."""
    n = len(fail)
    if n == 0 or w.sum() <= 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(reps, n))
    wf = (w * fail)[idx].sum(1)
    ws = w[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(ws > 0, wf / ws, np.nan)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(r, 100 * alpha / 2)), float(np.percentile(r, 100 * (1 - alpha / 2))))


def mode_breakdown(outcome: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """Weighted probability of each terminal outcome (sums to 1)."""
    s = float(w.sum())
    out: Dict[str, float] = {}
    for code, name in enumerate(RV.OUTCOMES):
        out[name] = float((w * (outcome == code)).sum() / s) if s > 0 else float("nan")
    return out


def dimension_sensitivity(X: np.ndarray, fail: np.ndarray, w: np.ndarray, bins: int = 6) -> Dict[str, dict]:
    """Weighted P(fail | dimension in bin) across each dimension's support —
    the one-at-a-time view a mission engineer asks for first."""
    res: Dict[str, dict] = {}
    for j, d in enumerate(DIMS):
        lo, hi = PR.BOUNDS[d]
        edges = np.linspace(lo, hi, bins + 1)
        b = np.clip(np.digitize(X[:, j], edges) - 1, 0, bins - 1)
        rate, cnt = [], []
        for k in range(bins):
            m = b == k
            ws = float(w[m].sum())
            rate.append(float((w[m] * fail[m]).sum() / ws) if ws > 0 else None)
            cnt.append(int(m.sum()))
        centres = 0.5 * (edges[:-1] + edges[1:])
        res[d] = {"centres": centres.tolist(), "rate": rate, "count": cnt, "unit": PR.UNITS[d]}
    return res


_COVERAGE_CACHE: Dict[str, float] = {}


def support_coverage(prior: Prior, n: int = 20000) -> float:
    """Fraction of the prior's own mass that lies where the proposal can sample.
    Anything below 1 means part of what this prior believes is *never*
    simulated (P3's 45 deg sun): importance weights cannot recover it."""
    if prior.name not in _COVERAGE_CACHE:
        Xp = prior.sample(np.random.default_rng(12345), n)
        inside = np.ones(n, bool)
        for j, d in enumerate(DIMS):
            lo, hi = PR.BOUNDS[d]
            inside &= (Xp[:, j] >= lo) & (Xp[:, j] <= hi)
        _COVERAGE_CACHE[prior.name] = float(inside.mean())
    return _COVERAGE_CACHE[prior.name]


# --------------------------------------------------------------------------
# campaign state
# --------------------------------------------------------------------------
@dataclass
class Campaign:
    """Accumulates episodes from the proposal q and answers questions under
    every candidate prior p_k without re-simulating."""
    seed: int = 0
    gt_pose: bool = False
    X: np.ndarray = field(default_factory=lambda: np.zeros((0, N_DIMS)))
    seeds: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    outcome: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    metrics: Dict[str, np.ndarray] = field(default_factory=dict)
    elapsed_s: float = 0.0
    source: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # 0 = MC, 1 = CEM

    METRIC_KEYS = ("t_end", "final_dist", "max_tilt_deg", "max_slip", "max_sinkage",
                   "min_visibility", "mean_visibility", "rocks_total", "rocks_detected",
                   "vo_error", "severity")

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._boot_rng = np.random.default_rng(self.seed + 1_000_003)
        self._next_seed = self.seed * 1_000_000

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def fail(self) -> np.ndarray:
        return self.outcome != RV.OUTCOME_CODE["success"]

    # ---------------------------------------------------------------- append
    def _append(self, res: RV.EpisodeResult, source: int) -> None:
        self.X = np.vstack([self.X, res.X])
        self.seeds = np.concatenate([self.seeds, res.seed])
        self.outcome = np.concatenate([self.outcome, res.outcome])
        self.source = np.concatenate([self.source, np.full(len(res.seed), source, np.int64)])
        for k in self.METRIC_KEYS:
            v = np.asarray(getattr(res, k), float)
            self.metrics[k] = np.concatenate([self.metrics.get(k, np.zeros(0)), v])

    def _take_seeds(self, n: int) -> np.ndarray:
        s = np.arange(self._next_seed, self._next_seed + n, dtype=np.int64)
        self._next_seed += n
        return s

    def run_chunk(self, n: int, progress: Optional[Callable] = None) -> RV.EpisodeResult:
        """Simulate n fresh episodes drawn from the proposal and fold them in."""
        X = PR.sample_proposal(self._rng, n)
        t0 = time.perf_counter()
        res = RV.simulate(X, self._take_seeds(n), gt_pose=self.gt_pose, progress=progress)
        self.elapsed_s += time.perf_counter() - t0
        self._append(res, source=0)
        return res

    # --------------------------------------------------------------- weights
    def weights(self, prior: Prior) -> np.ndarray:
        """Importance weights for the MC episodes only (CEM samples are not
        drawn from q and must never enter a prior estimate)."""
        mc = self.source == 0
        w = np.zeros(self.n)
        if mc.any():
            w[mc] = PR.importance_weights(self.X[mc], prior)
        return w

    def prior_report(self, prior: Prior) -> Dict[str, object]:
        w = self.weights(prior)
        mc = self.source == 0
        cov = support_coverage(prior)
        if mc.sum() == 0:
            return {"prior": prior.name, "label": prior.label, "n": 0, "ess": 0.0,
                    "reliable": False, "support_coverage": round(cov, 4), "p_fail": None,
                    "ci": [None, None], "modes": {}, "top_mode": None, "max_weight_frac": None}
        fail = self.fail.astype(float)
        e = PR.ess(w[mc])
        lo, hi = bootstrap_ci(fail[mc], w[mc], self._boot_rng)
        modes = mode_breakdown(self.outcome[mc], w[mc])
        top = sorted(((m, p) for m, p in modes.items() if m != "success"), key=lambda t: -t[1])
        return {
            "prior": prior.name, "label": prior.label, "n": int(mc.sum()),
            "ess": round(e, 1), "ess_frac": round(e / max(int(mc.sum()), 1), 3),
            "reliable": bool(e >= MIN_ESS and cov >= 0.5),
            "support_coverage": round(cov, 4),
            "p_fail": round(weighted_rate(fail[mc], w[mc]), 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "modes": {m: round(p, 4) for m, p in modes.items()},
            "top_mode": top[0][0] if top else None,
            "max_weight_frac": round(float(w[mc].max() / w[mc].sum()), 4) if w[mc].sum() > 0 else None,
        }

    def all_priors(self) -> List[Dict[str, object]]:
        return [self.prior_report(p) for p in PR.PRIORS.values()]

    def shift_matrix(self) -> Dict[str, object]:
        """How the failure-mode ranking moves across priors — the
        distributional-shift picture in one table."""
        reports = self.all_priors()
        modes = [m for m in RV.OUTCOMES if m != "success"]
        rows = []
        for r in reports:
            rows.append({"prior": r["prior"], "label": r["label"], "p_fail": r["p_fail"],
                         "ess": r["ess"], "reliable": r["reliable"],
                         "ranking": sorted(modes, key=lambda m: -(r["modes"].get(m) or 0.0)),
                         "modes": {m: r["modes"].get(m) for m in modes}})
        # does the top failure mode change between priors?
        tops = {r["prior"]: r["ranking"][0] for r, rep in zip(rows, reports)
                if r["p_fail"] is not None and rep["support_coverage"] >= 0.5}
        return {"rows": rows, "top_modes": tops, "ranking_shifts": len(set(tops.values())) > 1}

    def sensitivity(self, prior: Prior) -> Dict[str, dict]:
        mc = self.source == 0
        return dimension_sensitivity(self.X[mc], self.fail[mc].astype(float), self.weights(prior)[mc])

    # ---------------------------------------------------------------- elites
    def elites(self, k: int = 8, prior: Optional[Prior] = None) -> List[int]:
        """Indices of the most severe failures. With a prior, severity is
        weighted by plausibility under that prior so an elite is a failure the
        mission might actually meet, not just the nastiest corner of q."""
        sev = self.metrics["severity"].copy()
        sev[~self.fail] = -np.inf
        if prior is not None:
            w = PR.importance_weights(self.X, prior)
            sev = sev + np.log(np.maximum(w, 1e-12))
        order = np.argsort(-sev)
        return [int(i) for i in order[:k] if np.isfinite(sev[i])]

    def episode(self, i: int) -> Dict[str, object]:
        d = {k: float(v) for k, v in zip(DIMS, self.X[i])}
        d.update({"index": int(i), "seed": int(self.seeds[i]), "outcome": RV.OUTCOMES[int(self.outcome[i])],
                  "source": "cem" if self.source[i] == 1 else "mc"})
        for k in self.METRIC_KEYS:
            d[k] = float(self.metrics[k][i])
        return d

    def replay(self, i: int) -> RV.EpisodeResult:
        """Bit-for-bit re-simulation of episode i with the trajectory recorded."""
        return RV.simulate_one(self.X[i], int(self.seeds[i]), gt_pose=self.gt_pose)

    # ------------------------------------------------------------------- CEM
    def cem_search(self, prior: Prior, iters: int = 6, batch: int = 200, elite_frac: float = 0.15,
                   smoothing: float = 0.6, progress: Optional[Callable] = None) -> Iterator[Dict[str, object]]:
        """Cross-entropy search for high-severity, plausible failures.

        Objective J(x) = severity(x) + λ·log p(x): severity comes from the full
        simulation, log-plausibility from the target prior, so the search
        cannot wander into corners the prior says never happen. The sampling
        distribution is a truncated Gaussian in the 8-D box, moment-matched to
        the elites each iteration and smoothed to avoid collapse. Yields one
        dict per iteration for streaming."""
        lo = np.array([PR.BOUNDS[d][0] for d in DIMS])
        hi = np.array([PR.BOUNDS[d][1] for d in DIMS])
        mu = prior.sample(self._rng, 4000).mean(0)
        sig = (hi - lo) / 4.0
        lam = 0.25
        best: Optional[Dict[str, object]] = None
        for it in range(iters):
            Xc = PR.clip_to_bounds(self._rng.normal(mu, sig, size=(batch, N_DIMS)))
            Xc[:, DIMS.index("sun_psi")] = np.mod(Xc[:, DIMS.index("sun_psi")], 360.0)
            t0 = time.perf_counter()
            res = RV.simulate(Xc, self._take_seeds(batch), gt_pose=self.gt_pose, progress=progress)
            self.elapsed_s += time.perf_counter() - t0
            logp = prior.log_pdf(Xc)
            J = res.severity + lam * np.where(np.isfinite(logp), logp, -50.0)
            order = np.argsort(-J)
            n_el = max(4, int(elite_frac * batch))
            el = order[:n_el]
            mu = smoothing * Xc[el].mean(0) + (1 - smoothing) * mu
            sig = smoothing * np.maximum(Xc[el].std(0), 0.02 * (hi - lo)) + (1 - smoothing) * sig
            start = self.n
            self._append(res, source=1)
            i_best = int(order[0])
            cand = {"index": start + i_best, "J": float(J[i_best]), "severity": float(res.severity[i_best]),
                    "outcome": RV.OUTCOMES[int(res.outcome[i_best])], "x": PR.to_dict(Xc[i_best])}
            if best is None or cand["J"] > best["J"]:
                best = cand
            yield {
                "iter": it + 1, "iters": iters, "prior": prior.name,
                "fail_rate": float(res.fail.mean()),
                "mu": PR.to_dict(mu), "sigma": PR.to_dict(sig),
                "elite_mean_severity": float(res.severity[el].mean()),
                "best": best,
                "modes": {m: int((res.outcome == RV.OUTCOME_CODE[m]).sum()) for m in RV.OUTCOMES},
            }

    # ----------------------------------------------------------- persistence
    def to_rows(self) -> List[Dict[str, object]]:
        return [self.episode(i) for i in range(self.n)]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=self.X, seeds=self.seeds, outcome=self.outcome, source=self.source,
                            elapsed_s=self.elapsed_s, seed=self.seed, gt_pose=self.gt_pose,
                            **{"m_" + k: v for k, v in self.metrics.items()})

    @classmethod
    def load(cls, path: Path) -> "Campaign":
        z = np.load(Path(path), allow_pickle=False)
        c = cls(seed=int(z["seed"]), gt_pose=bool(z["gt_pose"]))
        c.X, c.seeds, c.outcome, c.source = z["X"], z["seeds"], z["outcome"], z["source"]
        c.elapsed_s = float(z["elapsed_s"])
        c.metrics = {k[2:]: z[k] for k in z.files if k.startswith("m_")}
        c._next_seed = int(c.seeds.max()) + 1 if c.n else c._next_seed
        return c

    def summary(self) -> Dict[str, object]:
        mc = self.source == 0
        return {
            "n_total": self.n, "n_mc": int(mc.sum()), "n_cem": int((~mc).sum()),
            "gt_pose": self.gt_pose, "elapsed_s": round(self.elapsed_s, 2),
            "episodes_per_s": round(self.n / self.elapsed_s, 1) if self.elapsed_s > 0 else None,
            "raw_fail_rate_q": round(float(self.fail[mc].mean()), 4) if mc.any() else None,
            "priors": self.all_priors(),
            "shift": self.shift_matrix(),
        }


def write_summary_txt(c: Campaign, path: Path) -> None:
    """Human-readable report in the spirit of the v1 analyse.py summary."""
    lines = [f"Astraeus campaign — {c.n} episodes ({int((c.source == 0).sum())} MC, "
             f"{int((c.source == 1).sum())} CEM), gt_pose={c.gt_pose}, {c.elapsed_s:.1f} s sim", ""]
    for r in c.all_priors():
        if r["support_coverage"] < 0.5:
            flag = "   [%.0f%% of this prior lies outside the simulated support — not estimable]" % (
                100 * (1 - r["support_coverage"]))
        else:
            flag = "" if r["reliable"] else "   [ESS < %d — unreliable]" % MIN_ESS
        lines.append(f"{r['prior']:>3} {r['label']:<34} P(fail)={r['p_fail']:.3f} "
                     f"CI95=[{r['ci'][0]:.3f},{r['ci'][1]:.3f}] ESS={r['ess']:.0f}/{r['n']}{flag}")
        modes = ", ".join(f"{m}={p:.3f}" for m, p in r["modes"].items() if m != "success")
        lines.append(f"      modes: {modes}")
    sh = c.shift_matrix()
    lines += ["", "Top failure mode by prior: " + json.dumps(sh["top_modes"]),
              "Ranking shifts across priors: " + ("YES" if sh["ranking_shifts"] else "no")]
    Path(path).write_text("\n".join(lines) + "\n")
