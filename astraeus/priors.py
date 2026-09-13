"""Disturbance space and priors (table v2.0 — eight dimensions).

Astraeus stress-tests a rover under a disturbance vector

    x = (k_soil, theta_r, r_terrain, rho_rock, slope_deg, sun_e, sun_psi, tau_dust)

whose true joint distribution on the target site is *unknown*. The framework
therefore never commits to one prior. It samples once from a wide proposal
``q`` and re-weights the same episodes onto every candidate prior ``P_k``
afterwards (importance sampling, ``w_k(x) = p_k(x) / q(x)``), reporting the
effective sample size next to every estimate so a degenerate re-weighting is
visible instead of silently wrong.

Every prior is a product of independent per-dimension marginals (table v2.0
assumes independence; the correlation structure between soil and lighting on a
real site is one of the open questions listed in docs/RESEARCH.md).

Dimensions
----------
k_soil      multiplier on Bekker sinkage moduli (k_c, k_phi); 1.0 = Lunar
            Sourcebook nominal regolith. Log-uniform under P1: an order of
            magnitude of ignorance either side of nominal is honest.
theta_r     soil internal friction angle [rad]; sets shear strength and the
            angle of repose of loose regolith.
r_terrain   relief scale multiplier on the fractal heightfield (1.0 = the
            DEM-derived nominal amplitude spectrum).
rho_rock    rock areal density [rocks per 100 m^2] with diameter >= 0.2 m.
slope_deg   regional slope magnitude [deg] along a random direction.
sun_e       solar elevation [deg]. Polar sites live at 0.5-6 deg.
sun_psi     solar azimuth [deg], measured clockwise from the +x drive axis.
tau_dust    optical depth of suspended / lens-deposited dust (0 = clean optics).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

DIMS: List[str] = ["k_soil", "theta_r", "r_terrain", "rho_rock",
                   "slope_deg", "sun_e", "sun_psi", "tau_dust"]
N_DIMS = len(DIMS)

UNITS: Dict[str, str] = {
    "k_soil": "x nominal", "theta_r": "rad", "r_terrain": "x nominal",
    "rho_rock": "per 100 m^2", "slope_deg": "deg", "sun_e": "deg",
    "sun_psi": "deg", "tau_dust": "optical depth",
}

# Physical support of each axis. Nothing outside these bounds is ever simulated,
# so every prior below must place all of its mass inside them (asserted in
# tests/selftest.py) — except P3's sun_e, which is the documented finding.
BOUNDS: Dict[str, tuple] = {
    "k_soil":    (0.5, 5.0),
    "theta_r":   (0.61, 0.87),      # 35-50 deg
    "r_terrain": (0.5, 2.0),
    "rho_rock":  (0.0, 12.0),
    "slope_deg": (0.0, 15.0),
    "sun_e":     (0.5, 6.0),
    "sun_psi":   (0.0, 360.0),
    "tau_dust":  (0.0, 0.6),
}

# Simulator defaults (OmniLRS / Isaac Sim lunar scene as shipped). sun_e = 45
# is an Earth-convention default and lies OUTSIDE the polar support above.
SIM_DEFAULTS: Dict[str, float] = {
    "k_soil": 1.0, "theta_r": 1.047, "r_terrain": 1.0, "rho_rock": 2.0,
    "slope_deg": 3.0, "sun_e": 45.0, "sun_psi": 180.0, "tau_dust": 0.0,
}

# A benign, in-support south-polar operating point (centre of P2). Used as the
# baseline for what-if / sensitivity sweeps.
NOMINAL_X: Dict[str, float] = {
    "k_soil": 2.0, "theta_r": 0.724, "r_terrain": 1.0, "rho_rock": 2.0,
    "slope_deg": 4.0, "sun_e": 1.5, "sun_psi": 200.0, "tau_dust": 0.05,
}


# --------------------------------------------------------------------------
# marginals
# --------------------------------------------------------------------------
class Marginal:
    """A 1-D distribution with vectorised pdf and sampling."""
    kind = "base"

    def pdf(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> Dict[str, object]:
        return {"kind": self.kind}


@dataclass
class Uniform(Marginal):
    lo: float
    hi: float
    kind = "uniform"

    def pdf(self, x):
        x = np.asarray(x, float)
        return np.where((x >= self.lo) & (x <= self.hi), 1.0 / (self.hi - self.lo), 0.0)

    def sample(self, rng, n):
        return rng.uniform(self.lo, self.hi, n)

    def describe(self):
        return {"kind": self.kind, "lo": self.lo, "hi": self.hi}


@dataclass
class LogUniform(Marginal):
    lo: float
    hi: float
    kind = "log-uniform"

    def pdf(self, x):
        x = np.asarray(x, float)
        ok = (x >= self.lo) & (x <= self.hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = 1.0 / (x * (math.log(self.hi) - math.log(self.lo)))
        return np.where(ok, d, 0.0)

    def sample(self, rng, n):
        return np.exp(rng.uniform(math.log(self.lo), math.log(self.hi), n))

    def describe(self):
        return {"kind": self.kind, "lo": self.lo, "hi": self.hi}


@dataclass
class Triangular(Marginal):
    lo: float
    hi: float
    mode: float
    log: bool = False
    kind = "triangular"

    def _t(self, v):
        return np.log(v) if self.log else v

    def pdf(self, x):
        x = np.asarray(x, float)
        ok = (x >= self.lo) & (x <= self.hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            t, a, b, m = self._t(np.maximum(x, 1e-12)), self._t(self.lo), self._t(self.hi), self._t(self.mode)
            left = 2 * (t - a) / ((b - a) * (m - a)) if m > a else np.zeros_like(t)
            right = 2 * (b - t) / ((b - a) * (b - m)) if b > m else np.zeros_like(t)
            d = np.where(t <= m, left, right)
            if self.log:
                d = d / x  # jacobian of the log transform
        return np.where(ok, np.maximum(d, 0.0), 0.0)

    def sample(self, rng, n):
        a, b, m = self._t(self.lo), self._t(self.hi), self._t(self.mode)
        t = rng.triangular(a, m, b, n)
        return np.exp(t) if self.log else t

    def describe(self):
        return {"kind": self.kind + ("-log" if self.log else ""),
                "lo": self.lo, "hi": self.hi, "mode": self.mode}


@dataclass
class TruncGauss(Marginal):
    """Gaussian truncated to [lo, hi] (renormalised). If the mean is far outside
    the support, the *un-truncated* density is used so that the mass actually
    placed on the support is honestly near zero — this is how P3's sun_e
    default of 45 deg reports ~0 weight on polar lighting."""
    mu: float
    sd: float
    lo: float
    hi: float
    truncate: bool = True
    kind = "gaussian"

    def _z(self):
        if not self.truncate:
            return 1.0
        from math import erf, sqrt
        a = 0.5 * (1 + erf((self.lo - self.mu) / (self.sd * sqrt(2))))
        b = 0.5 * (1 + erf((self.hi - self.mu) / (self.sd * sqrt(2))))
        return max(b - a, 1e-300)

    def pdf(self, x):
        x = np.asarray(x, float)
        ok = (x >= self.lo) & (x <= self.hi)
        d = np.exp(-0.5 * ((x - self.mu) / self.sd) ** 2) / (self.sd * math.sqrt(2 * math.pi))
        return np.where(ok, d / self._z(), 0.0)

    def sample(self, rng, n):
        if not self.truncate:
            # the prior's actual belief, which may lie outside the simulated support
            return rng.normal(self.mu, self.sd, n)
        out = np.empty(n)
        filled = 0
        while filled < n:
            cand = rng.normal(self.mu, self.sd, max(n - filled, 16) * 2)
            cand = cand[(cand >= self.lo) & (cand <= self.hi)]
            k = min(len(cand), n - filled)
            out[filled:filled + k] = cand[:k]
            filled += k
            if len(cand) == 0:  # mean far outside support: fall back to uniform
                out[filled:] = rng.uniform(self.lo, self.hi, n - filled)
                break
        return out

    def describe(self):
        return {"kind": self.kind, "mu": self.mu, "sd": self.sd,
                "lo": self.lo, "hi": self.hi, "truncated": self.truncate}


# --------------------------------------------------------------------------
# priors
# --------------------------------------------------------------------------
@dataclass
class Prior:
    name: str
    label: str
    blurb: str
    marginals: Dict[str, Marginal]

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """X: (n, 8) array in DIMS order -> (n,) joint density."""
        X = np.atleast_2d(np.asarray(X, float))
        p = np.ones(X.shape[0])
        for j, d in enumerate(DIMS):
            p *= self.marginals[d].pdf(X[:, j])
        return p

    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(X))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return np.stack([self.marginals[d].sample(rng, n) for d in DIMS], axis=1)

    def describe(self) -> Dict[str, object]:
        return {"name": self.name, "label": self.label, "blurb": self.blurb,
                "marginals": {d: self.marginals[d].describe() for d in DIMS}}


def _b(d):
    return BOUNDS[d]


P1 = Prior(
    "P1", "max ignorance",
    "Widest physically admissible support, log-uniform where the axis spans an "
    "order of magnitude. The coverage component of the sampling proposal q.",
    {
        "k_soil": LogUniform(*_b("k_soil")),
        "theta_r": Uniform(*_b("theta_r")),
        "r_terrain": Uniform(*_b("r_terrain")),
        "rho_rock": Uniform(*_b("rho_rock")),
        "slope_deg": Uniform(*_b("slope_deg")),
        "sun_e": Uniform(*_b("sun_e")),
        "sun_psi": Uniform(*_b("sun_psi")),
        "tau_dust": Uniform(*_b("tau_dust")),
    })

P2 = Prior(
    "P2", "VIPER-spec south pole",
    "Marginals skewed to the VIPER landing-site characterisation: firmer soil, "
    "moderate relief, low-sun-heavy lighting, light dust.",
    {
        "k_soil": Triangular(0.5, 5.0, mode=2.0, log=True),
        "theta_r": Uniform(0.663, 0.785),            # 38-45 deg central band
        "r_terrain": Uniform(0.75, 1.5),
        "rho_rock": Triangular(0.0, 8.0, mode=2.0),
        "slope_deg": Triangular(0.0, 12.0, mode=4.0),
        "sun_e": Triangular(0.5, 6.0, mode=1.5),     # low-sun skew
        "sun_psi": Uniform(0.0, 360.0),
        "tau_dust": Triangular(0.0, 0.3, mode=0.05),
    })

P3 = Prior(
    "P3", "simulator default",
    "Tight kernels (sd = 5% of the default) around the OmniLRS/Isaac Sim "
    "shipped scene. sun_e defaults to 45 deg, outside polar support: this "
    "prior believes none of the lighting the rover will actually meet.",
    {d: TruncGauss(SIM_DEFAULTS[d], max(0.05 * abs(SIM_DEFAULTS[d]), 1e-3),
                   *BOUNDS[d], truncate=(d != "sun_e"))
     for d in DIMS},
)
# rho_rock default 2.0 -> sd 0.1; tau_dust default 0 -> sd 1e-3 (essentially clean)

P4 = Prior(
    "P4", "equatorial mare",
    "Apollo-site style: high sun (sampled only at its low-elevation edge, as "
    "the rover would meet it near local dawn), flat mare, sparse rocks. "
    "Mostly out of the polar sampling support: expect a small ESS.",
    {
        "k_soil": Triangular(0.7, 3.0, mode=1.0, log=True),
        "theta_r": Uniform(0.61, 0.75),
        "r_terrain": Uniform(0.5, 1.0),
        "rho_rock": Triangular(0.0, 4.0, mode=1.0),
        "slope_deg": Triangular(0.0, 6.0, mode=1.5),
        "sun_e": Uniform(4.0, 6.0),
        "sun_psi": Uniform(0.0, 360.0),
        "tau_dust": Triangular(0.0, 0.2, mode=0.02),
    })

P5 = Prior(
    "P5", "PSR rim, deep winter",
    "Permanently-shadowed-region rim traverse: grazing sun, steep crater "
    "flanks, loose cold-trap regolith, rock-strewn ejecta.",
    {
        "k_soil": Triangular(0.5, 5.0, mode=0.8, log=True),
        "theta_r": Uniform(0.61, 0.72),
        "r_terrain": Uniform(1.0, 2.0),
        "rho_rock": Triangular(2.0, 12.0, mode=6.0),
        "slope_deg": Triangular(4.0, 15.0, mode=9.0),
        "sun_e": Triangular(0.5, 2.5, mode=0.8),
        "sun_psi": Uniform(0.0, 360.0),
        "tau_dust": Triangular(0.0, 0.4, mode=0.1),
    })

P6 = Prior(
    "P6", "post-landing plume",
    "Optics contaminated by descent-engine plume deposition; everything else "
    "as VIPER-spec.",
    {**P2.marginals, "tau_dust": Triangular(0.25, 0.6, mode=0.45)},
)

PRIORS: Dict[str, Prior] = {p.name: p for p in (P1, P2, P3, P4, P5, P6)}


class Mixture(Prior):
    """Defensive mixture proposal q = Σ α_k p_k (Hesterberg 1995; Veach & Guibas
    1995 'balance heuristic'). Sampling once from q and re-weighting by p_k/q
    keeps every candidate prior's ESS bounded below by roughly α_k·n instead of
    collapsing when a narrow prior sits inside a wide one."""

    def __init__(self, name: str, label: str, blurb: str, components: Sequence[Prior],
                 alphas: Sequence[float]):
        a = np.asarray(alphas, float)
        self.components = list(components)
        self.alphas = a / a.sum()
        super().__init__(name, label, blurb, dict(components[0].marginals))

    def pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return sum(a * c.pdf(X) for a, c in zip(self.alphas, self.components))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        k = rng.choice(len(self.components), size=n, p=self.alphas)
        X = np.empty((n, N_DIMS))
        for j, c in enumerate(self.components):
            m = k == j
            if m.any():
                X[m] = c.sample(rng, int(m.sum()))
        return X

    def describe(self) -> Dict[str, object]:
        return {"name": self.name, "label": self.label, "blurb": self.blurb,
                "components": [{"prior": c.name, "alpha": round(float(a), 3)}
                               for c, a in zip(self.components, self.alphas)]}


# q: 40 % max-ignorance for coverage, 15 % each of the four in-support mission
# priors so their weights stay bounded. P3 is deliberately NOT a component: its
# sun_e mass lies outside the physical support, so it cannot be sampled — the
# resulting ESS ~ 0 is the finding, not a bug.
PROPOSAL: Prior = Mixture(
    "Q", "defensive mixture proposal",
    "0.4 P1 + 0.15 (P2 + P4 + P5 + P6). Every simulated episode is drawn here.",
    (P1, P2, P4, P5, P6), (0.40, 0.15, 0.15, 0.15, 0.15))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def to_dict(x: Sequence[float]) -> Dict[str, float]:
    return {d: float(v) for d, v in zip(DIMS, x)}


def from_dict(d: Dict[str, float]) -> np.ndarray:
    return np.array([float(d[k]) for k in DIMS])


def clip_to_bounds(X: np.ndarray) -> np.ndarray:
    X = np.array(X, float, copy=True)
    for j, d in enumerate(DIMS):
        lo, hi = BOUNDS[d]
        X[..., j] = np.clip(X[..., j], lo, hi)
    return X


def importance_weights(X: np.ndarray, target: Prior, proposal: Prior = PROPOSAL) -> np.ndarray:
    """Self-normalised weights w_i ∝ p(x_i)/q(x_i), mean 1 (zeros stay zero)."""
    q = proposal.pdf(X)
    p = target.pdf(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(q > 0, p / q, 0.0)
    s = w.sum()
    return w * (len(w) / s) if s > 0 else w


def ess(w: np.ndarray) -> float:
    """Kish effective sample size (Σw)² / Σw²."""
    s = float(w.sum())
    return (s * s / float((w * w).sum())) if s > 0 else 0.0


def sample_proposal(rng: np.random.Generator, n: int) -> np.ndarray:
    return PROPOSAL.sample(rng, n)


def describe_all() -> List[Dict[str, object]]:
    return [p.describe() for p in PRIORS.values()]
