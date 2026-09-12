"""Astraeus — disturbance priors (table v1.1).

Sampling distribution q = P1 (max ignorance). P2/P3 enter ONLY as densities
for post-hoc importance re-weighting: w_k(x) = p_k(x) / q(x).

P3 is formally a delta at simulator defaults; a delta cannot re-weight a
continuous sample, so it is implemented as a tight kernel around the defaults
(documented modelling choice — state it in the report).

Two tiers (honest about what the sim lets us change when):
  BATCH tier  — load-time yaml params, fixed per sim launch: k_soil, theta_r
  EPISODE tier — runtime-changeable via ROS: r_terrain*, terrain_seed, sun e, psi
  (*if terrain randomisation topic accepts relief scaling; else r_terrain moves
   to the batch tier — flip the flag below.)
"""
import numpy as np

RNG = None  # set by init(seed)

RANGES = {
    "k_soil":    (0.5, 5.0),      # log-uniform under P1
    "theta_r":   (0.61, 0.87),    # rad (35-50 deg)
    "r_terrain": (0.5, 2.0),
    "sun_e":     (0.5, 6.0),      # deg
    "sun_psi":   (0.0, 360.0),    # deg
}
SIM_DEFAULTS = {"k_soil": 1.0, "theta_r": 1.047, "r_terrain": 1.0,
                "sun_e": 45.0, "sun_psi": 180.0}
# NOTE: P3's sun_e default (45.0) lies OUTSIDE the sampled range [0.5, 6].
# Its kernel therefore assigns ~zero weight to every sampled episode on that
# axis — which IS the finding (the sim-default prior believes none of the
# physically possible polar lighting). Report it; don't "fix" it.

R_TERRAIN_IS_EPISODE_TIER = True  # flip to False if relief is load-time only


def init(seed: int):
    global RNG
    RNG = np.random.default_rng(seed)


# ---------- P1: sampling ----------------------------------------------------
def sample_batch():
    lo, hi = RANGES["k_soil"]
    k = float(np.exp(RNG.uniform(np.log(lo), np.log(hi))))
    th = float(RNG.uniform(*RANGES["theta_r"]))
    out = {"k_soil": k, "theta_r": th}
    if not R_TERRAIN_IS_EPISODE_TIER:
        out["r_terrain"] = float(RNG.uniform(*RANGES["r_terrain"]))
    return out


def sample_episode():
    out = {
        "sun_e": float(RNG.uniform(*RANGES["sun_e"])),
        "sun_psi": float(RNG.uniform(*RANGES["sun_psi"])),
        "terrain_seed": int(RNG.integers(0, 2**31 - 1)),
    }
    if R_TERRAIN_IS_EPISODE_TIER:
        out["r_terrain"] = float(RNG.uniform(*RANGES["r_terrain"]))
    return out


# ---------- densities (per-component; independence per table v1.1) ----------
def _u(x, lo, hi):
    return (1.0 / (hi - lo)) if lo <= x <= hi else 0.0


def _logu(x, lo, hi):  # log-uniform pdf
    return (1.0 / (x * (np.log(hi) - np.log(lo)))) if lo <= x <= hi else 0.0


def _tri_log(x, lo, hi, mode):  # triangular on log scale (P2 k_soil)
    lx, llo, lhi, lm = np.log(x), np.log(lo), np.log(hi), np.log(mode)
    if not (llo <= lx <= lhi):
        return 0.0
    if lx <= lm:
        d = 2 * (lx - llo) / ((lhi - llo) * (lm - llo))
    else:
        d = 2 * (lhi - lx) / ((lhi - llo) * (lhi - lm))
    return d / x  # jacobian


def _gauss(x, mu, sd):
    return float(np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)))


def pdf_P1(x):
    p = _logu(x["k_soil"], *RANGES["k_soil"])
    p *= _u(x["theta_r"], *RANGES["theta_r"])
    p *= _u(x["r_terrain"], *RANGES["r_terrain"])
    p *= _u(x["sun_e"], *RANGES["sun_e"])
    p *= _u(x["sun_psi"], *RANGES["sun_psi"])
    return p


def pdf_P2(x):
    """VIPER-spec-weighted. Lighting marginal: until the ephemeris table is
    generated, approximate with a low-elevation-skewed triangular on [0.5, 6]
    (mode 1.5 deg) and uniform azimuth — TODO(ephemeris): replace with the
    empirical (e, psi) table density. Terrain: soft skew per table v1.1."""
    p = _tri_log(x["k_soil"], 0.5, 5.0, mode=2.0)
    p *= _u(x["theta_r"], 0.663, 0.785)          # 38-45 deg central band
    p *= _u(x["r_terrain"], 0.75, 1.5)           # moderate relief around site stats
    lo, hi, m = 0.5, 6.0, 1.5                    # low-sun skew (triangular)
    e = x["sun_e"]
    if lo <= e <= hi:
        pe = 2*(e-lo)/((hi-lo)*(m-lo)) if e <= m else 2*(hi-e)/((hi-lo)*(hi-m))
    else:
        pe = 0.0
    p *= pe
    p *= _u(x["sun_psi"], *RANGES["sun_psi"])
    return p


def pdf_P3(x, rel_sd=0.05):
    """Sim-default as tight kernels (sd = rel_sd * default, documented)."""
    p = 1.0
    for k, mu in SIM_DEFAULTS.items():
        p *= _gauss(x[k], mu, max(rel_sd * abs(mu), 1e-6))
    return p
