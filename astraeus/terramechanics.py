"""Wheel-soil interaction: Bekker pressure-sinkage and Janosi-Hanamoto shear.

The rover is a rigid-wheeled vehicle on dry granular regolith. Per wheel:

  static sinkage (Bekker 1969, rigid wheel, Wong & Reece 1967 form)
      z0 = [ 3 W / ((3 - n) b sqrt(D) (k_c / b + k_phi)) ] ^ (2 / (2n + 1))

  compaction resistance
      R_c = b (k_c / b + k_phi) z^(n+1) / (n + 1)

  maximum thrust (Mohr-Coulomb soil, Janosi & Hanamoto 1961 slip law)
      H(i) = (A c + W tan(phi)) · [ 1 - K / (i L) (1 - exp(-i L / K)) ]

  slip-sinkage: while a wheel spins it excavates. We integrate a digging
  rate proportional to i^2 v (Ding et al. 2011 report sinkage growing
  super-linearly in slip) on top of z0; entrapment is declared when the total
  exceeds a fraction of the wheel radius.

Nominal lunar regolith parameters follow the Lunar Sourcebook (Heiken,
Vaniman & French 1991, ch. 9) and the LRV mobility studies collected in Wong
(2008). `k_soil` scales both sinkage moduli; `theta_r` is phi.

All functions are vectorised over episodes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G_MOON = 1.62        # m/s^2


@dataclass(frozen=True)
class Soil:
    k_c: float = 1400.0        # N / m^(n+1)
    k_phi: float = 820_000.0   # N / m^(n+2)
    n: float = 1.0
    c: float = 170.0           # Pa cohesion
    K: float = 0.018           # m shear deformation modulus


@dataclass(frozen=True)
class Wheel:
    radius: float = 0.25       # m (VIPER-class 0.5 m diameter)
    width: float = 0.20        # m
    count: int = 4


NOMINAL_SOIL = Soil()
WHEEL = Wheel()

# Entrapment threshold: total sinkage beyond this fraction of the radius
# leaves the wheel bulldozing its own berm (Spirit at Troy, 2009).
Z_CRIT_FRACTION = 0.5
DIG_GAIN = 0.016              # m of sinkage per (unit i^2 * m of commanded travel)
I_MAX = 0.92                  # slip beyond which we declare immobilisation


def static_sinkage(W: np.ndarray, k_soil: np.ndarray, soil: Soil = NOMINAL_SOIL,
                   wheel: Wheel = WHEEL) -> np.ndarray:
    """Bekker rigid-wheel static sinkage (m) for per-wheel load W (N)."""
    kc = soil.k_c * k_soil
    kp = soil.k_phi * k_soil
    D = 2 * wheel.radius
    denom = (3 - soil.n) * wheel.width * np.sqrt(D) * (kc / wheel.width + kp)
    return np.power(np.clip(3 * W / denom, 0.0, None), 2.0 / (2 * soil.n + 1))


def compaction_resistance(z: np.ndarray, k_soil: np.ndarray, soil: Soil = NOMINAL_SOIL,
                          wheel: Wheel = WHEEL) -> np.ndarray:
    kc = soil.k_c * k_soil
    kp = soil.k_phi * k_soil
    return wheel.width * (kc / wheel.width + kp) * np.power(z, soil.n + 1) / (soil.n + 1)


def contact_length(z: np.ndarray, wheel: Wheel = WHEEL) -> np.ndarray:
    r = wheel.radius
    return r * np.arccos(np.clip((r - z) / r, -1.0, 1.0))


def slip_curve(i: np.ndarray, L: np.ndarray, K: float) -> np.ndarray:
    """Janosi-Hanamoto thrust fraction f(i) in [0, 1)."""
    i = np.clip(i, 1e-6, None)
    a = i * L / K
    return 1.0 - (1.0 - np.exp(-a)) / a


def max_thrust(W: np.ndarray, z: np.ndarray, phi: np.ndarray, soil: Soil = NOMINAL_SOIL,
               wheel: Wheel = WHEEL) -> np.ndarray:
    """A c + W tan(phi), per wheel."""
    A = wheel.width * contact_length(z, wheel)
    return A * soil.c + W * np.tan(phi)


_I_GRID = np.concatenate([np.linspace(0.0, 0.2, 41), np.linspace(0.21, I_MAX, 40)])


def required_slip(thrust_needed: np.ndarray, H_max: np.ndarray, L: np.ndarray,
                  K: float = NOMINAL_SOIL.K) -> tuple:
    """Invert H(i) = thrust_needed for i. Returns (slip, immobilised mask).

    f(i) is monotone in i, so we evaluate it on a fixed grid per episode and
    interpolate. Episodes whose demand exceeds H(I_MAX) are immobilised.
    """
    ratio = np.where(H_max > 0, thrust_needed / np.maximum(H_max, 1e-9), np.inf)
    f = slip_curve(_I_GRID[None, :], L[:, None], K)          # (n, G) increasing in i
    f_max = f[:, -1]
    immobilised = ratio >= f_max
    r = np.clip(ratio, 0.0, None)
    # vectorised monotone interpolation: index of first grid point with f >= ratio
    idx = (f < r[:, None]).sum(1)
    idx = np.clip(idx, 1, f.shape[1] - 1)
    f_lo = np.take_along_axis(f, (idx - 1)[:, None], 1)[:, 0]
    f_hi = np.take_along_axis(f, idx[:, None], 1)[:, 0]
    t = np.where(f_hi > f_lo, (r - f_lo) / np.maximum(f_hi - f_lo, 1e-12), 0.0)
    slip = _I_GRID[idx - 1] + np.clip(t, 0.0, 1.0) * (_I_GRID[idx] - _I_GRID[idx - 1])
    slip = np.where(r <= 0, 0.0, slip)
    slip = np.where(immobilised, 1.0, slip)
    return slip, immobilised


def step_wheel_soil(mass: float, pitch: np.ndarray, k_soil: np.ndarray, phi: np.ndarray,
                    z_dig: np.ndarray, v_cmd: np.ndarray, dt: float,
                    extra_resistance: np.ndarray,
                    soil: Soil = NOMINAL_SOIL, wheel: Wheel = WHEEL) -> dict:
    """One terramechanics step for a batch of rovers.

    pitch > 0 means nose-up (climbing). Returns effective forward speed, slip,
    updated excavated sinkage, and the immobilised / entrapped masks.
    """
    W_total = mass * G_MOON
    W_w = W_total * np.cos(pitch) / wheel.count
    z0 = static_sinkage(W_w, k_soil, soil, wheel)
    z = z0 + z_dig
    R_c = compaction_resistance(z, k_soil, soil, wheel) * wheel.count
    grade = W_total * np.sin(pitch)                      # downhill assists (negative)
    demand = R_c + grade + extra_resistance
    demand = np.clip(demand, 0.0, None)                  # brakes handle descent

    L = contact_length(z, wheel)
    H = max_thrust(W_w, z, phi, soil, wheel) * wheel.count
    slip, immobilised = required_slip(demand, H, L, soil.K)

    v_eff = v_cmd * (1.0 - slip)
    # excavation while slipping: rate ∝ i^2 * commanded travel
    z_dig_new = z_dig + DIG_GAIN * slip * slip * np.abs(v_cmd) * dt
    entrapped = (z0 + z_dig_new) > Z_CRIT_FRACTION * wheel.radius
    return {"v": v_eff, "slip": slip, "z": z0 + z_dig_new, "z_dig": z_dig_new,
            "R_c": R_c, "H_max": H, "demand": demand,
            "immobilised": immobilised, "entrapped": entrapped}
