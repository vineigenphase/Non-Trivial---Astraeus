"""Lighting, dust and their effect on the rover's perception.

Polar lighting is the axis the simulator default prior ignores entirely, so it
is modelled explicitly rather than folded into a noise term:

shadow fraction     f_sh = 1 - exp(-0.5 · tan(sigma_slope) / tan(sun_e))
                    the fraction of the local relief that lies in its own
                    shadow, with sigma_slope the rms slope of the terrain
                    (≈ 5 deg · r_terrain at the wheel scale).
glare               sun within ~35 deg of the camera boresight and below
                    ~6 deg elevation saturates the sensor: g = exp(-(Δψ/35)^2).
dust                Beer-Lambert contrast loss exp(-2 tau) — the factor two
                    because deposited dust attenuates both the scene and its
                    own scattered veiling glare (Hapke-style single-scatter
                    approximation).

The scalar visibility V in [0, 1] gates two things in the rover:
  * per-step rock detection rate lambda · V (a rock lying in shadow is further
    penalised by SHADOW_DETECT_PENALTY);
  * visual-odometry drift, sigma_vo · (1 - V) per metre travelled.

With the ground-truth-pose flag on, V still affects rock detection but the
pose is exact — the configuration the OmniLRS harness ran under.
"""
from __future__ import annotations

import numpy as np

RMS_SLOPE_NOMINAL_DEG = 5.0
GLARE_HALF_WIDTH_DEG = 35.0
GLARE_ELEV_SCALE_DEG = 6.0
DUST_EXTINCTION = 2.0
SHADOW_DETECT_PENALTY = 0.15       # detection rate multiplier for rocks in shadow
DETECT_RATE = 1.5                  # 1/s at V = 1, rock inside the sensor footprint
LOOKAHEAD_M = 7.0
SENSOR_HALF_FOV_DEG = 55.0
SIGMA_VO_PER_M = 0.035             # m of pose drift per m travelled at V = 0


def shadow_fraction(sun_e_deg: np.ndarray, r_terrain: np.ndarray) -> np.ndarray:
    sigma = np.radians(RMS_SLOPE_NOMINAL_DEG) * np.asarray(r_terrain, float)
    ratio = np.tan(sigma) / np.tan(np.radians(np.clip(sun_e_deg, 0.05, 89.0)))
    return 1.0 - np.exp(-0.5 * ratio)


def glare(sun_psi_deg: np.ndarray, heading_rad: np.ndarray, sun_e_deg: np.ndarray) -> np.ndarray:
    """0 = no glare, 1 = sun on the boresight at the horizon."""
    d = np.degrees(heading_rad) - np.asarray(sun_psi_deg, float)
    d = (d + 180.0) % 360.0 - 180.0
    ang = np.exp(-(d / GLARE_HALF_WIDTH_DEG) ** 2)
    elev = np.clip(1.0 - np.asarray(sun_e_deg, float) / GLARE_ELEV_SCALE_DEG, 0.0, 1.0)
    return ang * elev


def dust_transmission(tau: np.ndarray) -> np.ndarray:
    return np.exp(-DUST_EXTINCTION * np.asarray(tau, float))


def visibility(sun_e_deg, sun_psi_deg, tau_dust, r_terrain, heading_rad) -> np.ndarray:
    """Composite visibility V in [0, 1]."""
    f_sh = shadow_fraction(sun_e_deg, r_terrain)
    g = glare(sun_psi_deg, heading_rad, sun_e_deg)
    contrast = (1.0 - 0.6 * f_sh) * (1.0 - 0.85 * g)
    return np.clip(contrast * dust_transmission(tau_dust), 0.0, 1.0)


def visibility_components(sun_e_deg, sun_psi_deg, tau_dust, r_terrain, heading_rad) -> dict:
    return {
        "shadow_fraction": shadow_fraction(sun_e_deg, r_terrain),
        "glare": glare(sun_psi_deg, heading_rad, sun_e_deg),
        "dust_transmission": dust_transmission(tau_dust),
        "visibility": visibility(sun_e_deg, sun_psi_deg, tau_dust, r_terrain, heading_rad),
    }
