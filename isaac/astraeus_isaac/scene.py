"""Map an eight-dimensional disturbance vector onto concrete scene settings.

The mapping is the single place where Astraeus' abstract disturbance
dimensions meet simulator knobs, so every assumption is written down here:

  k_soil     -> PhysX material (static/dynamic friction, restitution) *and*
                the terramechanics multiplier passed to the wheel plugin, if
                one is installed. PhysX rigid contacts do not model sinkage; a
                friction-only proxy under-predicts entrapment. See docs/ISAAC_SIM.md.
  theta_r    -> friction coefficient mu = tan(theta_r) on the ground material.
  r_terrain  -> vertical exaggeration of the exported heightfield.
  rho_rock   -> number of rock prims spawned (Poisson mean rho * area / 100).
  slope_deg  -> regional tilt baked into the heightfield (not a stage rotation,
                so gravity stays -z).
  sun_e, sun_psi -> DistantLight elevation / azimuth. Intensity is fixed at
                the solar constant; exposure is handled by the camera.
  tau_dust   -> two effects: light attenuation exp(-1.3 tau) on the sun and a
                grey haze layer of matching optical depth on the render (fog).
                Deposited dust on optics is *not* modelled in Isaac; it is in
                the local simulator (perception.py) and noted as a gap.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Sequence

DIMS = ("k_soil", "theta_r", "r_terrain", "rho_rock", "slope_deg", "sun_e", "sun_psi", "tau_dust")

SOLAR_CONSTANT_LUX = 128_000.0   # ~1361 W/m^2, photopic
DUST_EXTINCTION = 1.3            # matches astraeus.perception.DUST_EXTINCTION
GOAL_X = 30.0


@dataclass(frozen=True)
class SceneParams:
    # ground
    static_friction: float
    dynamic_friction: float
    restitution: float
    terramechanics_k_soil: float
    height_scale: float
    slope_deg: float
    # rocks
    rock_density_per_100m2: float
    # light
    sun_elevation_deg: float
    sun_azimuth_deg: float
    sun_intensity_lux: float
    sun_color: Sequence[float]
    # atmosphere-like haze (dust)
    fog_density: float
    fog_color: Sequence[float]
    # recorded verbatim
    x: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def scene_from_x(x: Sequence[float]) -> SceneParams:
    if len(x) != 8:
        raise ValueError(f"expected 8 disturbance values in DIMS order, got {len(x)}")
    d = dict(zip(DIMS, (float(v) for v in x)))
    mu = math.tan(d["theta_r"])
    # denser, stiffer regolith also has marginally higher restitution; keep small
    rest = min(0.05 + 0.02 * (d["k_soil"] - 1.0), 0.15)
    transmission = math.exp(-DUST_EXTINCTION * d["tau_dust"])
    # dust on the Moon is dark grey with a warm cast; haze colour follows the sun
    fog_rgb = (0.42, 0.40, 0.37)
    return SceneParams(
        static_friction=round(mu, 4),
        dynamic_friction=round(0.9 * mu, 4),
        restitution=round(max(rest, 0.0), 4),
        terramechanics_k_soil=d["k_soil"],
        height_scale=d["r_terrain"],
        slope_deg=d["slope_deg"],
        rock_density_per_100m2=d["rho_rock"],
        sun_elevation_deg=d["sun_e"],
        sun_azimuth_deg=d["sun_psi"],
        sun_intensity_lux=round(SOLAR_CONSTANT_LUX * transmission, 1),
        sun_color=(1.0, 0.98, 0.95),
        fog_density=round(d["tau_dust"] * DUST_EXTINCTION / 50.0, 6),   # per metre over a 50 m sightline
        fog_color=fog_rgb,
        x=d,
    )


def sun_direction(elev_deg: float, azim_deg: float) -> tuple:
    """Unit vector *towards* the sun in the ENU-like world frame (x east/goal, z up)."""
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return (math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e))


def distant_light_rotation_xyz(elev_deg: float, azim_deg: float) -> tuple:
    """Euler XYZ (degrees) for a USD DistantLight, which emits along its local -Z.

    Rotating -Z onto the direction *from* the sun to the ground: first pitch
    about X by (90 - elevation) so the beam leaves the zenith, then yaw about
    Z by the azimuth measured from +x.
    """
    return (-(90.0 - elev_deg), 0.0, azim_deg + 90.0)
