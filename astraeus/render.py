"""Physically-motivated software renderer for Astraeus episodes.

Pure numpy ray-marching against the episode's height-field, with the same
sun and dust parameters the simulation used — so what you see is what the
rover's perception model was scored on:

  * Lommel-Seeliger regolith BRDF with an opposition surge (Hapke-like), the
    standard first-order model for lunar soil photometry.
  * Cast shadows by secondary ray-marching toward the sun; at 0.5-6 deg
    elevation these are the dominant feature of any polar scene.
  * Suspended dust as a single-scattering haze: Beer-Lambert extinction along
    the view ray plus forward-scattered glow around the sun.
  * The rover as analytic primitives (oriented boxes + wheel cylinders) placed
    on the terrain with the simulated pitch/roll/heading.
  * The true trajectory and the visual-odometry estimate painted onto the
    ground, the goal as a beacon, and a HUD with the 8 disturbance values.

Nothing here is a learned world model: the frames are synthetic, deterministic
and reproducible from (parameters, seed). A frame of 1280x720 takes ~1-3 s on a
laptop CPU; the live UI renders the same scene in WebGL and only asks this
module for stills.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import perception as P
from . import rover as RV
from .priors import DIMS, UNITS
from .terrain import GOAL_X, value_noise

# --------------------------------------------------------------------------- setup
DEM_RES = 0.20                       # m per DEM cell for ray marching
DEM_X = (-30.0, 70.0)
DEM_Y = (-45.0, 45.0)
ALBEDO = 0.11                        # mature highland regolith, normal albedo ~0.1
DUST_HAZE_LENGTH = 40.0              # m at which tau_dust = 1 gives e^-1 transmission
SUN_ANGULAR_RADIUS = math.radians(0.27)

WHEEL_R, WHEEL_W = 0.25, 0.20
# Driving lights (VIPER carries LED headlights for shadowed terrain): local
# position, boresight tilt below horizontal, cone half-angle, radiant intensity
# relative to solar irradiance at 1 m.
LAMPS = ((0.85, 0.45, 1.05), (0.85, -0.45, 1.05))
LAMP_TILT_DEG = 11.0
LAMP_CONE_DEG = 26.0
LAMP_I0 = 9.0
HALF_LEN, HALF_WID = RV.HALF_LEN, RV.HALF_WID

_FONT_CACHE: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(mono: bool, size: int) -> ImageFont.ImageFont:
    key = ("mono" if mono else "sans", size)
    if key not in _FONT_CACHE:
        try:
            import matplotlib
            base = matplotlib.get_data_path() + "/fonts/ttf/"
            _FONT_CACHE[key] = ImageFont.truetype(base + ("DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf"), size)
        except Exception:  # pragma: no cover - font fallback
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


# --------------------------------------------------------------------------- scene
@dataclass
class Scene:
    """Everything the renderer needs for one frame, extracted from a recorded episode."""
    dem_x: np.ndarray
    dem_y: np.ndarray
    dem_z: np.ndarray                 # (ny, nx)
    params: Dict[str, float]
    sun_dir: np.ndarray               # unit vector toward the sun
    tau_dust: float
    rover_pos: np.ndarray             # (3,) contact-centre position
    rover_att: Tuple[float, float, float]   # heading, pitch, roll (rad)
    path_true: np.ndarray             # (k, 2)
    path_est: np.ndarray              # (k, 2)
    t: float
    outcome: str
    telemetry: Dict[str, float]
    seed: int
    frame: int
    n_frames: int

    @classmethod
    def from_result(cls, res: RV.EpisodeResult, frame: int = -1) -> "Scene":
        assert res.trace is not None and res.terrain is not None, "record=True episode required"
        tr = res.trace
        n_frames = tr["x"].shape[1]
        k = n_frames - 1 if frame < 0 else min(frame, n_frames - 1)
        x = res.X[0]
        params = {d: float(v) for d, v in zip(DIMS, x)}
        e = math.radians(params["sun_e"])
        psi = math.radians(params["sun_psi"])
        sun = np.array([math.cos(psi) * math.cos(e), math.sin(psi) * math.cos(e), math.sin(e)])
        dem = res.terrain.dem(0, DEM_RES, DEM_X, DEM_Y)
        pos = np.array([tr["x"][0, k], tr["y"][0, k], tr["z"][0, k]])
        att = (float(tr["heading"][0, k]), float(tr["pitch"][0, k]), float(tr["roll"][0, k]))
        tele = {
            "v": float(tr["v"][0, k]), "slip": float(tr["slip"][0, k]),
            "sinkage": float(tr["sinkage"][0, k]), "visibility": float(tr["visibility"][0, k]),
            "pitch_deg": math.degrees(att[1]), "roll_deg": math.degrees(att[2]),
            "vo_error": float(math.hypot(tr["ex"][0, k] - pos[0], tr["ey"][0, k] - pos[1])),
            "dist_to_goal": float(math.hypot(GOAL_X - pos[0], pos[1])),
        }
        return cls(dem["x"], dem["y"], dem["z"], params, sun, params["tau_dust"], pos, att,
                   np.stack([tr["x"][0, :k + 1], tr["y"][0, :k + 1]], 1),
                   np.stack([tr["ex"][0, :k + 1], tr["ey"][0, :k + 1]], 1),
                   float(tr["t"][0, k]), RV.OUTCOMES[int(res.outcome[0])] if k == n_frames - 1 else "driving",
                   tele, int(res.seed[0]), k, n_frames)

    # height-field lookups --------------------------------------------------
    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear DEM lookup; outside the DEM returns -inf (sky)."""
        fx = (x - self.dem_x[0]) / DEM_RES
        fy = (y - self.dem_y[0]) / DEM_RES
        nx, ny = len(self.dem_x), len(self.dem_y)
        inside = (fx >= 0) & (fx < nx - 1) & (fy >= 0) & (fy < ny - 1)
        fx = np.clip(fx, 0, nx - 1.0001)
        fy = np.clip(fy, 0, ny - 1.0001)
        ix = fx.astype(np.int64)
        iy = fy.astype(np.int64)
        tx = fx - ix
        ty = fy - iy
        z = self.dem_z
        h = ((z[iy, ix] * (1 - tx) + z[iy, ix + 1] * tx) * (1 - ty)
             + (z[iy + 1, ix] * (1 - tx) + z[iy + 1, ix + 1] * tx) * ty)
        return np.where(inside, h, -np.inf)

    def normal(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = DEM_RES * 0.5
        dzdx = (self.height(x + d, y) - self.height(x - d, y)) / (2 * d)
        dzdy = (self.height(x, y + d) - self.height(x, y - d)) / (2 * d)
        n = np.stack([-dzdx, -dzdy, np.ones_like(x)], -1)
        n = np.where(np.isfinite(n), n, 0.0)
        return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


# --------------------------------------------------------------------------- camera
def camera_rays(scene: Scene, mode: str, w: int, h: int, fov_deg: float = 60.0):
    """Returns origin (3,) and per-pixel unit directions (h, w, 3)."""
    px, py, pz = scene.rover_pos
    hd = scene.rover_att[0]
    fwd = np.array([math.cos(hd), math.sin(hd), 0.0])
    if mode == "chase":
        eye = np.array([px, py, pz]) - 6.0 * fwd + np.array([0, 0, 2.6])
        look = np.array([px, py, pz + 0.7]) + 3.0 * fwd
    elif mode == "rover_cam":
        # mast-head camera: just ahead of the mast top (local x 0.55, z 2.03 + wheel radius)
        eye = np.array([px, py, pz + WHEEL_R + 2.12]) + 0.95 * fwd
        look = eye + 8.0 * fwd - np.array([0, 0, 1.4])
        fov_deg = 2 * P.SENSOR_HALF_FOV_DEG * 0.85
    elif mode == "cinematic":
        # low three-quarter front view; the sun rakes across the frame from the side
        side = np.array([-fwd[1], fwd[0], 0.0])
        eye = np.array([px, py, pz]) + 5.6 * fwd + 4.2 * side + np.array([0, 0, 1.9])
        look = np.array([px, py, pz + 0.55]) - 2.0 * fwd - 0.6 * side
        fov_deg = 44.0
    elif mode == "overview":
        eye = np.array([GOAL_X * 0.5 - 4.0, -24.0, 14.0])
        look = np.array([GOAL_X * 0.5, 0.0, 0.0])
        fov_deg = 58.0
    elif mode == "orbit":
        ang = 2 * math.pi * (scene.frame / max(scene.n_frames - 1, 1))
        eye = np.array([px + 9 * math.cos(ang), py + 9 * math.sin(ang), pz + 4.0])
        look = np.array([px, py, pz + 0.5])
    else:
        raise ValueError(f"unknown camera mode {mode!r}")
    f = look - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, np.array([0, 0, 1.0]))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    aspect = w / h
    tan = math.tan(math.radians(fov_deg) / 2)
    sx = (np.arange(w) + 0.5) / w * 2 - 1
    sy = 1 - (np.arange(h) + 0.5) / h * 2
    dx, dy = np.meshgrid(sx * tan * aspect, sy * tan)
    d = f[None, None] + dx[..., None] * r[None, None] + dy[..., None] * u[None, None]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return eye, d


# --------------------------------------------------------------------------- marching
def march_terrain(scene: Scene, o: np.ndarray, d: np.ndarray, t_max: float = 160.0,
                  steps: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    """Height-field ray march. o (3,), d (N, 3) -> (t (N,), hit (N,) bool)."""
    N = d.shape[0]
    t = np.full(N, 0.05)
    hit = np.zeros(N, bool)
    t_hit = np.full(N, np.inf)
    prev_t = t.copy()
    prev_dh = np.full(N, 1.0)
    active = np.ones(N, bool)
    for _ in range(steps):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        p = o[None, :] + t[idx, None] * d[idx]
        hgt = scene.height(p[:, 0], p[:, 1])
        dh = p[:, 2] - hgt                      # positive above ground
        below = dh < 0
        # bisection refine the crossing
        if below.any():
            j = idx[below]
            a, b = prev_t[j], t[j]
            for _ in range(6):
                m = 0.5 * (a + b)
                pm = o[None, :] + m[:, None] * d[j]
                above = pm[:, 2] - scene.height(pm[:, 0], pm[:, 1]) > 0
                a = np.where(above, m, a)
                b = np.where(above, b, m)
            t_hit[j] = 0.5 * (a + b)
            hit[j] = True
            active[j] = False
        keep = idx[~below]
        prev_t[keep] = t[keep]
        prev_dh[keep] = dh[~below]
        # conservative-ish step: proportional to height above ground and to distance
        step = np.clip(0.55 * dh[~below], 0.04 + 0.006 * t[keep], 6.0)
        # sky rays (outside DEM) that are climbing can stop
        sky = ~np.isfinite(hgt[~below]) & (d[keep, 2] >= 0) & (t[keep] > 2.0)
        t[keep] += step
        done = (t[keep] > t_max) | sky
        active[keep[done]] = False
    return t_hit, hit


def shadow(scene: Scene, p: np.ndarray, steps: int = 40, t_max: float = 120.0) -> np.ndarray:
    """Soft shadow factor in [0, 1] by marching from p toward the sun."""
    s = scene.sun_dir
    N = p.shape[0]
    t = np.full(N, 0.12)
    vis = np.ones(N)
    active = np.ones(N, bool)
    k_pen = 12.0                                  # penumbra sharpness
    for _ in range(steps):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        q = p[idx] + t[idx, None] * s[None, :]
        hgt = scene.height(q[:, 0], q[:, 1])
        dh = q[:, 2] - hgt
        v = np.clip(k_pen * dh / t[idx], 0.0, 1.0)
        vis[idx] = np.minimum(vis[idx], np.where(np.isfinite(hgt), v, 1.0))
        t[idx] += np.clip(0.6 * np.abs(dh) + 0.05, 0.08, 5.0)
        active[idx[(vis[idx] <= 0.0) | (t[idx] > t_max) | ~np.isfinite(hgt)]] = False
    return vis


# --------------------------------------------------------------------------- primitives
def _rot(heading: float, pitch: float, roll: float) -> np.ndarray:
    """Body-to-world rotation (yaw about z, pitch about y, roll about x)."""
    ch, sh = math.cos(heading), math.sin(heading)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])     # nose-up positive
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _ray_box(o: np.ndarray, d: np.ndarray, centre: np.ndarray, half: np.ndarray):
    """Axis-aligned slab test in the local frame. Returns t (N,), normal (N, 3)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
    lo = (centre - half - o) * inv
    hi = (centre + half - o) * inv
    tmin = np.minimum(lo, hi)
    tmax = np.maximum(lo, hi)
    t_in = tmin.max(-1)
    t_out = tmax.min(-1)
    ok = (t_out >= np.maximum(t_in, 0.0))
    axis = tmin.argmax(-1)
    n = np.zeros_like(d)
    n[np.arange(len(d)), axis] = -np.sign(d[np.arange(len(d)), axis])
    return np.where(ok, t_in, np.inf), n


def _ray_cyl_y(o: np.ndarray, d: np.ndarray, centre: np.ndarray, r: float, half_w: float):
    """Cylinder with axis along local y (the wheel axle)."""
    ox, oz = o[:, 0] - centre[0], o[:, 2] - centre[2]
    dx, dz = d[:, 0], d[:, 2]
    a = dx * dx + dz * dz
    b = 2 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - r * r
    disc = b * b - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (-b - np.sqrt(np.clip(disc, 0, None))) / (2 * a)
    y = o[:, 1] + t * d[:, 1] - centre[1]
    ok = (disc > 0) & (t > 0) & (np.abs(y) <= half_w)
    n = np.stack([ox + t * dx, np.zeros_like(t), oz + t * dz], -1) / r
    # end caps
    with np.errstate(divide="ignore", invalid="ignore"):
        for sgn in (-1.0, 1.0):
            tc = (centre[1] + sgn * half_w - o[:, 1]) / d[:, 1]
            px = ox + tc * dx
            pz = oz + tc * dz
            cap = (tc > 0) & (px * px + pz * pz <= r * r) & (~ok | (tc < t))
            t = np.where(cap, tc, t)
            n = np.where(cap[:, None], np.array([0.0, sgn, 0.0])[None, :], n)
            ok |= cap
    return np.where(ok, t, np.inf), n


def intersect_rover(scene: Scene, o: np.ndarray, d: np.ndarray):
    """Returns t (N,), world normal (N,3), material id (N,) for the rover."""
    R = _rot(*scene.rover_att)
    base = scene.rover_pos + np.array([0, 0, WHEEL_R])   # axle height
    ol = (o[None, :] - base[None, :]) @ R                # world -> local (R orthonormal)
    ol = np.broadcast_to(ol, d.shape)
    dl = d @ R
    t_best = np.full(len(d), np.inf)
    n_best = np.zeros_like(d)
    mat = np.zeros(len(d), np.int64)

    def add(t, n, m):
        nonlocal t_best, n_best, mat
        better = t < t_best
        t_best = np.where(better, t, t_best)
        n_best = np.where(better[:, None], n, n_best)
        mat = np.where(better, m, mat)

    # chassis, deck, mast, solar panel
    add(*_ray_box(ol, dl, np.array([0.0, 0.0, 0.32 + 0.22]), np.array([0.85, 0.62, 0.22])), 1)
    add(*_ray_box(ol, dl, np.array([-0.2, 0.0, 0.90]), np.array([0.5, 0.55, 0.14])), 1)
    add(*_ray_box(ol, dl, np.array([0.55, 0.0, 1.45]), np.array([0.05, 0.05, 0.45])), 2)
    add(*_ray_box(ol, dl, np.array([0.55, 0.0, 1.95]), np.array([0.16, 0.22, 0.08])), 2)
    add(*_ray_box(ol, dl, np.array([-0.55, 0.0, 1.20]), np.array([0.32, 0.65, 0.03])), 3)
    for sx in (HALF_LEN, -HALF_LEN):
        for sy in (HALF_WID, -HALF_WID):
            add(*_ray_cyl_y(ol, dl, np.array([sx, sy, 0.0]), WHEEL_R, WHEEL_W / 2), 4)
            add(*_ray_box(ol, dl, np.array([sx, sy * 0.82, 0.25]), np.array([0.06, 0.18, 0.28])), 2)
    n_world = n_best @ R.T
    return t_best, n_world, mat


# --------------------------------------------------------------------------- shading
def _brdf_regolith(mu0: np.ndarray, mu: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Lommel-Seeliger with an opposition surge; returns radiance factor."""
    ls = mu0 / np.maximum(mu0 + mu, 1e-4)
    surge = 1.0 + 0.6 * np.exp(-phase / math.radians(6.0))
    return ALBEDO * ls * surge * 2.0


def _headlights(scene: Scene, p: np.ndarray, n: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Radiance factor from the two driving lights on terrain points p."""
    R = _rot(*scene.rover_att)
    base = scene.rover_pos + np.array([0, 0, WHEEL_R])
    tilt = math.radians(LAMP_TILT_DEG)
    bore_local = np.array([math.cos(tilt), 0.0, -math.sin(tilt)])
    bore = R @ bore_local
    cone = math.cos(math.radians(LAMP_CONE_DEG))
    out = np.zeros(len(p))
    mu = np.clip(-(n * d).sum(-1), 0.0, 1.0)
    for lp in LAMPS:
        pos = base + R @ np.array(lp)
        L = p - pos[None, :]
        dist = np.maximum(np.linalg.norm(L, axis=1), 0.3)
        Lh = L / dist[:, None]
        spot = np.clip((Lh @ bore - cone) / (1 - cone), 0.0, 1.0)
        spot = spot * spot * (3 - 2 * spot)
        mu0 = np.clip(-(n * Lh).sum(-1), 0.0, 1.0)
        phase = np.arccos(np.clip((-d * -Lh).sum(-1), -1, 1))
        out += _brdf_regolith(mu0, mu, phase) * spot * LAMP_I0 / (dist * dist)
    return out


def _stars(d: np.ndarray, seed: int) -> np.ndarray:
    """Sparse deterministic star field on the sky directions (two magnitude classes)."""
    s = np.int64((int(seed) * 2654435761) & 0x7FFFFFFFFFFFFFFF)
    out = np.zeros(d.shape[0])
    for scale, thresh, gain in ((420.0, 0.9994, 1.2), (900.0, 0.9990, 0.35)):
        q = np.floor(d * scale).astype(np.int64)
        hsh = (q[..., 0] * 73856093) ^ (q[..., 1] * 19349663) ^ (q[..., 2] * 83492791) ^ s
        u = ((hsh & 0xFFFFFF) / float(0xFFFFFF))
        out += np.where(u > thresh, 0.15 + gain * (u - thresh) / (1 - thresh), 0.0)
    return out


def _regolith_detail(xy: np.ndarray, seed: int, fade: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-octave regolith texture: (albedo factor, height-detail gradient (N, 2)).

    Octaves are faded with distance so the fine grain does not alias far away."""
    # (spatial frequency 1/m, height amplitude m): metre-scale undulation down to cm-scale grain
    octaves = ((0.35, 0.060), (1.6, 0.020), (6.5, 0.0055), (23.0, 0.0012))

    def field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(x)
        for k, (f, amp) in enumerate(octaves):
            sd = np.int64((seed * 31 + 7 * k) & 0x7FFFFFFF)
            acc += amp * value_noise(x * f, y * f, sd) * fade ** k
        return acc

    h0 = field(xy[:, 0], xy[:, 1])
    e = 0.02
    gx = (field(xy[:, 0] + e, xy[:, 1]) - h0) / e
    gy = (field(xy[:, 0], xy[:, 1] + e) - h0) / e
    albedo = 1.0 + 2.2 * h0 / octaves[0][1] * 0.08
    return albedo, np.stack([gx, gy], 1)


def _ambient_occlusion(scene: Scene, p: np.ndarray) -> np.ndarray:
    """Horizon-based occlusion of the sky hemisphere in [0, 1] (1 = fully open)."""
    occ = np.zeros(len(p))
    n_dir = 6
    for r in (0.7, 1.8, 4.5):
        for k in range(n_dir):
            a = 2 * math.pi * k / n_dir + 0.3 * r
            hgt = scene.height(p[:, 0] + r * math.cos(a), p[:, 1] + r * math.sin(a))
            hgt = np.where(np.isfinite(hgt), hgt, p[:, 2])
            occ += np.clip((hgt - p[:, 2]) / r, 0.0, 1.0)
    return 1.0 - 0.75 * occ / (3 * n_dir)


def _offset_path(path: np.ndarray, offset: float) -> np.ndarray:
    """Polyline shifted sideways by `offset` (left positive) - one wheel track."""
    if len(path) < 2:
        return path
    d = np.gradient(path, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    perp = np.stack([-d[:, 1], d[:, 0]], 1)
    return path + offset * perp


def _filmic(x: np.ndarray) -> np.ndarray:
    """ACES-style filmic tone curve (Narkowicz fit), keeps highlights from greying out."""
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def _path_mask(hit_xy: np.ndarray, path: np.ndarray, width: float, dashed: bool = False) -> np.ndarray:
    """Fraction of ribbon coverage for each hit point (0..1), chunked over segments."""
    out = np.zeros(len(hit_xy))
    if len(path) < 2:
        return out
    n_seg = min(len(path) - 1, 90)
    idx = np.linspace(0, len(path) - 1, n_seg + 1).astype(int)
    pts = path[idx]
    lo, hi = pts.min(0) - width, pts.max(0) + width
    cand = np.flatnonzero((hit_xy[:, 0] >= lo[0]) & (hit_xy[:, 0] <= hi[0])
                          & (hit_xy[:, 1] >= lo[1]) & (hit_xy[:, 1] <= hi[1]))
    if cand.size == 0:
        return out
    q = hit_xy[cand]
    best = np.full(len(q), np.inf)
    along = np.zeros(len(q))
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    for k in range(n_seg):
        a, b = pts[k], pts[k + 1]
        ab = b - a
        L2 = float(ab @ ab) + 1e-12
        u = np.clip(((q - a) @ ab) / L2, 0.0, 1.0)
        dx = q[:, 0] - (a[0] + u * ab[0])
        dy = q[:, 1] - (a[1] + u * ab[1])
        dist = np.sqrt(dx * dx + dy * dy)
        better = dist < best
        best = np.where(better, dist, best)
        along = np.where(better, cum[k] + u * math.sqrt(L2), along)
    m = np.clip(1.0 - (best - width * 0.6) / (width * 0.4), 0.0, 1.0)
    if dashed:
        m *= (np.mod(along, 1.0) < 0.55).astype(float)
    out[cand] = m
    return out


def render_frame(scene: Scene, camera: str = "chase", size: Tuple[int, int] = (1280, 720),
                 hud: bool = True, quality: int = 1, supersample: int = 1) -> Image.Image:
    """Render one frame.

    quality=1 full, 2 renders at half resolution and upsamples (interactive use).
    supersample=k renders at k x k the pixel count and downsamples with Lanczos
    (anti-aliased stills for documentation; cost grows as k^2)."""
    W, H = size
    ss = max(1, int(supersample))
    w, h = W * ss // quality, H * ss // quality
    o, D = camera_rays(scene, camera, w, h)
    d = D.reshape(-1, 3)
    N = d.shape[0]
    sun = scene.sun_dir
    tau = scene.tau_dust

    t_ter, hit_ter = march_terrain(scene, o, d)
    t_rov, n_rov, mat = intersect_rover(scene, o, d)
    hit_rov = np.isfinite(t_rov) & (t_rov < np.where(hit_ter, t_ter, np.inf))
    hit_ter = hit_ter & ~hit_rov                  # visible terrain only
    t = np.where(hit_rov, t_rov, np.where(hit_ter, t_ter, np.inf))
    hit = hit_rov | hit_ter
    p = o[None, :] + np.where(np.isfinite(t), t, 0.0)[:, None] * d

    # normals (+ regolith micro-relief bump on the terrain, faded with distance)
    n = np.zeros_like(d)
    n[hit_ter] = scene.normal(p[hit_ter, 0], p[hit_ter, 1])
    n[hit_rov] = n_rov[hit_rov]
    grain = np.ones(N)
    if hit_ter.any():
        fade = np.exp(-t[hit_ter] / 45.0)
        alb, grad = _regolith_detail(p[hit_ter, :2], scene.seed, fade)
        grain[hit_ter] = alb
        bump = 0.9 * fade[:, None] * grad
        nt = n[hit_ter].copy()
        nt[:, 0] -= bump[:, 0]
        nt[:, 1] -= bump[:, 1]
        n[hit_ter] = nt / np.linalg.norm(nt, axis=1, keepdims=True)
        ao = _ambient_occlusion(scene, p[hit_ter])
    else:
        ao = np.ones(0)
    # shadows for every hit point (rover included: it sits in terrain shadow too)
    sh = np.ones(N)
    if hit.any():
        sh[hit] = shadow(scene, p[hit] + n[hit] * 0.03)

    mu0 = np.clip(n @ sun, 0.0, 1.0)
    mu = np.clip(-(n * d).sum(-1), 0.0, 1.0)
    phase = np.arccos(np.clip((-d) @ sun, -1.0, 1.0))

    col = np.zeros((N, 3))
    reg = _brdf_regolith(mu0, mu, phase) * sh
    tint = np.array([1.0, 0.97, 0.92])
    col[hit_ter] = (reg[hit_ter] * grain[hit_ter])[:, None] * tint[None, :]

    # driving lights on the terrain (rover surfaces are excluded: no self-illumination)
    lamp = _headlights(scene, p[hit_ter], n[hit_ter], d[hit_ter])
    col[hit_ter] += (lamp * grain[hit_ter])[:, None] * np.array([1.0, 0.98, 0.94])

    # rover materials: 1 body (white/gold MLI), 2 structure (dark), 3 solar (blue-black), 4 wheels
    mats = {1: np.array([0.85, 0.80, 0.62]), 2: np.array([0.22, 0.22, 0.24]),
            3: np.array([0.10, 0.12, 0.28]), 4: np.array([0.30, 0.30, 0.30])}
    if hit_rov.any():
        Rr = _rot(*scene.rover_att)
        pl = (p[hit_rov] - (scene.rover_pos + np.array([0, 0, WHEEL_R]))[None, :]) @ Rr
        tex = np.ones(N)
        m_rov = mat[hit_rov]
        # MLI blanket seams on the body, cell grid on the solar array, tread on the wheels
        seam = (np.mod(pl[:, 2] + 0.02, 0.11) < 0.010) | (np.mod(pl[:, 1] + 0.05, 0.31) < 0.012)
        cell = (np.mod(pl[:, 0] + 0.03, 0.16) < 0.012) | (np.mod(pl[:, 1] + 0.03, 0.16) < 0.012)
        ang_w = np.arctan2(pl[:, 2], pl[:, 0] - np.round(pl[:, 0] / HALF_LEN) * HALF_LEN)
        tread = np.mod(ang_w, 2 * math.pi / 24) < (2 * math.pi / 24) * 0.35
        tex_r = np.where(m_rov == 1, np.where(seam, 0.72, 1.0), 1.0)
        tex_r = np.where(m_rov == 3, np.where(cell, 2.6, 1.0), tex_r)
        tex_r = np.where(m_rov == 4, np.where(tread, 0.7, 1.0), tex_r)
        tex[hit_rov] = tex_r
        for m_id, albedo in mats.items():
            sel = hit_rov & (mat == m_id)
            if sel.any():
                diff = albedo[None, :] * (mu0[sel] * sh[sel] * tex[sel])[:, None]
                hvec = sun[None, :] - d[sel]
                hvec /= np.linalg.norm(hvec, axis=1, keepdims=True)
                spec = np.clip((n[sel] * hvec).sum(1), 0, 1) ** (60 if m_id == 3 else 18) * sh[sel]
                col[sel] = diff + (0.6 if m_id in (1, 3) else 0.15) * spec[:, None]
    # ambient: earthshine (faintly blue) plus dust multiple scattering; occluded in hollows
    amb = 0.022 + 0.10 * tau
    amb_col = np.array([0.78, 0.86, 1.0]) * (1 - min(tau, 1.0) * 0.5) + tint * min(tau, 1.0) * 0.5
    amb_scale = np.full(N, 0.6)
    amb_scale[hit_ter] = ALBEDO * 2.5 * ao * grain[hit_ter]
    col[hit] += amb * amb_scale[hit, None] * amb_col[None, :]

    # wheel ruts (compacted, shadowed regolith) and painted overlays: true path (cyan),
    # VO estimate (amber, dashed), goal ring
    if hit_ter.any():
        hxy = p[hit_ter, :2]
        rut = np.maximum(_path_mask(hxy, _offset_path(scene.path_true, HALF_WID), WHEEL_W * 0.55),
                         _path_mask(hxy, _offset_path(scene.path_true, -HALF_WID), WHEEL_W * 0.55))
        col[hit_ter] *= (1.0 - 0.45 * rut)[:, None]
        m_true = _path_mask(hxy, scene.path_true, 0.12)
        m_est = _path_mask(hxy, scene.path_est, 0.09, dashed=True)
        gdist = np.linalg.norm(hxy - np.array([GOAL_X, 0.0]), axis=1)
        ring = np.clip(1.0 - np.abs(gdist - RV.GOAL_TOL) / 0.12, 0.0, 1.0)
        base = col[hit_ter]
        # overlays are emissive markers scaled to the exposure so they read the same in any lighting
        lvl = 0.55 * max(0.35, math.sqrt(max(sun[2], 0.02)) * 2.2) / 1.6
        base = base * (1 - 0.7 * m_true[:, None]) + (lvl * m_true)[:, None] * np.array([0.10, 0.75, 0.95])
        base = base * (1 - 0.7 * m_est[:, None]) + (lvl * m_est)[:, None] * np.array([1.0, 0.55, 0.10])
        base = base * (1 - 0.7 * ring[:, None]) + (lvl * ring)[:, None] * np.array([0.35, 1.0, 0.45])
        col[hit_ter] = base

    # goal beacon: additive vertical glow column above the goal
    ox, oy = o[0] - GOAL_X, o[1]
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (ox * d[:, 0] + oy * d[:, 1])
    c0 = ox * ox + oy * oy
    with np.errstate(invalid="ignore", divide="ignore"):
        tc = -b / (2 * a)
        dperp = np.sqrt(np.clip(c0 - b * b / (4 * a), 0, None))
    zc = o[2] + tc * d[:, 2]
    zg = float(scene.height(np.array([GOAL_X]), np.array([0.0]))[0])
    beacon = (tc > 0) & (tc < t) & (zc > zg) & (zc < zg + 4.0)
    glow = np.where(beacon, np.exp(-(dperp / 0.25) ** 2) * (1 - (zc - zg) / 4.0), 0.0)
    col += glow[:, None] * np.array([0.3, 1.0, 0.5]) * 0.7

    # sky: black, stars, sun disc + dust corona
    sky = ~hit
    cosang = np.clip(d @ sun, -1, 1)
    ang = np.arccos(cosang)
    star = _stars(d, scene.seed) * sky
    col += star[:, None] * np.array([0.9, 0.95, 1.0])
    disc = (ang < SUN_ANGULAR_RADIUS) & sky
    col[disc] = np.array([40.0, 38.0, 34.0])
    corona = (0.02 + 1.8 * tau) * np.exp(-ang / (0.03 + 0.25 * tau)) * sky
    col += corona[:, None] * np.array([1.0, 0.92, 0.80]) * 3.0

    # dust haze along the view ray: extinction + forward scattering
    dist = np.where(np.isfinite(t), t, 400.0)
    T = np.exp(-tau * dist / DUST_HAZE_LENGTH)
    fwd = 0.25 + 0.9 * np.exp(-ang / 0.6)
    haze = tau * fwd * np.clip(sun[2] * 12.0, 0.15, 1.0) * 0.35
    haze_col = np.array([0.86, 0.80, 0.70])
    col = col * T[:, None] + ((1 - T) * haze)[:, None] * haze_col[None, :]

    # lens glare when the sun is in frame (the perception 'glare' term made visible)
    in_frame = ang < math.radians(40)
    if in_frame.any():
        g = (0.12 * np.exp(-ang / 0.30) + 0.02 * np.exp(-ang / 0.6)) * np.clip(1 - ang / math.radians(40), 0, 1)
        col += (g * (0.5 + tau))[:, None] * np.array([1.0, 0.9, 0.75])

    # tone map + gamma: exposure set for a sunlit surface at this elevation, filmic curve,
    # gentle optical vignette
    exposure = 1.15 / max(0.35, math.sqrt(max(sun[2], 0.02)) * 2.2)
    col = _filmic(col * exposure)
    vx = (np.arange(w) + 0.5) / w * 2 - 1
    vy = (np.arange(h) + 0.5) / h * 2 - 1
    r2 = (vx[None, :] ** 2 + vy[:, None] ** 2).reshape(-1)
    col *= (1.0 - 0.18 * r2 * r2)[:, None]
    col = np.clip(col, 0, 1) ** (1 / 2.2)
    img = (col.reshape(h, w, 3) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(img, "RGB")
    if (w, h) != (W, H):
        im = im.resize((W, H), Image.LANCZOS if ss > 1 else Image.BICUBIC)
    if hud:
        _draw_hud(im, scene, camera)
    return im


# --------------------------------------------------------------------------- HUD
_LABELS = {"k_soil": "k_soil", "theta_r": "theta_r", "r_terrain": "r_terrain", "rho_rock": "rho_rock",
           "slope_deg": "slope", "sun_e": "sun_e", "sun_psi": "sun_psi", "tau_dust": "tau_dust"}


def _draw_hud(im: Image.Image, s: Scene, camera: str) -> None:
    W, H = im.size
    draw = ImageDraw.Draw(im, "RGBA")
    f_small = _font(True, max(11, H // 54))
    f_med = _font(False, max(13, H // 40))
    f_big = _font(False, max(16, H // 26))
    pad = W // 64
    lh = int(f_small.size * 1.45)

    # left panel: disturbance vector
    pw = int(W * 0.235)
    ph = pad + lh * (len(DIMS) + 2)
    draw.rectangle([pad, pad, pad + pw, pad + ph], fill=(6, 8, 12, 165), outline=(80, 90, 110, 200))
    draw.text((pad + 10, pad + 6), "DISTURBANCE  x ∈ R^8", font=f_med, fill=(200, 210, 230, 255))
    for k, d in enumerate(DIMS):
        v = s.params[d]
        txt = f"{_LABELS[d]:<10} {v:8.3f}  {UNITS[d]}"
        draw.text((pad + 10, pad + 10 + lh * (k + 1.4)), txt, font=f_small, fill=(230, 232, 236, 255))

    # right panel: telemetry
    tw = int(W * 0.235)
    rows = [("t", f"{s.t:6.1f} s"), ("dist→goal", f"{s.telemetry['dist_to_goal']:6.2f} m"),
            ("speed", f"{s.telemetry['v']:6.2f} m/s"), ("slip", f"{s.telemetry['slip']:6.2f}"),
            ("sinkage", f"{s.telemetry['sinkage'] * 100:6.1f} cm"),
            ("pitch/roll", f"{s.telemetry['pitch_deg']:5.1f}/{s.telemetry['roll_deg']:5.1f} °"),
            ("visibility", f"{s.telemetry['visibility']:6.2f}"), ("VO error", f"{s.telemetry['vo_error']:6.2f} m")]
    th = pad + lh * (len(rows) + 2)
    x0 = W - pad - tw
    draw.rectangle([x0, pad, W - pad, pad + th], fill=(6, 8, 12, 165), outline=(80, 90, 110, 200))
    draw.text((x0 + 10, pad + 6), "TELEMETRY", font=f_med, fill=(200, 210, 230, 255))
    for k, (a, b) in enumerate(rows):
        col = (230, 232, 236, 255)
        if a == "slip" and s.telemetry["slip"] > 0.6:
            col = (255, 120, 90, 255)
        if a == "pitch/roll" and max(abs(s.telemetry["pitch_deg"]), abs(s.telemetry["roll_deg"])) > 22:
            col = (255, 120, 90, 255)
        if a == "visibility" and s.telemetry["visibility"] < 0.35:
            col = (255, 120, 90, 255)
        draw.text((x0 + 10, pad + 10 + lh * (k + 1.4)), f"{a:<11}{b}", font=f_small, fill=col)

    # bottom bar: outcome + camera + seed
    bar_h = int(f_big.size * 2.0)
    draw.rectangle([0, H - bar_h, W, H], fill=(4, 5, 8, 190))
    oc = {"success": (90, 230, 120), "driving": (200, 205, 215), "stuck": (255, 150, 60),
          "tip_over": (255, 80, 80), "collision": (255, 90, 140), "nav_miss": (255, 210, 70),
          "timeout": (170, 170, 190)}[s.outcome]
    title = f"ASTRAEUS  ·  {s.outcome.upper()}"
    draw.text((pad, H - bar_h + bar_h // 2 - f_big.size // 2), title, font=f_big, fill=oc + (255,))
    # footer fields drop from the right until the line fits beside the title (narrow frames)
    fields = [f"seed {s.seed}", f"frame {s.frame + 1}/{s.n_frames}", f"cam {camera}", "software render (synthetic)"]
    avail = W - 3 * pad - draw.textlength(title, font=f_big)
    while len(fields) > 1 and draw.textlength("   ".join(fields), font=f_small) > avail:
        fields.pop()
    foot = "   ".join(fields)
    tw2 = draw.textlength(foot, font=f_small)
    draw.text((W - pad - tw2, H - bar_h + bar_h // 2 - f_small.size // 2), foot, font=f_small,
              fill=(170, 178, 192, 255))

    # legend
    lx = pad
    ly = H - bar_h - lh * 3.2
    for label, colr in (("true path", (26, 190, 242)), ("VO estimate", (255, 158, 26)), ("goal ring", (90, 255, 115))):
        draw.rectangle([lx, ly + 3, lx + 14, ly + 3 + lh * 0.55], fill=colr + (230,))
        draw.text((lx + 20, ly), label, font=f_small, fill=(220, 224, 232, 255))
        ly += lh


def draw_caption(im: Image.Image, s: Scene, title: str = "ASTRAEUS", subtitle: str = "") -> None:
    """Minimal lower-third for documentation stills: wordmark, outcome, and the
    disturbance vector on one line over a soft gradient. Use instead of the HUD."""
    W, H = im.size
    band = int(H * 0.26)
    grad = Image.new("L", (1, band))
    for y in range(band):
        grad.putpixel((0, y), int(225 * (y / band) ** 1.4))
    grad = grad.resize((W, band))
    dark = Image.new("RGBA", (W, band), (3, 4, 7, 255))
    dark.putalpha(grad)
    im.paste(dark, (0, H - band), dark)
    draw = ImageDraw.Draw(im, "RGBA")
    f_title = _font(False, max(18, H // 22))
    f_sub = _font(False, max(12, H // 46))
    f_mono = _font(True, max(11, H // 58))
    pad = W // 40
    y0 = H - band + int(band * 0.34)
    oc = {"success": (90, 230, 120), "driving": (200, 205, 215), "stuck": (255, 150, 60),
          "tip_over": (255, 80, 80), "collision": (255, 90, 140), "nav_miss": (255, 210, 70),
          "timeout": (170, 170, 190)}[s.outcome]
    draw.text((pad, y0), title, font=f_title, fill=(240, 242, 246, 255))
    tw = draw.textlength(title, font=f_title)
    draw.text((pad + tw + f_title.size * 0.6, y0 + f_title.size * 0.18), s.outcome.replace("_", " ").upper(),
              font=_font(False, max(14, H // 30)), fill=oc + (255,))
    if subtitle:
        draw.text((pad, y0 + f_title.size * 1.25), subtitle, font=f_sub, fill=(190, 198, 212, 255))
    xs = "   ".join(f"{_LABELS[d]} {s.params[d]:.3g}" for d in DIMS)
    xw = draw.textlength(xs, font=f_mono)
    draw.text((W - pad - xw, H - pad - f_mono.size * 1.1), xs, font=f_mono, fill=(160, 170, 188, 255))
    foot = f"seed {s.seed} · t = {s.t:.1f} s · deterministic software render"
    draw.text((W - pad - draw.textlength(foot, font=f_mono), H - pad - f_mono.size * 2.5), foot,
              font=f_mono, fill=(120, 130, 150, 255))


# --------------------------------------------------------------------------- helpers
CAMERAS = ("chase", "rover_cam", "overview", "orbit", "cinematic")


def to_png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def render_episode(res: RV.EpisodeResult, camera: str = "chase", frame: int = -1,
                   size: Tuple[int, int] = (1280, 720), quality: int = 1, hud: bool = True,
                   supersample: int = 1) -> Image.Image:
    return render_frame(Scene.from_result(res, frame), camera, size, hud=hud, quality=quality,
                        supersample=supersample)


def render_contact_sheet(res: RV.EpisodeResult, n: int = 6, size: Tuple[int, int] = (640, 360),
                         camera: str = "chase", quality: int = 1) -> Image.Image:
    """n frames through the episode, tiled 3 wide."""
    assert res.trace is not None
    T = res.trace["x"].shape[1]
    frames = np.linspace(0, T - 1, n).astype(int)
    cols = 3
    rows = int(math.ceil(n / cols))
    sheet = Image.new("RGB", (size[0] * cols, size[1] * rows), (0, 0, 0))
    for k, fr in enumerate(frames):
        im = render_episode(res, camera, int(fr), size, quality=quality)
        sheet.paste(im, ((k % cols) * size[0], (k // cols) * size[1]))
    return sheet
