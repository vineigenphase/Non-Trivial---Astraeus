"""Procedural lunar terrain, evaluated analytically and vectorised across episodes.

Every episode owns its own terrain (seed, relief scale, regional slope, crater
and rock population), yet a campaign simulates thousands of episodes in
lockstep. A gridded DEM per episode would make that memory- and cache-bound, so
the heightfield is an *analytic* function

    h(x, y) = slope plane + r_terrain * Σ_k A_k · noise_k(x, y; seed)
              + Σ craters(x, y) + Σ rocks(x, y)

that can be queried at arbitrary (episode, point) pairs with one numpy call.
The same function is rasterised into a DEM only for rendering and for the
Isaac Sim exporter (isaac/build_terrain_usd.py), so the picture on screen and
the surface the rover drove are the same surface.

Value noise is hashed on an integer lattice (no tables, no state), smoothstep-
interpolated, with a per-(episode, octave) lattice offset so episodes are
decorrelated. Amplitudes follow a ~1/f^1.2 spectrum, the slope of lunar
highland DEM power spectra at metre scales (see docs/RESEARCH.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# Corridor the rover drives: start at the origin, goal on +x.
GOAL_X = 30.0
CORRIDOR_HALF_WIDTH = 8.0            # rocks / craters are placed within +-this of the axis
X_MIN, X_MAX = -6.0, GOAL_X + 8.0

# Fractal octaves: (wavelength m, nominal amplitude m at r_terrain = 1)
OCTAVES = ((24.0, 0.45), (12.0, 0.24), (6.0, 0.13), (3.0, 0.065),
           (1.5, 0.032), (0.75, 0.014))
N_CRATERS = 4
N_ROCK_SLOTS = 64                    # capacity per episode; count ~ Poisson(rho * area/100)
ROCK_AREA_M2 = (X_MAX - X_MIN) * 2 * CORRIDOR_HALF_WIDTH
ROCK_D_MIN, ROCK_D_MAX = 0.20, 1.60  # m; power-law size-frequency (exponent -2.5)
ROCK_BURIAL = 0.35                   # fraction of the rock radius below grade


def _hash01(ix: np.ndarray, iy: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Deterministic uniform [0,1) from integer lattice coordinates and a seed."""
    ix = ix.astype(np.uint32)
    iy = iy.astype(np.uint32)
    s = seed.astype(np.uint32)
    h = ix * np.uint32(0x8DA6B343) ^ iy * np.uint32(0xD8163841) ^ s * np.uint32(0xCB1AB31F)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(12)
    h *= np.uint32(0x297A2D39)
    h ^= h >> np.uint32(15)
    return h.astype(np.float64) * (1.0 / 4294967296.0)


def episode_uniforms(seed: np.ndarray, k: int, stream: int = 0) -> np.ndarray:
    """(n, k) deterministic uniforms per episode. Independent of batch
    composition, so a single episode can be re-simulated bit-for-bit later."""
    seed = np.asarray(seed, np.int64)
    j = np.arange(k, dtype=np.int64)[None, :]
    return _hash01(j + stream * 1_000_003, np.full_like(j, 977), seed[:, None])


def episode_normals(seed: np.ndarray, k: int, stream: int = 0) -> np.ndarray:
    """(n, k) deterministic standard normals (Box-Muller on two hash streams)."""
    u1 = np.clip(episode_uniforms(seed, k, stream), 1e-12, 1.0)
    u2 = episode_uniforms(seed, k, stream + 7)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)


def _poisson_from_uniform(lam: np.ndarray, u: np.ndarray, kmax: int) -> np.ndarray:
    """Inverse-CDF Poisson, vectorised, truncated at kmax."""
    lam = np.asarray(lam, float)
    p = np.exp(-lam)
    cdf = p.copy()
    k = np.zeros_like(lam, dtype=np.int64)
    for i in range(1, kmax + 1):
        k = np.where(u > cdf, i, k)
        p = p * lam / i
        cdf = cdf + p
    return np.minimum(k, kmax)


def _smooth(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def value_noise(x: np.ndarray, y: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Lattice value noise in [-1, 1]. x, y, seed broadcast together."""
    x0 = np.floor(x)
    y0 = np.floor(y)
    tx = _smooth(x - x0)
    ty = _smooth(y - y0)
    ix = x0.astype(np.int64)
    iy = y0.astype(np.int64)
    seed = np.broadcast_to(seed, ix.shape)
    a = _hash01(ix, iy, seed)
    b = _hash01(ix + 1, iy, seed)
    c = _hash01(ix, iy + 1, seed)
    d = _hash01(ix + 1, iy + 1, seed)
    top = a + (b - a) * tx
    bot = c + (d - c) * tx
    return 2.0 * (top + (bot - top) * ty) - 1.0


@dataclass
class TerrainBatch:
    """Terrain parameters for n episodes (all arrays have leading dim n)."""
    seed: np.ndarray            # (n,) int
    r_terrain: np.ndarray       # (n,)
    slope: np.ndarray           # (n,) radians
    slope_dir: np.ndarray       # (n,) radians, downhill direction
    crater_xy: np.ndarray       # (n, C, 2)
    crater_R: np.ndarray        # (n, C)
    crater_depth: np.ndarray    # (n, C)
    rock_xy: np.ndarray         # (n, M, 2)
    rock_r: np.ndarray          # (n, M) radius, 0 for inactive slots
    rock_active: np.ndarray     # (n, M) bool
    rock_count: np.ndarray      # (n,) int

    @property
    def n(self) -> int:
        return int(self.seed.shape[0])

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, seed: np.ndarray, r_terrain: np.ndarray, slope_deg: np.ndarray,
              rho_rock: np.ndarray) -> "TerrainBatch":
        """Deterministic in (seed, parameters): no external RNG is consumed."""
        seed = np.asarray(seed, np.int64)
        r_terrain = np.asarray(r_terrain, float)
        slope = np.radians(np.asarray(slope_deg, float))
        U = episode_uniforms(seed, 1 + 4 * N_CRATERS + 1 + 3 * N_ROCK_SLOTS, stream=1)
        col = [0]

        def take(k):
            a = U[:, col[0]:col[0] + k]
            col[0] += k
            return a

        def uni(lo, hi, k):
            return lo + (hi - lo) * take(k)

        slope_dir = uni(0.0, 2 * np.pi, 1)[:, 0]

        # craters — kept off the first 4 m so the rover always starts on grade
        cx = uni(4.0, X_MAX - 2.0, N_CRATERS)
        cy = uni(-CORRIDOR_HALF_WIDTH, CORRIDOR_HALF_WIDTH, N_CRATERS)
        R = uni(1.5, 5.0, N_CRATERS)
        # depth/diameter: fresh simple craters ~0.10, degraded metre-scale ones far shallower
        dd = uni(0.02, 0.10, N_CRATERS)
        depth = dd * 2 * R * r_terrain[:, None]

        # rocks — Poisson count from areal density, power-law diameters
        lam = np.asarray(rho_rock, float) * ROCK_AREA_M2 / 100.0
        count = _poisson_from_uniform(lam, take(1)[:, 0], N_ROCK_SLOTS)
        slot = np.arange(N_ROCK_SLOTS)[None, :]
        active = slot < count[:, None]
        rx = uni(2.5, X_MAX - 1.0, N_ROCK_SLOTS)
        ry = uni(-CORRIDOR_HALF_WIDTH, CORRIDOR_HALF_WIDTH, N_ROCK_SLOTS)
        u = take(N_ROCK_SLOTS)
        # inverse-CDF of a truncated power law N(>D) ∝ D^-2.5
        a = ROCK_D_MIN ** -1.5
        b = ROCK_D_MAX ** -1.5
        D = (a - u * (a - b)) ** (-1.0 / 1.5)
        rr = np.where(active, 0.5 * D, 0.0)
        return cls(seed, r_terrain, slope, slope_dir,
                   np.stack([cx, cy], -1), R, depth,
                   np.stack([rx, ry], -1), rr, active, count)

    # ------------------------------------------------------------------ query
    def _bcast(self, x: np.ndarray, arr: np.ndarray) -> np.ndarray:
        """Reshape a per-episode (n,) array so it broadcasts against x (n, ...)."""
        return arr.reshape((self.n,) + (1,) * (x.ndim - 1))

    def height(self, x: np.ndarray, y: np.ndarray, rocks: bool = True) -> np.ndarray:
        """Height (m) at points x, y of shape (n, ...) — one episode per row."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        h = np.zeros(np.broadcast(x, y).shape)

        # regional slope plane: height falls in the downhill direction
        sl = self._bcast(x, np.tan(self.slope))
        d = self._bcast(x, self.slope_dir)
        h -= sl * (x * np.cos(d) + y * np.sin(d))

        # fractal relief
        rt = self._bcast(x, self.r_terrain)
        for k, (lam, amp) in enumerate(OCTAVES):
            seed_k = self._bcast(x, self.seed * 7919 + k * 104729)
            h += rt * amp * value_noise(x / lam, y / lam, seed_k)

        # craters: parabolic bowl + gaussian rim
        for c in range(N_CRATERS):
            cx = self._bcast(x, self.crater_xy[:, c, 0])
            cy = self._bcast(x, self.crater_xy[:, c, 1])
            R = self._bcast(x, self.crater_R[:, c])
            dep = self._bcast(x, self.crater_depth[:, c])
            r = np.hypot(x - cx, y - cy) / R
            bowl = -dep * np.clip(1.0 - r * r, 0.0, None)
            rim = 0.18 * dep * np.exp(-((r - 1.0) / 0.22) ** 2)
            h += bowl + rim

        if rocks:
            h += self.rock_height(x, y)
        return h

    def rock_height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Contribution of partially buried hemispherical rocks."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        shp = np.broadcast(x, y).shape
        xr = x.reshape(shp + (1,))
        yr = y.reshape(shp + (1,))
        rx = self.rock_xy[:, :, 0].reshape((self.n,) + (1,) * (len(shp) - 1) + (N_ROCK_SLOTS,))
        ry = self.rock_xy[:, :, 1].reshape(rx.shape)
        rr = self.rock_r.reshape(rx.shape)
        d2 = (xr - rx) ** 2 + (yr - ry) ** 2
        cap = np.sqrt(np.clip(rr * rr - d2, 0.0, None)) - ROCK_BURIAL * rr
        return np.clip(cap, 0.0, None).sum(-1)

    def normal_angles(self, x: np.ndarray, y: np.ndarray, heading: np.ndarray,
                      half_len: float, half_wid: float, rocks: bool = False) -> tuple:
        """Pitch (nose-up positive) and roll (right-side-down positive) of a
        rigid footprint of the given half-dimensions centred at (x, y) with
        the given heading. Returns (pitch, roll) in radians, plus the four
        corner heights (n, 4). Rocks are excluded by default: their effect on
        the body is handled as wheel bumps / collisions in rover.py."""
        c, s = np.cos(heading), np.sin(heading)
        # corners: front-left, front-right, rear-left, rear-right
        dx = np.stack([half_len * c - half_wid * s, half_len * c + half_wid * s,
                       -half_len * c - half_wid * s, -half_len * c + half_wid * s], -1)
        dy = np.stack([half_len * s + half_wid * c, half_len * s - half_wid * c,
                       -half_len * s + half_wid * c, -half_len * s - half_wid * c], -1)
        hz = self.height(x[:, None] + dx, y[:, None] + dy, rocks=rocks)
        front = 0.5 * (hz[:, 0] + hz[:, 1])
        rear = 0.5 * (hz[:, 2] + hz[:, 3])
        left = 0.5 * (hz[:, 0] + hz[:, 2])
        right = 0.5 * (hz[:, 1] + hz[:, 3])
        pitch = np.arctan2(front - rear, 2 * half_len)
        roll = np.arctan2(left - right, 2 * half_wid)
        return pitch, roll, hz

    # ------------------------------------------------------------- rasterise
    def dem(self, i: int, res: float = 0.25, x_range=(X_MIN, X_MAX),
            y_range=(-CORRIDOR_HALF_WIDTH, CORRIDOR_HALF_WIDTH)) -> Dict[str, np.ndarray]:
        """Rasterise episode i into a DEM for rendering / export."""
        xs = np.arange(x_range[0], x_range[1] + 1e-9, res)
        ys = np.arange(y_range[0], y_range[1] + 1e-9, res)
        X, Y = np.meshgrid(xs, ys)
        sub = self.select([i])
        Z = sub.height(X[None], Y[None])[0]
        return {"x": xs, "y": ys, "z": Z}

    def select(self, idx) -> "TerrainBatch":
        idx = np.asarray(idx)
        return TerrainBatch(self.seed[idx], self.r_terrain[idx], self.slope[idx],
                            self.slope_dir[idx], self.crater_xy[idx], self.crater_R[idx],
                            self.crater_depth[idx], self.rock_xy[idx], self.rock_r[idx],
                            self.rock_active[idx], self.rock_count[idx])

    def rocks_of(self, i: int) -> list:
        m = self.rock_active[i]
        return [{"x": float(px), "y": float(py), "r": float(r)}
                for (px, py), r in zip(self.rock_xy[i][m], self.rock_r[i][m])]

    def craters_of(self, i: int) -> list:
        return [{"x": float(px), "y": float(py), "R": float(R), "depth": float(d)}
                for (px, py), R, d in zip(self.crater_xy[i], self.crater_R[i], self.crater_depth[i])]
