"""Export an Astraeus episode's terrain for Isaac Sim (or any DCC tool).

Given the same (x, seed) the local simulator uses, produce
  * a heightfield .npy  (rows = y, cols = x, metres)
  * an OBJ mesh with normals and a planar UV, in metres, z-up
  * a rocks .json  (x, y, radius, burial) for spawning sphere/mesh prims
  * a craters .json
so the Isaac scene is *the same terrain* the renderer and the campaign saw.
Requires the `astraeus` package on sys.path (it is, when run from the repo).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from astraeus import terrain as TR
from astraeus.priors import DIMS

ROCK_BURIAL = TR.ROCK_BURIAL


def episode_terrain(x: Sequence[float], seed: int) -> TR.TerrainBatch:
    d = dict(zip(DIMS, (float(v) for v in x)))
    return TR.TerrainBatch.build(np.array([seed]), np.array([d["r_terrain"]]),
                                 np.array([d["slope_deg"]]), np.array([d["rho_rock"]]))


def heightfield(t: TR.TerrainBatch, res: float = 0.10, margin: float = 6.0) -> Dict[str, np.ndarray]:
    return t.dem(0, res=res, x_range=(TR.X_MIN - margin, TR.X_MAX + margin),
                 y_range=(-TR.CORRIDOR_HALF_WIDTH - margin, TR.CORRIDOR_HALF_WIDTH + margin))


def write_obj(path: Path, dem: Dict[str, np.ndarray]) -> None:
    xs, ys, Z = dem["x"], dem["y"], dem["z"]
    ny, nx = Z.shape
    gy, gx = np.gradient(Z, ys, xs)
    N = np.stack([-gx, -gy, np.ones_like(Z)], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    with open(path, "w") as f:
        f.write("# Astraeus terrain export, metres, z-up\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"v {xs[i]:.4f} {ys[j]:.4f} {Z[j, i]:.4f}\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"vt {i / (nx - 1):.5f} {j / (ny - 1):.5f}\n")
        for j in range(ny):
            for i in range(nx):
                n = N[j, i]
                f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i + 1
                b, c, d = a + 1, a + nx + 1, a + nx
                f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
                f.write(f"f {a}/{a}/{a} {c}/{c}/{c} {d}/{d}/{d}\n")


def export_episode(x: Sequence[float], seed: int, out_dir: Path, res: float = 0.10) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = episode_terrain(x, seed)
    dem = heightfield(t, res=res)
    # rock heights above local grade, for spawning: hemisphere of radius r buried by ROCK_BURIAL*r
    rocks = t.rocks_of(0)
    rx = np.array([r["x"] for r in rocks]) if rocks else np.zeros(0)
    ry = np.array([r["y"] for r in rocks]) if rocks else np.zeros(0)
    if len(rocks):
        gz = t.height(rx[None, :], ry[None, :], rocks=False)[0]
        for r, z in zip(rocks, gz):
            r["z_ground"] = float(z)
            r["burial"] = ROCK_BURIAL
    stem = f"ep_seed{seed}"
    files = {
        "heightfield": out_dir / f"{stem}_heightfield.npy",
        "obj": out_dir / f"{stem}_terrain.obj",
        "rocks": out_dir / f"{stem}_rocks.json",
        "craters": out_dir / f"{stem}_craters.json",
        "meta": out_dir / f"{stem}_meta.json",
    }
    np.save(files["heightfield"], dem["z"].astype(np.float32))
    write_obj(files["obj"], dem)
    files["rocks"].write_text(json.dumps(rocks, indent=1))
    files["craters"].write_text(json.dumps(t.craters_of(0), indent=1))
    files["meta"].write_text(json.dumps({
        "x": dict(zip(DIMS, (float(v) for v in x))), "seed": int(seed), "res_m": res,
        "x0": float(dem["x"][0]), "y0": float(dem["y"][0]), "nx": int(len(dem["x"])), "ny": int(len(dem["y"])),
        "goal": [TR.GOAL_X, 0.0], "start": [0.0, 0.0],
    }, indent=1))
    return {k: str(v) for k, v in files.items()}
