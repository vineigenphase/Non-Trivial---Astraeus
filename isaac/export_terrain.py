#!/usr/bin/env python3
"""Export an Astraeus episode's terrain (heightfield, OBJ, rocks, craters) for Isaac Sim.

    python isaac/export_terrain.py --x slope_deg=9 rho_rock=6 --seed 17 --out isaac/out/terrain
    python isaac/export_terrain.py --run runs/c1.npz --episode 379 --out isaac/out/terrain

The result is the exact terrain the local simulator/renderer used for that
(x, seed), so Isaac replays and local replays are on the same ground.
No Isaac Sim needed; plain numpy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "isaac"))

from astraeus import priors as PR                 # noqa: E402
from astraeus_isaac.export import export_episode  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", nargs="*", default=[], help="DIM=VALUE overrides on the nominal polar point")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", help="campaign .npz to take (x, seed) from")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--res", type=float, default=0.10)
    ap.add_argument("--out", default=str(REPO / "isaac/out/terrain"))
    a = ap.parse_args(argv)

    if a.run:
        z = np.load(a.run, allow_pickle=False)
        x, seed = z["X"][a.episode], int(z["seeds"][a.episode])
    else:
        x = np.array([PR.NOMINAL_X[d] for d in PR.DIMS])
        for it in a.x:
            k, v = it.split("=")
            x[PR.DIMS.index(k)] = float(v)
        seed = a.seed
    files = export_episode(x, seed, Path(a.out), res=a.res)
    print(json.dumps(files, indent=1))


if __name__ == "__main__":
    main()
