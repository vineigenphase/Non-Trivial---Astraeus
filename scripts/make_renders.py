"""Regenerate the documentation renders in docs/img from a saved campaign.

    python scripts/make_renders.py runs/val.npz [--prior P1] [--quality 3]

Picks, for each terminal outcome, the most plausible elite under the chosen
prior (the first success by index), renders one frame per outcome from a fixed
camera and a six-frame contact sheet for the stuck case, and prints the
episode index behind every file so docs/VALIDATION.md can cite it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astraeus import campaign as C  # noqa: E402
from astraeus import priors as PR  # noqa: E402
from astraeus import render as RD  # noqa: E402
from astraeus import rover as RV  # noqa: E402

SHOTS = (  # (outcome, camera, file stem)
    ("stuck", "chase", "stuck_chase"),
    ("tip_over", "chase", "tip_over_chase"),
    ("collision", "rover_cam", "collision_rover_cam"),
    ("nav_miss", "overview", "nav_miss_overview"),
    ("success", "orbit", "success_orbit"),
)


def pick(c: C.Campaign, outcome: str, prior: PR.Prior) -> int:
    code = RV.OUTCOMES.index(outcome)
    if outcome == "success":
        idx = np.flatnonzero((c.outcome == code) & (c.source == 0))
        if len(idx) == 0:
            raise SystemExit("no successful proposal episode in the run")
        return int(idx[0])
    for i in c.elites(k=c.n, prior=prior):
        if c.outcome[i] == code:
            return i
    raise SystemExit(f"no {outcome} episode in the run")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--prior", default="P1", choices=list(PR.PRIORS))
    ap.add_argument("--quality", type=int, default=3, choices=(1, 2, 3))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "img")
    a = ap.parse_args()

    c = C.Campaign.load(a.run)
    prior = PR.PRIORS[a.prior]
    a.out.mkdir(parents=True, exist_ok=True)
    for outcome, camera, stem in SHOTS:
        i = pick(c, outcome, prior)
        t0 = time.perf_counter()
        res = c.replay(i)
        RD.render_episode(res, camera=camera, frame=-1, size=(1280, 720), quality=a.quality).save(a.out / f"{stem}.png")
        print(f"{stem}.png  episode #{i} {outcome} seed {int(c.seeds[i])}  {time.perf_counter() - t0:.1f}s")
        if outcome == "stuck":
            RD.render_contact_sheet(res, n=6, size=(640, 360), quality=a.quality).save(a.out / f"{stem.split('_')[0]}_sheet.png")
            print(f"stuck_sheet.png  episode #{i}")


if __name__ == "__main__":
    main()
