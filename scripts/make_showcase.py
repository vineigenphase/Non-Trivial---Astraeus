"""Generate the README showcase assets (hero still, outcome gallery, animated demo)
from a saved campaign with the local software renderer.

    python scripts/make_showcase.py runs/val.npz            # everything (~10 min on a laptop CPU)
    python scripts/make_showcase.py runs/val.npz --only hero gallery
    python scripts/make_showcase.py runs/val.npz --hero-episode 111

Every asset is a deterministic function of (campaign file, episode index): the
script prints the index and seed behind each file so the README can cite them.

Selection is purely photogenic, not statistical: the hero is the highest-weight
elite under --prior whose sun is low but not grazing and whose sky is reasonably
clear, so the raking light shows terrain relief.  Failure probabilities in the
README come from `astraeus report`, never from these picks.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astraeus import campaign as C  # noqa: E402
from astraeus import priors as PR  # noqa: E402
from astraeus import render as RD  # noqa: E402
from astraeus import rover as RV  # noqa: E402

SUBTITLE = "adversarial stress testing of off-world rover autonomy under unknown priors"
GALLERY = (  # (outcome, camera, caption)
    ("stuck", "cinematic", "wheel sinkage exceeds traction on soft, sloped regolith"),
    ("tip_over", "chase", "static tip-over on crater rim relief"),
    ("collision", "rover_cam", "rock not resolved in low-sun glare and dust"),
    ("nav_miss", "overview", "visual-odometry drift misses the goal ring"),
    ("success", "cinematic", "nominal traverse under a benign draw"),
)


def photogenic(c: C.Campaign, idx: Sequence[int]) -> List[int]:
    """Keep episodes whose draw lights the terrain well: low raking sun, clear sky."""
    X = c.X
    keep = []
    for i in idx:
        x = dict(zip(PR.DIMS, X[i]))
        if 2.2 <= x["sun_e"] <= 5.8 and x["tau_dust"] <= 0.15 and x["rho_rock"] >= 3.0:
            keep.append(int(i))
    return keep


def pick(c: C.Campaign, outcome: str, prior: PR.Prior) -> int:
    code = RV.OUTCOMES.index(outcome)
    if outcome == "success":
        cand = np.flatnonzero((c.outcome == code) & (c.source == 0))
        good = photogenic(c, cand)
        if not good:
            raise SystemExit("no successful proposal episode in the run")
        return good[0]
    ranked = [i for i in c.elites(k=c.n, prior=prior) if c.outcome[i] == code]
    good = photogenic(c, ranked)
    if good:
        return good[0]
    if ranked:
        return int(ranked[0])
    raise SystemExit(f"no {outcome} episode in the run")


def still(res: RV.EpisodeResult, camera: str, size, ss: int, frame: int = -1, subtitle: str = SUBTITLE) -> Image.Image:
    scene = RD.Scene.from_result(res, frame)
    im = RD.render_frame(scene, camera, size, hud=False, supersample=ss)
    RD.draw_caption(im, scene, subtitle=subtitle)
    return im


def animate(res: RV.EpisodeResult, camera: str, size, n_frames: int, out: Path, fps: int = 12) -> None:
    assert res.trace is not None
    T = res.trace["x"].shape[1]
    frames = []
    for k, fr in enumerate(np.linspace(0, T - 1, n_frames).astype(int)):
        scene = RD.Scene.from_result(res, int(fr))
        im = RD.render_frame(scene, camera, size, hud=False)
        RD.draw_caption(im, scene, subtitle=f"frame {k + 1}/{n_frames}  ·  t = {scene.t:.1f} s")
        frames.append(im)
    hold = [frames[-1]] * fps                       # linger on the terminal frame
    frames[0].save(out, save_all=True, append_images=frames[1:] + hold, duration=int(1000 / fps),
                   loop=0, quality=82, method=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--prior", default="P2", choices=list(PR.PRIORS))
    ap.add_argument("--hero-episode", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=("hero", "gallery", "demo"), choices=("hero", "gallery", "demo"))
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "img")
    a = ap.parse_args()

    c = C.Campaign.load(a.run)
    prior = PR.PRIORS[a.prior]
    a.out.mkdir(parents=True, exist_ok=True)

    hero_i = a.hero_episode if a.hero_episode is not None else pick(c, "stuck", prior)
    hero = c.replay(hero_i)

    if "hero" in a.only:
        t0 = time.perf_counter()
        still(hero, "cinematic", (1920, 1080), a.supersample).save(a.out / "hero.png", optimize=True)
        print(f"hero.png          episode #{hero_i} seed {int(c.seeds[hero_i])}  {time.perf_counter() - t0:.0f}s", flush=True)

    if "gallery" in a.only:
        for outcome, camera, caption in GALLERY:
            i = hero_i if outcome == "stuck" else pick(c, outcome, prior)
            t0 = time.perf_counter()
            res = hero if i == hero_i else c.replay(i)
            still(res, camera, (1280, 720), a.supersample, subtitle=caption).save(
                a.out / f"gallery_{outcome}.png", optimize=True)
            print(f"gallery_{outcome + '.png':<14}episode #{i} seed {int(c.seeds[i])}  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)

    if "demo" in a.only:
        t0 = time.perf_counter()
        animate(hero, "chase", (800, 450), 40, a.out / "demo.webp")
        print(f"demo.webp         episode #{hero_i} chase, 40 frames  {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
