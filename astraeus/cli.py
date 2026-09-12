"""Command-line front end.

    python -m astraeus run      --n 4000 --out runs/c1.npz     # campaign under q, report for P1..P6
    python -m astraeus cem      runs/c1.npz --prior P2         # adversarial search, appended to the run
    python -m astraeus report   runs/c1.npz                    # text + JSON summary
    python -m astraeus render   runs/c1.npz --episode 17       # PNG frames / contact sheet
    python -m astraeus whatif   --x slope_deg=9 k_soil=0.8     # one deterministic episode
    python -m astraeus ingest   runs/c1.npz isaac/out/log.csv  # fold Isaac / OmniLRS episodes in
    python -m astraeus serve    --port 8000                    # the live world model
    python -m astraeus selftest                                # invariants, ~20 s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from . import campaign as C
from . import priors as PR
from . import render as RD
from . import rover as RV


def _progress(step: int, steps: int, done: int, total: int) -> None:
    sys.stderr.write(f"\r  step {step}/{steps}  finished {done}/{total}")
    if done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _parse_x(items) -> np.ndarray:
    x = np.array([PR.NOMINAL_X[d] for d in PR.DIMS], float)
    for it in items or []:
        k, v = it.split("=")
        if k not in PR.DIMS:
            raise SystemExit(f"unknown dimension {k!r}; choose from {PR.DIMS}")
        x[PR.DIMS.index(k)] = float(v)
    return PR.clip_to_bounds(x[None, :])[0]


def cmd_run(a) -> None:
    c = C.Campaign.load(a.out) if a.resume and Path(a.out).exists() else C.Campaign(seed=a.seed, gt_pose=a.gt_pose)
    print(f"Astraeus {__version__} — {a.n} episodes from q, seed {a.seed}, gt_pose={a.gt_pose}")
    c.run_chunk(a.n, progress=_progress)
    c.save(a.out)
    C.write_summary_txt(c, Path(a.out).with_suffix(".txt"))
    print(Path(a.out).with_suffix(".txt").read_text())
    print(f"saved {a.out}  ({c.n} episodes, {c.summary()['episodes_per_s']} ep/s)")


def cmd_cem(a) -> None:
    c = C.Campaign.load(a.run)
    prior = PR.PRIORS[a.prior]
    print(f"CEM under {prior.name} ({prior.label}): {a.iters} x {a.batch}")
    for it in c.cem_search(prior, iters=a.iters, batch=a.batch):
        b = it["best"]
        print(f"  iter {it['iter']}: fail {it['fail_rate']:.2f}  best J={b['J']:.3f} {b['outcome']} "
              f"#{b['index']}  x={json.dumps({k: round(v, 3) for k, v in b['x'].items()})}")
    c.save(a.run)


def cmd_report(a) -> None:
    c = C.Campaign.load(a.run)
    s = c.summary()
    if a.json:
        print(json.dumps(s, indent=2, default=float))
        return
    C.write_summary_txt(c, Path(a.run).with_suffix(".txt"))
    print(Path(a.run).with_suffix(".txt").read_text())
    for name, p in PR.PRIORS.items():
        el = c.elites(k=5, prior=p)
        print(f"{name} elites: " + ", ".join(f"#{i} {RV.OUTCOMES[int(c.outcome[i])]}" for i in el))


def cmd_render(a) -> None:
    c = C.Campaign.load(a.run)
    idx = [a.episode] if a.episode is not None else c.elites(k=a.k, prior=PR.PRIORS[a.prior])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in idx:
        res = c.replay(i)
        t0 = time.perf_counter()
        if a.sheet:
            img = RD.render_contact_sheet(res, n=6, size=(a.width, a.height), quality=a.quality)
            p = out / f"ep{i:05d}_sheet.png"
        else:
            img = RD.render_episode(res, camera=a.camera, frame=a.frame, size=(a.width, a.height), quality=a.quality)
            p = out / f"ep{i:05d}_{a.camera}_f{a.frame}.png"
        img.save(p)
        print(f"  {p}  {RV.OUTCOMES[int(res.outcome[0])]:<10} {time.perf_counter() - t0:.1f}s")


def cmd_whatif(a) -> None:
    x = _parse_x(a.x)
    res = RV.simulate_one(x, a.seed, gt_pose=a.gt_pose)
    print(json.dumps(res.row(0), indent=2))
    if a.png:
        RD.render_episode(res, camera=a.camera, frame=-1, size=(1280, 720), quality=1).save(a.png)
        print(f"wrote {a.png}")


def cmd_ingest(a) -> None:
    c = C.Campaign.load(a.run) if Path(a.run).exists() else C.Campaign(seed=0)
    n = c.ingest_csv(a.csv, source=0 if a.from_proposal else 2)
    c.save(a.run)
    print(f"ingested {n} rows from {a.csv} -> {a.run} ({c.n} episodes total)")
    C.write_summary_txt(c, Path(a.run).with_suffix(".txt"))
    print(Path(a.run).with_suffix(".txt").read_text())


def cmd_serve(a) -> None:
    import uvicorn
    uvicorn.run("astraeus.app:app", host=a.host, port=a.port, reload=a.reload, log_level="info")


def cmd_selftest(a) -> None:
    from . import selftest
    sys.exit(selftest.main(verbose=True))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="astraeus", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sp = p.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("run", help="Monte Carlo campaign under the proposal q")
    r.add_argument("--n", type=int, default=2000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="runs/campaign.npz")
    r.add_argument("--resume", action="store_true", help="append to an existing run")
    r.add_argument("--gt-pose", action="store_true", help="oracle localisation (no VO drift)")
    r.set_defaults(fn=cmd_run)

    e = sp.add_parser("cem", help="cross-entropy adversarial search under a prior")
    e.add_argument("run")
    e.add_argument("--prior", default="P2", choices=list(PR.PRIORS))
    e.add_argument("--iters", type=int, default=6)
    e.add_argument("--batch", type=int, default=200)
    e.set_defaults(fn=cmd_cem)

    q = sp.add_parser("report", help="summarise a saved run")
    q.add_argument("run")
    q.add_argument("--json", action="store_true")
    q.set_defaults(fn=cmd_report)

    d = sp.add_parser("render", help="software-render episodes from a run")
    d.add_argument("run")
    d.add_argument("--episode", type=int)
    d.add_argument("--prior", default="P2", choices=list(PR.PRIORS))
    d.add_argument("--k", type=int, default=4, help="elite failures to render when --episode is absent")
    d.add_argument("--camera", default="chase", choices=RD.CAMERAS)
    d.add_argument("--frame", type=int, default=-1)
    d.add_argument("--sheet", action="store_true")
    d.add_argument("--quality", type=int, default=1, choices=(1, 2, 3))
    d.add_argument("--width", type=int, default=1280)
    d.add_argument("--height", type=int, default=720)
    d.add_argument("--out", default="renders")
    d.set_defaults(fn=cmd_render)

    w = sp.add_parser("whatif", help="one deterministic episode at a chosen x")
    w.add_argument("--x", nargs="*", metavar="DIM=VALUE")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--gt-pose", action="store_true")
    w.add_argument("--png")
    w.add_argument("--camera", default="chase", choices=RD.CAMERAS)
    w.set_defaults(fn=cmd_whatif)

    g = sp.add_parser("ingest", help="fold Isaac Sim / OmniLRS harness CSV rows into a run")
    g.add_argument("run")
    g.add_argument("csv")
    g.add_argument("--from-proposal", action="store_true", default=True,
                   help="rows were sampled from PROPOSAL (default; enables re-weighting)")
    g.add_argument("--not-from-proposal", dest="from_proposal", action="store_false")
    g.set_defaults(fn=cmd_ingest)

    s = sp.add_parser("serve", help="FastAPI server + live world model")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_serve)

    t = sp.add_parser("selftest", help="run the built-in invariant checks")
    t.set_defaults(fn=cmd_selftest)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
