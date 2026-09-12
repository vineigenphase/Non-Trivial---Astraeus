#!/usr/bin/env python3
"""Verify configs/ros2_topics.yaml against the topics a running sim actually exposes.

    python isaac/discover_topics.py                       # diff only
    python isaac/discover_topics.py --write               # set verified: true if everything matches
    python isaac/discover_topics.py --from-file topics.txt  # offline, from a saved `ros2 topic list -t`

Exit status 0 iff every configured topic exists with the configured type.
Missing optional topics (contacts, rocks_density, sun_intensity, reset, imu)
are reported as warnings; the bridge degrades gracefully without them.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict

import yaml

OPTIONAL = {"contacts", "rocks_density", "sun_intensity", "reset", "imu", "rocks_randomize"}
LINE = re.compile(r"^(\S+)\s+\[(\S+)\]\s*$")


def parse_topic_list(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def live_topics() -> Dict[str, str]:
    try:
        text = subprocess.run(["ros2", "topic", "list", "-t"], check=True, capture_output=True, text=True, timeout=30).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"could not run `ros2 topic list -t`: {e}\nsource your ROS 2 env (pixi shell) with the sim running")
    return parse_topic_list(text)


def check(cfg: dict, live: Dict[str, str]) -> bool:
    ok = True
    for key, spec in cfg["topics"].items():
        name, typ = spec["name"], spec["type"]
        if name not in live:
            level = "WARN" if key in OPTIONAL else "MISSING"
            ok &= key in OPTIONAL
            cands = [t for t in live if t.rsplit("/", 1)[-1].lower() == name.rsplit("/", 1)[-1].lower()]
            hint = f"   candidates: {cands}" if cands else ""
            print(f"[{level:7s}] {key:18s} {name}{hint}")
        elif live[name] != typ:
            ok = False
            print(f"[TYPE   ] {key:18s} {name}: config {typ} but sim publishes {live[name]}")
        else:
            print(f"[ok     ] {key:18s} {name} [{typ}]")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("configs") / "ros2_topics.yaml"))
    ap.add_argument("--from-file", help="saved output of `ros2 topic list -t`")
    ap.add_argument("--write", action="store_true", help="set verified: true on success")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    live = parse_topic_list(Path(a.from_file).read_text()) if a.from_file else live_topics()
    ok = check(cfg, live)
    print(f"\n{len(live)} live topics; config {'MATCHES' if ok else 'DOES NOT MATCH'}")
    if ok and a.write:
        text = Path(a.config).read_text()
        Path(a.config).write_text(re.sub(r"^verified:\s*\w+", "verified: true", text, count=1, flags=re.M))
        print(f"wrote verified: true -> {a.config}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
