#!/usr/bin/env python3
"""Astraeus <-> OmniLRS (Isaac Sim lunar environments) ROS 2 bridge.

Successor to legacy/omnilrs_harness.py. Per episode:

  sample x   -> publish sun pose/intensity, terrain seed, rock density
             -> teleport + reset rover, wait settle_s
             -> drive with GoalSeekController from odometry (or GT with --gt-pose)
             -> label with EpisodeMonitor, append CSV row (flushed)

Load-time dims (k_soil, theta_r, slope_deg for OmniLRS) cannot change live;
`isaac/batch_run.sh` rewrites the sim YAML and relaunches per batch, passing
the fixed values here with --fixed k_soil=.. theta_r=.. slope_deg=.. so every
row still carries the full eight-vector.

Topic names/types come from configs/ros2_topics.yaml. The bridge REFUSES to
run a campaign while that file says `verified: false` unless --i-know is
passed; run discover_topics.py first.

    pixi run python isaac/omnilrs_bridge.py --episodes 50 --mode proposal
    pixi run python isaac/omnilrs_bridge.py --mode prior:P5 --episodes 200 --fixed k_soil=1.2 theta_r=0.70 slope_deg=8
"""
from __future__ import annotations

import argparse
import importlib
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
import yaml
from rclpy.node import Node

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "isaac"))

from astraeus_isaac import EpisodeLog, EpisodeMonitor, Pose, scene_from_x  # noqa: E402
from astraeus_isaac.monitor import GoalSeekController                # noqa: E402
from astraeus_isaac.scene import DIMS, sun_direction                 # noqa: E402


def msg_class(type_str: str):
    """'geometry_msgs/msg/Twist' -> class, without importing every message package up front."""
    pkg, _, name = type_str.split("/")
    return getattr(importlib.import_module(f"{pkg}.msg"), name)


def quat_from_dir(d: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) rotating -Z onto direction d (light shines along -Z)."""
    a = np.array([0.0, 0.0, -1.0])
    b = -np.array(d, float)
    b /= np.linalg.norm(b)
    v = np.cross(a, b)
    w = 1.0 + float(a @ b)
    if w < 1e-8:
        return (1.0, 0.0, 0.0, 0.0)
    q = np.array([v[0], v[1], v[2], w])
    q /= np.linalg.norm(q)
    return tuple(float(c) for c in q)


class Bridge(Node):
    def __init__(self, topics: Dict[str, dict]):
        super().__init__("astraeus_bridge")
        self.T = topics
        self.pose: Optional[Pose] = None
        self.contact = False
        self._t0 = time.monotonic()
        self.create_subscription(msg_class(self.T["odom"]["type"]), self.T["odom"]["name"], self._on_odom, 20)
        if "contacts" in self.T:
            self.create_subscription(msg_class(self.T["contacts"]["type"]), self.T["contacts"]["name"],
                                     self._on_contact, 10)
        self.pubs = {k: self.create_publisher(msg_class(v["type"]), v["name"], 10)
                     for k, v in self.T.items() if k not in ("odom", "imu", "contacts")}

    def now(self) -> float:
        return time.monotonic() - self._t0

    def _on_odom(self, msg) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = Pose.from_quat(self.now(), p.x, p.y, p.z, q.x, q.y, q.z, q.w)

    def _on_contact(self, msg) -> None:
        self.contact = self.contact or bool(msg.data)

    # ------------------------------------------------------------- scene control
    def apply_scene(self, x: np.ndarray, seed: int) -> None:
        sp = scene_from_x(x)
        Pose_ = msg_class(self.T["sun_pose"]["type"])
        m = Pose_()
        qx, qy, qz, qw = quat_from_dir(sun_direction(sp.sun_elevation_deg, sp.sun_azimuth_deg))
        m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = qx, qy, qz, qw
        self.pubs["sun_pose"].publish(m)
        if "sun_intensity" in self.pubs:
            f = msg_class(self.T["sun_intensity"]["type"])()
            f.data = float(sp.sun_intensity_lux / 128_000.0)
            self.pubs["sun_intensity"].publish(f)
        if "rocks_density" in self.pubs:
            f = msg_class(self.T["rocks_density"]["type"])()
            f.data = float(sp.rock_density_per_100m2)
            self.pubs["rocks_density"].publish(f)
        for key in ("terrain_randomize", "rocks_randomize"):
            if key in self.pubs:
                i = msg_class(self.T[key]["type"])()
                i.data = int(seed % 2**31)
                self.pubs[key].publish(i)

    def teleport(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        m = msg_class(self.T["teleport"]["type"])()
        m.header.frame_id = "world"
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.pubs["teleport"].publish(m)
        if "reset" in self.pubs:
            s = msg_class(self.T["reset"]["type"])()
            s.data = "all"
            self.pubs["reset"].publish(s)

    def cmd(self, v: float, w: float) -> None:
        t = msg_class(self.T["cmd_vel"]["type"])()
        t.linear.x, t.angular.z = float(v), float(w)
        self.pubs["cmd_vel"].publish(t)

    def spin_for(self, s: float) -> None:
        end = time.monotonic() + s
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)


# --------------------------------------------------------------------------- episode
def run_episode(b: Bridge, x: np.ndarray, seed: int, camp: dict, gt_pose: bool, vo_drift: float,
                rate_hz: float = 10.0) -> Tuple[dict, Optional[float]]:
    b.apply_scene(x, seed)
    b.spin_for(0.5)
    b.contact = False
    b.teleport(0.0, 0.0, 0.5)
    b.spin_for(camp["episode"]["settle_s"])
    ctl = GoalSeekController()
    mon = EpisodeMonitor(t_max=camp["episode"]["t_max_s"])
    rng = np.random.default_rng(seed ^ 0xA57)
    est = np.zeros(2)
    prev = None
    dt = 1.0 / rate_hz
    outcome = None
    wall0 = time.monotonic()
    while outcome is None:
        b.spin_for(dt)
        p = b.pose
        if p is None:
            if time.monotonic() - wall0 > 10.0:
                raise RuntimeError(f"no odometry on {b.T['odom']['name']} — check ros2_topics.yaml")
            continue
        if prev is None:
            prev = p
        step = np.array([p.x - prev.x, p.y - prev.y])
        if gt_pose:
            est[:] = (p.x, p.y)
        else:
            est += step + rng.normal(0, vo_drift * float(np.linalg.norm(step)), 2)
        v, w = ctl(est[0], est[1], p.yaw)
        b.cmd(v, w)
        outcome = mon.update(p)
        if outcome is None and ctl.halted:
            outcome = mon.policy_halted()
        if outcome is None and b.contact:
            outcome = mon.collision()
        if time.monotonic() - wall0 > camp["episode"]["wall_timeout_s"]:
            outcome = mon._end("timeout")
        prev = p
    b.cmd(0.0, 0.0)
    vo = None if gt_pose or prev is None else float(math.hypot(est[0] - prev.x, est[1] - prev.y))
    return mon.metrics(), vo


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", default=str(REPO / "isaac/configs/ros2_topics.yaml"))
    ap.add_argument("--campaign", default=str(REPO / "isaac/configs/campaign.yaml"))
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--mode", help="proposal | prior:P2 | replay:runs/c1.npz")
    ap.add_argument("--fixed", nargs="*", default=[], help="load-time dims fixed for this batch, DIM=VALUE")
    ap.add_argument("--batch-id", default="")
    ap.add_argument("--csv")
    ap.add_argument("--gt-pose", action="store_true")
    ap.add_argument("--vo-drift", type=float, default=0.02)
    ap.add_argument("--i-know", action="store_true", help="run even if ros2_topics.yaml is not verified")
    a = ap.parse_args(argv)

    tcfg = yaml.safe_load(Path(a.topics).read_text())
    if not tcfg.get("verified") and not a.i_know:
        raise SystemExit("ros2_topics.yaml is not verified against a running sim. "
                         "Run isaac/discover_topics.py, or pass --i-know.")
    camp = yaml.safe_load(Path(a.campaign).read_text())
    if a.mode:
        camp["sampling"]["mode"] = a.mode
    n = a.episodes or camp["sampling"]["episodes"]
    seed = a.seed if a.seed is not None else camp["sampling"]["seed"]

    from standalone_env import sample_plan   # noqa: PLC0415 - same sampling logic, no Isaac import
    X, seeds, source = sample_plan(camp, n, seed)
    fixed = {k: float(v) for k, v in (f.split("=") for f in a.fixed)}
    for k, v in fixed.items():
        X[:, DIMS.index(k)] = v
    if fixed and source == 0:
        # rows with clamped load-time dims are no longer draws from Q
        source = 2
        print("[astraeus-bridge] --fixed given: rows marked source=2 (not proposal draws). "
              "Use the batch runner's stratified design + `ingest --not-from-proposal`.")

    rclpy.init()
    b = Bridge(tcfg["topics"])
    log = EpisodeLog(Path(a.csv or camp["logging"]["csv"]), simulator="omnilrs/ros2",
                     controller="goalseek-P" + ("/gt" if a.gt_pose else "/vo-proxy"),
                     gt_pose=a.gt_pose, batch_id=a.batch_id or f"bridge-{seed}")
    try:
        for i in range(log.index, n):
            w0 = time.monotonic()
            m, vo = run_episode(b, X[i], int(seeds[i]), camp, a.gt_pose, a.vo_drift)
            row = log.write(X[i], int(seeds[i]), m, source=source, vo_error=vo, wall_s=time.monotonic() - w0)
            print(f"  ep {i:4d} {m['outcome']:<9s} t={m['t_end']:6.1f}s dist={m['final_dist']:5.2f} wall={row['wall_s']}s")
    finally:
        b.cmd(0.0, 0.0)
        log.close()
        b.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
