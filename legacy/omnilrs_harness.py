#!/usr/bin/env python3
"""Astraeus overnight MC harness.

Loop: sample x -> push disturbances via ROS 2 -> teleport rover -> run episode
      -> detect failure -> append CSV row (flushed) -> repeat.

Run INSIDE the pixi ros2 env, with the sim already up:
    pixi shell            # or: pixi run python astraeus/harness.py ...
    python harness.py --episodes 400 --base-seed 20260808 --log /workspace/astraeus/log.csv

Batch-tier params (k_soil, theta_r): load-time yaml. Launch the sim once per
batch with those values written into astraeus_base.yaml, pass them here via
--k-soil/--theta-r so every row records them. batch_runner.sh automates this.

!!! TOPICS block below is best-guess. Fix from `ros2 topic list -t` BEFORE the
overnight run — every sim-specific name lives here and nowhere else.
"""
import argparse, csv, math, os, sys, time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Int32

import priors

# ═══════════════ TODO: FIX FROM `ros2 topic list -t` ═══════════════
TOPICS = {
    "odom":            ("/husky/odom", Odometry),        # rover odometry (or /odom)
    "cmd_vel":         ("/husky/cmd_vel", Twist),        # velocity command
    "teleport":        ("/OmniLRS/Robots/Teleport", PoseStamped),  # rover teleport
    "sun_pose":        ("/OmniLRS/Sun/Pose", Pose),      # sun orientation control
    "terrain_random":  ("/OmniLRS/Terrain/Randomize", Int32),      # regen w/ seed
}
# If sun control is intensity+angles as separate Float topics, or terrain
# randomisation takes Empty/other msg, adjust publish_* functions below.
# ════════════════════════════════════════════════════════════════════

# episode constants (tune in pilot)
GOAL_XY = (12.0, 0.0)        # waypoint in world frame, ~12 m dash
START_XY = (0.0, 0.0)
GOAL_TOL = 0.75              # m
EP_TIMEOUT = 120.0           # s wall-clock per episode
STUCK_WINDOW = 12.0          # s with < STUCK_DIST progress => entrapment proxy
STUCK_DIST = 0.15            # m
TIP_DEG = 35.0               # |roll| or |pitch| beyond this => tip-over
V_MAX, W_MAX = 0.6, 0.8      # controller limits (m/s, rad/s)
K_LIN, K_ANG = 0.6, 1.5      # P gains
SETTLE_AFTER_RESET = 3.0     # s to let terrain/teleport settle

CSV_FIELDS = ["episode", "seed", "k_soil", "theta_r", "r_terrain",
              "terrain_seed", "sun_e", "sun_psi",
              "outcome", "t_elapsed", "final_dist", "max_tilt_deg",
              "min_progress_rate", "wall_time", "batch_id"]


def yaw_rp_from_quat(q):
    # returns roll, pitch, yaw (rad)
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


class Harness(Node):
    def __init__(self):
        super().__init__("astraeus_harness")
        self.pose = None; self.rp = (0.0, 0.0); self.yaw = 0.0
        t, m = TOPICS["odom"];   self.create_subscription(m, t, self.on_odom, 10)
        t, m = TOPICS["cmd_vel"];        self.pub_cmd = self.create_publisher(m, t, 10)
        t, m = TOPICS["teleport"];       self.pub_tp = self.create_publisher(m, t, 10)
        t, m = TOPICS["sun_pose"];       self.pub_sun = self.create_publisher(m, t, 10)
        t, m = TOPICS["terrain_random"]; self.pub_terr = self.create_publisher(m, t, 10)

    def on_odom(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, p.position.z)
        r, pch, y = yaw_rp_from_quat(p.orientation)
        self.rp = (r, pch); self.yaw = y

    # ---- disturbance pushers (adjust msg construction to real types) ------
    def push_sun(self, elev_deg, azim_deg):
        msg = Pose()
        # OmniLRS convention TBC: encode az/el as orientation quaternion.
        el, az = math.radians(elev_deg), math.radians(azim_deg)
        # ZYX: yaw=azimuth, pitch=-elevation (sun looking down at elev)
        cy, sy = math.cos(az/2), math.sin(az/2)
        cp, sp = math.cos(-el/2), math.sin(-el/2)
        msg.orientation.w = cy*cp; msg.orientation.x = 0.0
        msg.orientation.y = sp*cy; msg.orientation.z = sy*cp
        self.pub_sun.publish(msg)

    def push_terrain(self, seed):
        m = Int32(); m.data = int(seed % (2**31 - 1))
        self.pub_terr.publish(m)

    def teleport(self, x, y):
        m = PoseStamped()
        m.header.frame_id = "world"
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.5
        m.pose.orientation.w = 1.0
        self.pub_tp.publish(m)

    def stop(self):
        self.pub_cmd.publish(Twist())

    # ---- one episode ------------------------------------------------------
    def run_episode(self):
        gx, gy = GOAL_XY
        t0 = time.time()
        max_tilt = 0.0
        last_check_t, last_check_d = t0, None
        min_rate = float("inf")
        outcome = "timeout"

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.pose is None:
                if time.time() - t0 > 20:
                    return "no_odom", 0.0, -1.0, 0.0, 0.0
                continue
            x, y, _ = self.pose
            d = math.hypot(gx - x, gy - y)
            tilt = math.degrees(max(abs(self.rp[0]), abs(self.rp[1])))
            max_tilt = max(max_tilt, tilt)
            now = time.time()

            if tilt > TIP_DEG:
                outcome = "tip_over"; break
            if d < GOAL_TOL:
                outcome = "success"; break
            if now - t0 > EP_TIMEOUT:
                outcome = "timeout"; break
            if last_check_d is None:
                last_check_d = d
            if now - last_check_t >= STUCK_WINDOW:
                progressed = last_check_d - d
                min_rate = min(min_rate, progressed / STUCK_WINDOW)
                if progressed < STUCK_DIST:
                    outcome = "stuck"; break
                last_check_t, last_check_d = now, d

            # P-controller: heading then drive
            hdg = math.atan2(gy - y, gx - x)
            err = math.atan2(math.sin(hdg - self.yaw), math.cos(hdg - self.yaw))
            cmd = Twist()
            cmd.angular.z = max(-W_MAX, min(W_MAX, K_ANG * err))
            cmd.linear.x = 0.0 if abs(err) > 0.8 else max(0.0, min(V_MAX, K_LIN * d))
            self.pub_cmd.publish(cmd)

        self.stop()
        if min_rate == float("inf"):
            min_rate = 0.0
        return outcome, time.time() - t0, d, max_tilt, min_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--base-seed", type=int, default=20260808)
    ap.add_argument("--k-soil", type=float, required=True)   # batch tier
    ap.add_argument("--theta-r", type=float, required=True)  # batch tier
    ap.add_argument("--batch-id", type=str, default="b0")
    ap.add_argument("--log", type=str, default="/workspace/astraeus/log.csv")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    new = not os.path.exists(a.log)
    f = open(a.log, "a", newline="")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if new:
        w.writeheader(); f.flush()

    rclpy.init()
    node = Harness()
    print(f"[astraeus] batch {a.batch_id}: k_soil={a.k_soil} theta_r={a.theta_r}"
          f" episodes={a.episodes} -> {a.log}", flush=True)

    for i in range(a.episodes):
        seed = a.base_seed + i
        priors.init(seed)
        ep = priors.sample_episode()

        node.push_terrain(ep["terrain_seed"])
        node.push_sun(ep["sun_e"], ep["sun_psi"])
        time.sleep(0.5)
        node.teleport(*START_XY)
        t_settle = time.time()
        while time.time() - t_settle < SETTLE_AFTER_RESET:
            rclpy.spin_once(node, timeout_sec=0.05)

        outcome, t_el, fd, mt, mr = node.run_episode()
        row = {"episode": i, "seed": seed, "k_soil": a.k_soil,
               "theta_r": a.theta_r, "r_terrain": ep.get("r_terrain", 1.0),
               "terrain_seed": ep["terrain_seed"], "sun_e": ep["sun_e"],
               "sun_psi": ep["sun_psi"], "outcome": outcome,
               "t_elapsed": round(t_el, 2), "final_dist": round(fd, 3),
               "max_tilt_deg": round(mt, 2), "min_progress_rate": round(mr, 4),
               "wall_time": round(time.time(), 1), "batch_id": a.batch_id}
        w.writerow(row); f.flush(); os.fsync(f.fileno())   # survives 3am crash
        print(f"[{i+1}/{a.episodes}] {outcome:9s} t={t_el:6.1f}s "
              f"e={ep['sun_e']:.2f} tilt={mt:5.1f}", flush=True)

    node.stop(); rclpy.shutdown(); f.close()
    print("[astraeus] batch complete.", flush=True)


if __name__ == "__main__":
    main()
