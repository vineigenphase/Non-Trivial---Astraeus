"""Outcome detection from a stream of rover poses.

Mirrors the rules in astraeus.rover so that an Isaac Sim / OmniLRS episode is
labelled exactly the way a local one is:

  success     true position within GOAL_TOL of the goal
  stuck       < STUCK_DIST of progress over any STUCK_WINDOW seconds
  tip_over    |roll| or |pitch| > TIP_DEG
  collision   contact report against a rock prim (reported by the caller)
  nav_miss    policy halted (believes it arrived) but true dist > GOAL_TOL
  timeout     none of the above by T_MAX

The monitor is a pure function of what it is fed, so it is testable without a
simulator and identical between the standalone env and the ROS 2 bridge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple
from collections import deque

OUTCOMES = ("success", "stuck", "tip_over", "collision", "nav_miss", "timeout")

GOAL_X = 30.0
GOAL_TOL = 1.5
TIP_DEG = 30.0
T_MAX = 180.0
STUCK_WINDOW = 15.0
STUCK_DIST = 0.30


@dataclass(frozen=True)
class Pose:
    t: float
    x: float
    y: float
    z: float
    roll: float    # rad
    pitch: float   # rad
    yaw: float     # rad

    @staticmethod
    def from_quat(t: float, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float) -> "Pose":
        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return Pose(t, x, y, z, roll, pitch, yaw)


@dataclass
class EpisodeMonitor:
    goal: Tuple[float, float] = (GOAL_X, 0.0)
    goal_tol: float = GOAL_TOL
    tip_deg: float = TIP_DEG
    t_max: float = T_MAX
    stuck_window: float = STUCK_WINDOW
    stuck_dist: float = STUCK_DIST

    outcome: Optional[str] = None
    t_end: float = 0.0
    max_tilt_deg: float = 0.0
    max_slip: float = 0.0
    max_sinkage: float = 0.0
    min_visibility: float = 1.0
    vis_sum: float = 0.0
    n_pose: int = 0
    _hist: Deque[Tuple[float, float, float]] = field(default_factory=deque)
    _t0: Optional[float] = None
    _last: Optional[Pose] = None

    # ------------------------------------------------------------- feeding
    def update(self, p: Pose, slip: float = 0.0, sinkage: float = 0.0, visibility: float = 1.0) -> Optional[str]:
        """Feed one pose (+ optional wheel/perception telemetry). Returns the outcome once decided."""
        if self.outcome is not None:
            return self.outcome
        if self._t0 is None:
            self._t0 = p.t
        self._last = p
        self.n_pose += 1
        t = p.t - self._t0
        self.t_end = t
        tilt = math.degrees(max(abs(p.roll), abs(p.pitch)))
        self.max_tilt_deg = max(self.max_tilt_deg, tilt)
        self.max_slip = max(self.max_slip, slip)
        self.max_sinkage = max(self.max_sinkage, sinkage)
        self.min_visibility = min(self.min_visibility, visibility)
        self.vis_sum += visibility

        if tilt > self.tip_deg:
            return self._end("tip_over")
        if self.final_dist() <= self.goal_tol:
            return self._end("success")

        self._hist.append((t, p.x, p.y))
        while self._hist and self._hist[0][0] < t - self.stuck_window:
            self._hist.popleft()
        if self._hist and t - self._hist[0][0] >= self.stuck_window - 1e-6:
            t0, x0, y0 = self._hist[0]
            if math.hypot(p.x - x0, p.y - y0) < self.stuck_dist:
                return self._end("stuck")
        if t >= self.t_max:
            return self._end("timeout")
        return None

    def collision(self) -> str:
        """Call when the simulator reports body contact with a rock."""
        return self._end("collision") if self.outcome is None else self.outcome

    def policy_halted(self) -> Optional[str]:
        """Call when the controller declares arrival. Success if truly there, else nav_miss."""
        if self.outcome is not None:
            return self.outcome
        return self._end("success" if self.final_dist() <= self.goal_tol else "nav_miss")

    # ------------------------------------------------------------- queries
    def final_dist(self) -> float:
        if self._last is None:
            return math.hypot(*self.goal)
        return math.hypot(self._last.x - self.goal[0], self._last.y - self.goal[1])

    def mean_visibility(self) -> float:
        return self.vis_sum / self.n_pose if self.n_pose else 1.0

    def _end(self, o: str) -> str:
        assert o in OUTCOMES
        self.outcome = o
        return o

    def metrics(self) -> dict:
        fail = 0.0 if self.outcome == "success" else 1.0
        severity = (fail + 0.5 * self.max_tilt_deg / self.tip_deg + 0.5 * self.max_slip
                    + 0.3 * (1.0 - self.min_visibility) + self.final_dist() / GOAL_X)
        return {
            "outcome": self.outcome or "timeout",
            "t_end": round(self.t_end, 3),
            "final_dist": round(self.final_dist(), 4),
            "max_tilt_deg": round(self.max_tilt_deg, 3),
            "max_slip": round(self.max_slip, 4),
            "max_sinkage": round(self.max_sinkage, 4),
            "min_visibility": round(self.min_visibility, 4),
            "mean_visibility": round(self.mean_visibility(), 4),
            "severity": round(severity, 4),
        }


class GoalSeekController:
    """The same proportional heading/speed law astraeus.rover uses, for the
    Isaac rover. Rock avoidance is left to the simulator-side perception stack
    (or absent, which is itself a legitimate stress-test configuration and is
    recorded in the CSV `controller` column)."""

    def __init__(self, goal: Tuple[float, float] = (GOAL_X, 0.0), v_max: float = 0.5, w_max: float = 0.6,
                 k_ang: float = 1.6, arrive_tol: float = 0.5):
        self.goal, self.v_max, self.w_max, self.k_ang, self.arrive_tol = goal, v_max, w_max, k_ang, arrive_tol
        self.halted = False

    def __call__(self, est_x: float, est_y: float, yaw: float) -> Tuple[float, float]:
        dx, dy = self.goal[0] - est_x, self.goal[1] - est_y
        if math.hypot(dx, dy) < self.arrive_tol:
            self.halted = True
            return 0.0, 0.0
        err = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
        w = max(-self.w_max, min(self.w_max, self.k_ang * err))
        v = self.v_max * max(0.0, math.cos(err))
        return v, w
