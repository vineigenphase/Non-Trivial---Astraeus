"""The rover, its autonomy stack and the episode loop — vectorised over episodes.

One episode: the rover starts at the origin facing +x and must reach the
waypoint at (GOAL_X, 0) within GOAL_TOL metres before T_MAX seconds of sim
time. The autonomy stack is deliberately simple and fully visible:

  perception   rocks inside a 7 m / 110 deg sensor footprint are detected at a
               Poisson rate lambda · V, with V the composite visibility from
               perception.py; pose comes from visual odometry whose drift grows
               with (1 - V) — or from ground truth with `gt_pose=True`.
  planning     a proportional heading controller towards the goal with a
               reactive lateral repulsion from detected rocks in the path.
  mobility     four rigid wheels on Bekker/Janosi regolith (terramechanics.py);
               attitude from the terrain under a rigid 1.8 x 1.4 m footprint.

Failure modes (the things the stress test ranks):
  stuck        wheels immobilised (demand > available thrust) for STUCK_HOLD
               seconds, slip-sinkage past the entrapment threshold, or progress
               below STUCK_DIST in a STUCK_WINDOW — Spirit-at-Troy class.
  tip_over     |pitch| or |roll| beyond TIP_DEG — the static stability margin.
  collision    body contact with a rock taller than ground clearance.
  nav_miss     the rover *believes* it has arrived and halts, but the true
               position is outside GOAL_TOL — a pure perception failure.
  timeout      none of the above, but out of time.

Everything is deterministic in (parameters, seed): the same episode can be
re-simulated on its own with `record=True` to obtain the full trajectory that
the renderer and the Isaac Sim exporter consume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import perception as P
from . import terramechanics as TM
from .priors import DIMS
from .terrain import (GOAL_X, N_ROCK_SLOTS, TerrainBatch, _hash01,
                      episode_normals, episode_uniforms)

# ---------------------------------------------------------------- constants
MASS_KG = 430.0                # VIPER-class
HALF_LEN = 0.9                 # m, half wheelbase
HALF_WID = 0.7                 # m, half track
CLEARANCE = 0.30               # m body ground clearance
GOAL_TOL = 1.5                 # m, true-position radius that counts as arrival
ARRIVE_TOL = 0.5               # m, estimated-position radius at which the policy halts
T_MAX = 180.0                  # s sim time
DT = 0.25                      # s
MAX_STEPS = int(T_MAX / DT)
V_MAX = 0.5                    # m/s commanded
W_MAX = 0.6                    # rad/s
K_ANG = 1.6
K_AVOID = 1.4
AVOID_RANGE = 5.0              # m
AVOID_MARGIN = 0.45            # m beyond body half-width
TIP_DEG = 30.0
STUCK_WINDOW = 15.0            # s
STUCK_DIST = 0.30              # m progress required per window
STUCK_HOLD = 8.0               # s of immobilisation before declaring stuck
RECORD_EVERY = 2               # steps between trajectory samples when recording
HALF_FOV_RAD = np.radians(P.SENSOR_HALF_FOV_DEG)
AVOID_HALF_ANGLE_RAD = np.radians(50.0)

OUTCOMES: List[str] = ["success", "stuck", "tip_over", "collision", "nav_miss", "timeout"]
OUTCOME_CODE: Dict[str, int] = {o: i for i, o in enumerate(OUTCOMES)}
FAIL_CODES = np.array([OUTCOME_CODE[o] for o in OUTCOMES if o != "success"])


@dataclass
class EpisodeResult:
    """Batch results (arrays of length n) plus optional recorded trajectories."""
    X: np.ndarray                 # (n, 8) disturbance vectors
    seed: np.ndarray              # (n,)
    outcome: np.ndarray           # (n,) int codes
    t_end: np.ndarray             # (n,) s
    final_dist: np.ndarray        # (n,) m
    max_tilt_deg: np.ndarray      # (n,)
    max_slip: np.ndarray          # (n,)
    max_sinkage: np.ndarray       # (n,) m
    min_visibility: np.ndarray    # (n,)
    mean_visibility: np.ndarray   # (n,)
    rocks_total: np.ndarray       # (n,)
    rocks_detected: np.ndarray    # (n,)
    vo_error: np.ndarray          # (n,) m — |estimate - truth| at the end
    severity: np.ndarray          # (n,) scalar used to rank elites
    trace: Optional[Dict[str, np.ndarray]] = None   # (n, T, ...) when recorded
    terrain: Optional[TerrainBatch] = None

    @property
    def fail(self) -> np.ndarray:
        return self.outcome != OUTCOME_CODE["success"]

    def outcome_names(self) -> List[str]:
        return [OUTCOMES[int(c)] for c in self.outcome]

    def row(self, i: int) -> Dict[str, object]:
        d = {k: float(v) for k, v in zip(DIMS, self.X[i])}
        d.update({
            "seed": int(self.seed[i]), "outcome": OUTCOMES[int(self.outcome[i])],
            "t_end": round(float(self.t_end[i]), 2),
            "final_dist": round(float(self.final_dist[i]), 3),
            "max_tilt_deg": round(float(self.max_tilt_deg[i]), 2),
            "max_slip": round(float(self.max_slip[i]), 3),
            "max_sinkage_m": round(float(self.max_sinkage[i]), 4),
            "min_visibility": round(float(self.min_visibility[i]), 3),
            "mean_visibility": round(float(self.mean_visibility[i]), 3),
            "rocks_total": int(self.rocks_total[i]),
            "rocks_detected": int(self.rocks_detected[i]),
            "vo_error_m": round(float(self.vo_error[i]), 3),
            "severity": round(float(self.severity[i]), 3),
        })
        return d


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def severity_score(fail: np.ndarray, max_tilt_deg, max_slip, min_vis, final_dist) -> np.ndarray:
    return (fail.astype(float) + 0.5 * np.asarray(max_tilt_deg) / TIP_DEG
            + 0.5 * np.asarray(max_slip) + 0.3 * (1.0 - np.asarray(min_vis))
            + np.asarray(final_dist) / GOAL_X)


def simulate(X: np.ndarray, seed: np.ndarray, gt_pose: bool = False,
             record: bool = False, progress=None) -> EpisodeResult:
    """Run n episodes in lockstep. X is (n, 8) in priors.DIMS order."""
    X = np.atleast_2d(np.asarray(X, float))
    seed = np.asarray(seed, np.int64)
    n = X.shape[0]
    k_soil, theta_r, r_terrain, rho_rock, slope_deg, sun_e, sun_psi, tau_dust = X.T

    terrain = TerrainBatch.build(seed, r_terrain, slope_deg, rho_rock)
    f_sh = P.shadow_fraction(sun_e, r_terrain)
    rock_shadowed = episode_uniforms(seed, N_ROCK_SLOTS, stream=3) < f_sh[:, None]
    vo_drift = episode_normals(seed, 2, stream=5) * P.SIGMA_VO_PER_M   # m per m, per axis

    # state
    x, y, hdg = np.zeros(n), np.zeros(n), np.zeros(n)
    ex, ey = np.zeros(n), np.zeros(n)                    # VO estimate
    z_dig = np.zeros(n)
    done = np.zeros(n, bool)
    outcome = np.full(n, OUTCOME_CODE["timeout"], np.int64)
    t_end = np.full(n, T_MAX)
    max_tilt, max_slip, max_z = np.zeros(n), np.zeros(n), np.zeros(n)
    min_vis, sum_vis, n_vis = np.ones(n), np.zeros(n), np.zeros(n)
    detected = np.zeros((n, N_ROCK_SLOTS), bool)
    immob_time = np.zeros(n)
    last_check_t = np.zeros(n)
    last_check_d = np.full(n, GOAL_X)
    halted = np.zeros(n, bool)                            # policy believes it has arrived

    rock_x = terrain.rock_xy[:, :, 0]
    rock_y = terrain.rock_xy[:, :, 1]
    rock_r = terrain.rock_r
    rock_act = terrain.rock_active
    hazard = rock_act & (rock_r >= CLEARANCE)
    rock_h = np.clip(rock_r * (1.0 - 0.35), 0.0, None)   # exposed height

    W_w = MASS_KG * TM.G_MOON / TM.WHEEL.count
    slot_ids = np.arange(N_ROCK_SLOTS, dtype=np.int64)[None, :]

    trace: Dict[str, list] = {k: [] for k in
                              ("t", "x", "y", "z", "heading", "pitch", "roll", "slip",
                               "v", "visibility", "ex", "ey", "sinkage")} if record else {}

    for step in range(MAX_STEPS):
        t = step * DT
        idx = np.flatnonzero(~done)
        m = idx.size
        if m == 0:
            break
        # gather the active subset — finished episodes cost nothing
        xs, ys, hs = x[idx], y[idx], hdg[idx]
        exs, eys = ex[idx], ey[idx]
        sub = terrain if m == n else terrain.select(idx)
        rx, ry, rr = rock_x[idx], rock_y[idx], rock_r[idx]
        ract, rhaz, rh = rock_act[idx], hazard[idx], rock_h[idx]
        det = detected[idx]
        s_e, s_psi, tau, rt = sun_e[idx], sun_psi[idx], tau_dust[idx], r_terrain[idx]
        ar = np.arange(m)

        # ---------------------------------------------------------- perception
        dx = rx - xs[:, None]
        dy = ry - ys[:, None]
        dist = np.hypot(dx, dy)
        rel = _wrap(np.arctan2(dy, dx) - hs[:, None])
        vis = P.visibility(s_e, s_psi, tau, rt, hs)
        in_fp = ract & (dist < P.LOOKAHEAD_M) & (np.abs(rel) < HALF_FOV_RAD)
        rate = P.DETECT_RATE * vis[:, None] * np.where(rock_shadowed[idx], P.SHADOW_DETECT_PENALTY, 1.0)
        p_det = 1.0 - np.exp(-rate * DT)
        u = _hash01(slot_ids + step * 4099, np.full_like(slot_ids, 31 + step), seed[idx, None])
        det = det | (in_fp & (u < p_det))
        detected[idx] = det

        # ------------------------------------------------------------ planning
        gx_err = GOAL_X - exs
        gy_err = -eys
        d_est = np.hypot(gx_err, gy_err)
        err = _wrap(np.arctan2(gy_err, gx_err) - hs)
        # reactive avoidance of detected rocks ahead
        lateral = dist * np.sin(rel)
        ahead = det & (dist < AVOID_RANGE) & (np.abs(rel) < AVOID_HALF_ANGLE_RAD) \
            & (np.abs(lateral) < rr + HALF_WID + AVOID_MARGIN)
        dist_masked = np.where(ahead, dist, np.inf)
        j = dist_masked.argmin(1)
        has_threat = np.isfinite(dist_masked[ar, j])
        side = np.where(lateral[ar, j] >= 0, -1.0, 1.0)           # steer away from the rock
        avoid = np.where(has_threat, side * K_AVOID * np.clip(1.0 - dist[ar, j] / AVOID_RANGE, 0.0, 1.0), 0.0)
        err_total = err + avoid
        omega = np.clip(K_ANG * err_total, -W_MAX, W_MAX)
        v_cmd = V_MAX * np.clip(1.0 - np.abs(err_total) / 0.9, 0.0, 1.0) * np.clip(d_est / 2.0, 0.2, 1.0)
        halt = halted[idx] | (d_est < ARRIVE_TOL)
        halted[idx] = halt
        v_cmd = np.where(halt, 0.0, v_cmd)
        omega = np.where(halt, 0.0, omega)

        # -------------------------------------------------------------- terrain
        pitch, roll, _ = sub.normal_angles(xs, ys, hs, HALF_LEN, HALF_WID)
        tilt_deg = np.degrees(np.maximum(np.abs(pitch), np.abs(roll)))

        # rock contact: bumps for small rocks under the wheels, collision for hazards
        near = dist < rr + HALF_WID * 0.8
        bump = (near & ract & ~rhaz) * (rh / TM.WHEEL.radius) * W_w
        extra_R = bump.sum(1)
        collide = (near & rhaz).any(1)

        # ------------------------------------------------------------ mobility
        ts = TM.step_wheel_soil(MASS_KG, pitch, k_soil[idx], theta_r[idx], z_dig[idx], v_cmd, DT, extra_R)
        v = ts["v"]
        slip = ts["slip"]
        z_dig[idx] = ts["z_dig"]
        ds = v * DT
        xs_new = xs + ds * np.cos(hs)
        ys_new = ys + ds * np.sin(hs)
        hs_new = _wrap(hs + omega * (1.0 - 0.5 * slip) * DT)
        # VO: true displacement plus per-episode drift growing with (1 - V)
        if gt_pose:
            exs_new, eys_new = xs_new, ys_new
        else:
            drift = (1.0 - vis)[:, None] * vo_drift[idx] * ds[:, None]
            exs_new = exs + (xs_new - xs) + drift[:, 0]
            eys_new = eys + (ys_new - ys) + drift[:, 1]
        x[idx], y[idx], hdg[idx] = xs_new, ys_new, hs_new
        ex[idx], ey[idx] = exs_new, eys_new

        # ------------------------------------------------------------ bookkeeping
        max_tilt[idx] = np.maximum(max_tilt[idx], tilt_deg)
        max_slip[idx] = np.maximum(max_slip[idx], slip)
        max_z[idx] = np.maximum(max_z[idx], ts["z"])
        min_vis[idx] = np.minimum(min_vis[idx], vis)
        sum_vis[idx] += vis
        n_vis[idx] += 1
        immob = immob_time[idx]
        immob = np.where(ts["immobilised"] & (v_cmd > 0), immob + DT, 0.0)
        immob_time[idx] = immob

        d_true = np.hypot(GOAL_X - xs_new, ys_new)
        window = (t - last_check_t[idx]) >= STUCK_WINDOW
        stalled = window & ((last_check_d[idx] - d_true) < STUCK_DIST)
        last_check_t[idx] = np.where(window, t, last_check_t[idx])
        last_check_d[idx] = np.where(window, d_true, last_check_d[idx])

        # ---------------------------------------------------------- termination
        out = np.full(m, -1, np.int64)
        for code, cond in (
            (OUTCOME_CODE["success"], d_true < GOAL_TOL),
            (OUTCOME_CODE["tip_over"], tilt_deg > TIP_DEG),
            (OUTCOME_CODE["collision"], collide),
            (OUTCOME_CODE["stuck"], ts["entrapped"] | (immob >= STUCK_HOLD)),
            (OUTCOME_CODE["nav_miss"], halt & (d_true >= GOAL_TOL)),
            (OUTCOME_CODE["stuck"], stalled & ~halt),
            (OUTCOME_CODE["timeout"], np.full(m, t + DT >= T_MAX)),
        ):
            out = np.where((out < 0) & cond, code, out)
        fin = out >= 0
        fin_idx = idx[fin]
        outcome[fin_idx] = out[fin]
        t_end[fin_idx] = t + DT
        done[fin_idx] = True

        if record and (step % RECORD_EVERY == 0):
            zc = terrain.height(x[:, None], y[:, None])[:, 0]
            full = {}
            for k, arr in (("pitch", pitch), ("roll", roll), ("slip", slip), ("v", v),
                           ("visibility", vis), ("sinkage", ts["z"])):
                a = np.zeros(n)
                a[idx] = arr
                full[k] = a
            for k, arr in (("t", np.full(n, t)), ("x", x), ("y", y), ("z", zc), ("heading", hdg),
                           ("ex", ex), ("ey", ey)):
                full[k] = arr.copy()
            for k in trace:
                trace[k].append(full[k])
        if progress is not None and step % 40 == 0:
            progress(step, MAX_STEPS, int(done.sum()), n)

    final_dist = np.hypot(GOAL_X - x, y)
    fail = outcome != OUTCOME_CODE["success"]
    sev = severity_score(fail, max_tilt, max_slip, min_vis, final_dist)
    res = EpisodeResult(
        X=X, seed=seed, outcome=outcome, t_end=t_end, final_dist=final_dist,
        max_tilt_deg=max_tilt, max_slip=max_slip, max_sinkage=max_z,
        min_visibility=min_vis, mean_visibility=np.where(n_vis > 0, sum_vis / np.maximum(n_vis, 1), 1.0),
        rocks_total=terrain.rock_count, rocks_detected=(detected & rock_act).sum(1),
        vo_error=np.hypot(ex - x, ey - y), severity=sev, terrain=terrain,
    )
    if record:
        res.trace = {k: np.stack(v, 1) for k, v in trace.items()}
    return res


def simulate_one(x: np.ndarray, seed: int, gt_pose: bool = False) -> EpisodeResult:
    """Re-simulate a single episode with a full trajectory recorded."""
    return simulate(np.asarray(x, float)[None, :], np.array([seed]), gt_pose=gt_pose, record=True)
