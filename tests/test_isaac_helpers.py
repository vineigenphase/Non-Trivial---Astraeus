"""The Isaac-side helpers must behave without Isaac Sim or ROS 2 installed."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "isaac"))

from astraeus import campaign as C                                    # noqa: E402
from astraeus import priors as PR                                     # noqa: E402
from astraeus import rover as RV                                      # noqa: E402
from astraeus_isaac import EpisodeLog, EpisodeMonitor, Pose, scene_from_x  # noqa: E402
from astraeus_isaac.export import export_episode                      # noqa: E402
from astraeus_isaac.monitor import GoalSeekController                 # noqa: E402
from astraeus_isaac.scene import DIMS, distant_light_rotation_xyz, sun_direction  # noqa: E402

NOMINAL = np.array([PR.NOMINAL_X[d] for d in PR.DIMS])


def test_dims_match_core():
    assert list(DIMS) == PR.DIMS


def test_scene_mapping_monotone():
    a = scene_from_x(NOMINAL)
    hi = NOMINAL.copy()
    hi[DIMS.index("theta_r")] = 0.85
    hi[DIMS.index("tau_dust")] = 0.5
    b = scene_from_x(hi)
    assert b.static_friction > a.static_friction
    assert math.isclose(a.static_friction, math.tan(NOMINAL[1]), rel_tol=1e-3)
    assert b.sun_intensity_lux < a.sun_intensity_lux and b.fog_density > a.fog_density
    assert a.x == dict(zip(DIMS, NOMINAL))


def test_sun_direction_and_light_rotation():
    d = sun_direction(90.0, 0.0)
    assert np.allclose(d, (0, 0, 1), atol=1e-9)
    d = sun_direction(0.0, 90.0)
    assert np.allclose(d, (0, 1, 0), atol=1e-9)
    rx, _, rz = distant_light_rotation_xyz(1.5, 200.0)
    assert -90 < rx < 0 and rz == 290.0


def _pose(t, x, y=0.0, roll=0.0, pitch=0.0):
    return Pose(t, x, y, 0.0, roll, pitch, 0.0)


def test_monitor_success_and_timeout():
    m = EpisodeMonitor()
    for k in range(200):
        out = m.update(_pose(k * 0.5, k * 0.2))
        if out:
            break
    assert out == "success" and m.final_dist() <= 1.5
    m = EpisodeMonitor(t_max=10.0)
    for t in np.arange(0, 12, 0.5):
        m.update(_pose(t, 0.5 * t))
    assert m.outcome == "timeout"


def test_monitor_stuck_tip_collision_navmiss():
    m = EpisodeMonitor()
    for t in np.arange(0, 20, 0.5):
        m.update(_pose(t, 5.0 + 0.001 * t))
    assert m.outcome == "stuck"
    m = EpisodeMonitor()
    assert m.update(_pose(0, 0)) is None and m.update(_pose(0.5, 0.1, roll=math.radians(31))) == "tip_over"
    m = EpisodeMonitor()
    m.update(_pose(0, 0))
    assert m.collision() == "collision" and m.update(_pose(1, 30)) == "collision"
    m = EpisodeMonitor()
    m.update(_pose(0, 25.0))
    assert m.policy_halted() == "nav_miss"
    met = m.metrics()
    assert met["outcome"] == "nav_miss" and met["severity"] > 1.0


def test_monitor_matches_core_constants():
    assert EpisodeMonitor().goal_tol == RV.GOAL_TOL and EpisodeMonitor().tip_deg == RV.TIP_DEG
    assert EpisodeMonitor().t_max == RV.T_MAX and EpisodeMonitor().stuck_window == RV.STUCK_WINDOW


def test_controller_turns_toward_goal_and_halts():
    c = GoalSeekController()
    v, w = c(0.0, 5.0, 0.0)          # goal is to the right (-y): turn right
    assert w < 0 and 0 < v <= 0.5
    v, w = c(29.8, 0.0, 0.0)
    assert (v, w) == (0.0, 0.0) and c.halted


def test_log_roundtrip_and_ingest(tmp_path):
    p = tmp_path / "log.csv"
    with EpisodeLog(p, simulator="test", controller="c", gt_pose=False, batch_id="b0") as log:
        m = EpisodeMonitor()
        m.update(_pose(0, 0))
        m.update(_pose(1, 30))
        log.write(NOMINAL, 7, m.metrics(), source=2, vo_error=0.3, wall_s=1.2)
        log.write(NOMINAL, 8, m.metrics(), source=2)
    assert EpisodeLog.resume_index(p) == 2
    with EpisodeLog(p, simulator="test", controller="c", gt_pose=False) as log:   # append, not overwrite
        assert log.index == 2
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 2 and rows[0]["outcome"] == "success" and rows[0]["simulator"] == "test"
    c = C.Campaign(seed=1)
    assert c.ingest_csv(p, source=2) == 2 and c.n == 2
    assert np.all(c.source == 2) and c.prior_report(PR.P2)["n"] == 0
    assert math.isclose(float(c.metrics["vo_error"][0]), 0.3)


def test_export_matches_local_terrain(tmp_path):
    files = export_episode(NOMINAL, 5, tmp_path, res=0.5)
    hf = np.load(files["heightfield"])
    meta = json.loads(Path(files["meta"]).read_text())
    assert hf.shape == (meta["ny"], meta["nx"]) and np.isfinite(hf).all()
    rocks = json.loads(Path(files["rocks"]).read_text())
    res = RV.simulate_one(NOMINAL, 5)
    assert len(rocks) == int(res.rocks_total[0])
    obj = Path(files["obj"]).read_text().splitlines()
    assert sum(ln.startswith("v ") for ln in obj) == hf.size
    assert sum(ln.startswith("f ") for ln in obj) == 2 * (meta["nx"] - 1) * (meta["ny"] - 1)


def test_isaac_entrypoints_compile_without_isaac():
    import py_compile
    for f in ("standalone_env.py", "omnilrs_bridge.py", "discover_topics.py", "export_terrain.py"):
        py_compile.compile(str(REPO / "isaac" / f), doraise=True)


def test_discover_topics_offline(tmp_path):
    from discover_topics import check, parse_topic_list
    import yaml
    cfg = yaml.safe_load((REPO / "isaac/configs/ros2_topics.yaml").read_text())
    live = {v["name"]: v["type"] for v in cfg["topics"].values()}
    assert check(cfg, live)
    live["/husky/odom"] = "nav_msgs/msg/Path"
    assert not check(cfg, live)
    txt = "/husky/odom [nav_msgs/msg/Odometry]\n/rosout [rcl_interfaces/msg/Log]\n"
    assert parse_topic_list(txt) == {"/husky/odom": "nav_msgs/msg/Odometry", "/rosout": "rcl_interfaces/msg/Log"}
