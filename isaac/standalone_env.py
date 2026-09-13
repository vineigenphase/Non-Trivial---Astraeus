#!/usr/bin/env python3
"""Astraeus standalone Isaac Sim environment.

Builds, for each sampled disturbance vector x, the *same* terrain the local
simulator uses (same seed -> same heightfield, rocks, craters), lights it with
a DistantLight at (sun_e, sun_psi) attenuated by tau_dust, spawns a rover
(your USD articulation or the built-in four-wheel proxy), drives it to the
goal with the same proportional controller as astraeus.rover, labels the
outcome with the same rules, and appends a CSV row that
`python -m astraeus ingest` folds straight into a campaign.

    # inside the Isaac Sim python env
    ./python.sh isaac/standalone_env.py --episodes 200 --mode proposal
    ./python.sh isaac/standalone_env.py --mode replay:runs/c1.npz --episodes 50
    ./python.sh isaac/standalone_env.py --episodes 1 --no-headless --write-frames

Isaac Sim requires SimulationApp to exist before any `omni.*`/`pxr` import, so
those imports live in `_isaac_imports()` and are the only ones not at the top
of the file. Everything else (scene mapping, monitor, CSV) is importable and
tested without Isaac.

What PhysX does NOT give you and this script does not pretend to: sinkage,
slip-sinkage, bulldozing. Rigid contacts with a friction coefficient mu=tan(theta_r)
reproduce tip-over, collision and navigation failures faithfully; entrapment
is under-predicted unless a terramechanics wheel plugin is installed (hook:
`Rover.apply_terramechanics()` — implement against your plugin). The CSV
`simulator` column records "isaac-standalone/rigid" so the two regimes are
never confused downstream.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "isaac"))

from astraeus import priors as PR                                   # noqa: E402
from astraeus_isaac import EpisodeLog, EpisodeMonitor, Pose, scene_from_x  # noqa: E402
from astraeus_isaac.export import episode_terrain, heightfield     # noqa: E402
from astraeus_isaac.monitor import GoalSeekController              # noqa: E402
from astraeus_isaac.scene import distant_light_rotation_xyz        # noqa: E402


# --------------------------------------------------------------------------- sampling
def sample_plan(cfg: dict, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (X, seeds, source) per configs/campaign.yaml sampling.mode."""
    mode = cfg["sampling"]["mode"]
    rng = np.random.default_rng(seed)
    if mode == "proposal":
        return PR.sample_proposal(rng, n), rng.integers(0, 2**31 - 1, n), 0
    if mode.startswith("prior:"):
        p = PR.PRIORS[mode.split(":", 1)[1]]
        return PR.clip_to_bounds(p.sample(rng, n)), rng.integers(0, 2**31 - 1, n), 2
    if mode.startswith("replay:"):
        z = np.load(mode.split(":", 1)[1], allow_pickle=False)
        return z["X"][:n], z["seeds"][:n], 2
    raise SystemExit(f"unknown sampling.mode {mode!r}")


# --------------------------------------------------------------------------- isaac side
class IsaacScene:
    """Owns SimulationApp, the World, terrain/rocks/light prims and the rover."""

    def __init__(self, scene_cfg: dict, headless: bool):
        self.cfg = scene_cfg
        self._isaac_imports(headless)
        app = scene_cfg["app"]
        self.world = self.core.World(stage_units_in_meters=1.0, physics_dt=app["physics_dt"],
                                     rendering_dt=app["rendering_dt"])
        self.stage = self.world.stage
        phys = scene_cfg["physics"]
        scene = self.world.get_physics_context()
        scene.set_gravity(-abs(phys["gravity_m_s2"]))
        if phys.get("gpu_dynamics", True):
            scene.enable_gpu_dynamics(True)
        self.terrain_path = "/World/Terrain"
        self.rock_paths: List[str] = []
        self.light_path = "/World/Sun"
        self.rover = None

    def _isaac_imports(self, headless: bool) -> None:
        try:
            from isaacsim import SimulationApp          # Isaac Sim >= 4.0
        except ImportError:                             # Isaac Sim 2023.x
            from omni.isaac.kit import SimulationApp
        app = self.cfg["app"]
        self.sim_app = SimulationApp({"headless": headless, "renderer": app["renderer"],
                                      "width": app["width"], "height": app["height"]})
        import omni.isaac.core as core                  # noqa: PLC0415
        import omni.isaac.core.utils.prims as prim_utils
        import omni.isaac.core.utils.stage as stage_utils
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.materials import PhysicsMaterial
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
        self.core, self.prim_utils, self.stage_utils = core, prim_utils, stage_utils
        self.Articulation, self.PhysicsMaterial = Articulation, PhysicsMaterial
        self.Gf, self.PhysxSchema, self.Sdf, self.Usd = Gf, PhysxSchema, Sdf, Usd
        self.UsdGeom, self.UsdLux, self.UsdPhysics, self.UsdShade, self.Vt = UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

    # ------------------------------------------------------------------ terrain
    def build_terrain(self, x: Sequence[float], seed: int, sp) -> Dict[str, np.ndarray]:
        tcfg = self.cfg["terrain"]
        t = episode_terrain(x, seed)
        dem = heightfield(t, res=tcfg["res_m"], margin=tcfg["margin_m"])
        xs, ys, Z = dem["x"], dem["y"], dem["z"]
        ny, nx = Z.shape
        if self.stage.GetPrimAtPath(self.terrain_path):
            self.stage.RemovePrim(self.terrain_path)
        mesh = self.UsdGeom.Mesh.Define(self.stage, self.terrain_path)
        X, Y = np.meshgrid(xs, ys)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1).astype(np.float32)
        mesh.CreatePointsAttr(self.Vt.Vec3fArray.FromNumpy(pts))
        j, i = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
        a = (j * nx + i).ravel()
        quads = np.stack([a, a + 1, a + nx + 1, a + nx], -1).ravel()
        mesh.CreateFaceVertexCountsAttr(self.Vt.IntArray.FromNumpy(np.full((ny - 1) * (nx - 1), 4, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(self.Vt.IntArray.FromNumpy(quads.astype(np.int32)))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr(self.Vt.Vec3fArray([self.Gf.Vec3f(*tcfg["albedo"])]))
        prim = mesh.GetPrim()
        self.UsdPhysics.CollisionAPI.Apply(prim)
        mesh_col = self.UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_col.CreateApproximationAttr("none")           # exact triangle mesh for static ground
        pcol = self.PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pcol.CreateContactOffsetAttr(self.cfg["physics"]["ground_contact_offset_m"])
        # ground material: friction from theta_r
        mat = self.PhysicsMaterial("/World/Materials/Regolith", static_friction=sp.static_friction,
                                   dynamic_friction=sp.dynamic_friction, restitution=sp.restitution)
        self.UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            self.UsdShade.Material(self.stage.GetPrimAtPath(mat.prim_path)), self.UsdShade.Tokens.weakerThanDescendants,
            "physics")
        self._build_rocks(t)
        return dem

    def _build_rocks(self, t) -> None:
        for p in self.rock_paths:
            if self.stage.GetPrimAtPath(p):
                self.stage.RemovePrim(p)
        self.rock_paths = []
        rocks = t.rocks_of(0)
        if not rocks:
            return
        rx = np.array([r["x"] for r in rocks])
        ry = np.array([r["y"] for r in rocks])
        gz = t.height(rx[None, :], ry[None, :], rocks=False)[0]
        rcfg = self.cfg["rocks"]
        for k, (r, z) in enumerate(zip(rocks, gz)):
            path = f"/World/Rocks/rock_{k:03d}"
            rad = r["r"]
            if rcfg["mesh_usd"]:
                self.stage_utils.add_reference_to_stage(rcfg["mesh_usd"], path)
                xf = self.UsdGeom.Xformable(self.stage.GetPrimAtPath(path))
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(self.Gf.Vec3d(r["x"], r["y"], z - 0.35 * rad))
                xf.AddScaleOp().Set(self.Gf.Vec3f(rad, rad, rad))
            else:
                sph = self.UsdGeom.Sphere.Define(self.stage, path)
                sph.CreateRadiusAttr(rad)
                sph.AddTranslateOp().Set(self.Gf.Vec3d(r["x"], r["y"], z - 0.35 * rad))
                sph.CreateDisplayColorAttr(self.Vt.Vec3fArray([self.Gf.Vec3f(*rcfg["albedo"])]))
            prim = self.stage.GetPrimAtPath(path)
            self.UsdPhysics.CollisionAPI.Apply(prim)
            prim.CreateAttribute("astraeus:isRock", self.Sdf.ValueTypeNames.Bool).Set(True)
            self.rock_paths.append(path)

    # ------------------------------------------------------------------ light + dust
    def set_sun(self, sp) -> None:
        if not self.stage.GetPrimAtPath(self.light_path):
            self.UsdLux.DistantLight.Define(self.stage, self.light_path)
        light = self.UsdLux.DistantLight(self.stage.GetPrimAtPath(self.light_path))
        light.CreateIntensityAttr(sp.sun_intensity_lux / 128_000.0 * 3000.0)   # Kit intensity is unitless
        light.CreateAngleAttr(0.53)
        light.CreateColorAttr(self.Gf.Vec3f(*sp.sun_color))
        xf = self.UsdGeom.Xformable(light.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddRotateXYZOp().Set(self.Gf.Vec3f(*distant_light_rotation_xyz(sp.sun_elevation_deg, sp.sun_azimuth_deg)))
        # no sky: the Moon has none. Dust haze via RTX fog settings.
        import carb                                   # noqa: PLC0415 - Kit runtime only
        s = carb.settings.get_settings()
        s.set("/rtx/fog/enabled", sp.fog_density > 0)
        s.set("/rtx/fog/fogColor", list(sp.fog_color))
        s.set("/rtx/fog/fogColorIntensity", 0.6)
        s.set("/rtx/fog/fogZup/enabled", True)
        s.set("/rtx/fog/fogStartHeight", -50.0)
        s.set("/rtx/fog/fogHeightDensity", 1.0)
        s.set("/rtx/fog/fogStartDist", 1.0)
        s.set("/rtx/fog/fogEndDist", 1.0 / max(sp.fog_density, 1e-6))
        s.set("/rtx/fog/fogDistanceDensity", min(sp.fog_density * 50.0, 1.0))

    # ------------------------------------------------------------------ rover
    def spawn_rover(self, dem: Dict[str, np.ndarray]) -> None:
        rc = self.cfg["rover"]
        path = rc["prim_path"]
        if self.stage.GetPrimAtPath(path):
            self.stage.RemovePrim(path)
        x0, y0 = rc["start_xy"]
        z0 = float(np.interp(x0, dem["x"], dem["z"][np.argmin(np.abs(dem["y"] - y0))])) + rc["start_z_offset_m"]
        if rc["usd_path"]:
            self.stage_utils.add_reference_to_stage(rc["usd_path"], path)
        else:
            self._build_proxy_rover(path, rc)
        self.rover = self.Articulation(prim_path=path, name="rover", position=np.array([x0, y0, z0]))
        self.world.scene.add(self.rover)
        self._attach_camera(path)

    def _build_proxy_rover(self, path: str, rc: dict) -> None:
        """Box chassis + four revolute-driven cylinder wheels; masses from config."""
        G, UP, UG = self.Gf, self.UsdPhysics, self.UsdGeom
        root = UG.Xform.Define(self.stage, path)
        UP.ArticulationRootAPI.Apply(root.GetPrim())
        body = UG.Cube.Define(self.stage, f"{path}/chassis")
        body.CreateSizeAttr(1.0)
        body.AddScaleOp().Set(G.Vec3f(2 * rc["half_wheelbase_m"], 2 * rc["half_track_m"] - 2 * rc["wheel_width_m"], 0.5))
        body.AddTranslateOp().Set(G.Vec3d(0, 0, rc["clearance_m"] + 0.25))
        UP.RigidBodyAPI.Apply(body.GetPrim())
        UP.CollisionAPI.Apply(body.GetPrim())
        m = UP.MassAPI.Apply(body.GetPrim())
        m.CreateMassAttr(rc["mass_kg"] * 0.8)
        body.GetPrim().CreateAttribute("astraeus:isBody", self.Sdf.ValueTypeNames.Bool).Set(True)
        wheel_mass = rc["mass_kg"] * 0.2 / 4
        for name, sx, sy in zip(rc["wheel_joint_names"], (1, 1, -1, -1), (1, -1, 1, -1)):
            wp = f"{path}/{name.replace('_joint', '')}"
            w = UG.Cylinder.Define(self.stage, wp)
            w.CreateRadiusAttr(rc["wheel_radius_m"])
            w.CreateHeightAttr(rc["wheel_width_m"])
            w.CreateAxisAttr("Y")
            w.AddTranslateOp().Set(G.Vec3d(sx * rc["half_wheelbase_m"], sy * rc["half_track_m"], rc["wheel_radius_m"]))
            UP.RigidBodyAPI.Apply(w.GetPrim())
            UP.CollisionAPI.Apply(w.GetPrim())
            UP.MassAPI.Apply(w.GetPrim()).CreateMassAttr(wheel_mass)
            j = UP.RevoluteJoint.Define(self.stage, f"{path}/{name}")
            j.CreateBody0Rel().SetTargets([body.GetPath()])
            j.CreateBody1Rel().SetTargets([w.GetPath()])
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(G.Vec3f(sx * rc["half_wheelbase_m"] / (2 * rc["half_wheelbase_m"]),
                                          sy * rc["half_track_m"] / max(2 * rc["half_track_m"] - 2 * rc["wheel_width_m"], 1e-3),
                                          (rc["wheel_radius_m"] - rc["clearance_m"] - 0.25) / 0.5))
            j.CreateLocalPos1Attr(G.Vec3f(0, 0, 0))
            drive = UP.DriveAPI.Apply(j.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateDampingAttr(1e3)
            drive.CreateStiffnessAttr(0.0)
            drive.CreateMaxForceAttr(rc["max_wheel_torque_nm"])

    def _attach_camera(self, rover_path: str) -> None:
        cc = self.cfg["camera"]
        cam = self.UsdGeom.Camera.Define(self.stage, cc["prim_path"])
        cam.CreateFocalLengthAttr(cc["focal_mm"])
        xf = self.UsdGeom.Xformable(cam.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(self.Gf.Vec3d(*cc["offset_xyz"]))
        xf.AddRotateXYZOp().Set(self.Gf.Vec3f(90.0 + cc["pitch_deg"], 0.0, -90.0))

    # ------------------------------------------------------------------ stepping
    def reset(self) -> None:
        self.world.reset()

    def step(self, render: bool) -> None:
        self.world.step(render=render)

    def rover_pose(self, t: float) -> Pose:
        pos, quat = self.rover.get_world_pose()      # quat is (w, x, y, z) in omni.isaac.core
        return Pose.from_quat(t, float(pos[0]), float(pos[1]), float(pos[2]),
                              float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0]))

    def drive(self, v: float, w: float) -> None:
        rc = self.cfg["rover"]
        r, half_track = rc["wheel_radius_m"], rc["half_track_m"]
        wl, wr = (v - w * half_track) / r, (v + w * half_track) / r
        names = rc["wheel_joint_names"]
        idx = [self.rover.get_dof_index(n) for n in names]
        targets = np.zeros(self.rover.num_dof)
        for i, n in zip(idx, names):
            targets[i] = wl if n.startswith(("fl", "rl")) else wr
        self.rover.set_joint_velocities(targets, joint_indices=np.array(idx))

    def wheel_slip(self, v_body: float) -> float:
        rc = self.cfg["rover"]
        idx = [self.rover.get_dof_index(n) for n in rc["wheel_joint_names"]]
        omega = float(np.mean(np.abs(self.rover.get_joint_velocities()[idx])))
        v_wheel = omega * rc["wheel_radius_m"]
        return 0.0 if v_wheel < 1e-3 else float(np.clip(1.0 - abs(v_body) / v_wheel, 0.0, 1.0))

    def body_rock_contact(self) -> bool:
        """True if any contact report pairs the chassis with a rock prim."""
        try:
            from omni.physx import get_physx_simulation_interface   # noqa: PLC0415
        except ImportError:
            return False
        rc = self.cfg["rover"]["prim_path"]
        for header, _ in get_physx_simulation_interface().get_contact_report():
            a, b = str(header.actor0), str(header.actor1)
            if (a.startswith(rc) and b.startswith("/World/Rocks")) or (b.startswith(rc) and a.startswith("/World/Rocks")):
                return True
        return False

    def apply_terramechanics(self, k_soil: float) -> None:
        """Hook for a wheel-soil plugin (Bekker/Wong sinkage). Rigid PhysX ignores it."""
        return None

    def capture_frame(self, path: Path) -> None:
        import omni.kit.viewport.utility as vu                           # noqa: PLC0415
        vp = vu.get_active_viewport()
        vp.set_active_camera(self.cfg["camera"]["prim_path"])
        vu.capture_viewport_to_file(vp, str(path))

    def close(self) -> None:
        self.sim_app.close()


# --------------------------------------------------------------------------- episode loop
def run_episode(scene: IsaacScene, x: np.ndarray, seed: int, cfg: dict, camp: dict,
                frames_dir: Optional[Path]) -> Tuple[dict, Optional[float]]:
    sp = scene_from_x(x)
    dem = scene.build_terrain(x, seed, sp)
    scene.set_sun(sp)
    scene.spawn_rover(dem)
    scene.apply_terramechanics(sp.terramechanics_k_soil)
    scene.reset()

    cc = cfg["controller"]
    ctl = GoalSeekController(v_max=cc["v_max"], w_max=cc["w_max"], k_ang=cc["k_ang"], arrive_tol=cc["arrive_tol_m"])
    mon = EpisodeMonitor(t_max=camp["episode"]["t_max_s"])
    rng = np.random.default_rng(seed ^ 0xA57)
    dt = cfg["app"]["physics_dt"]
    settle = int(camp["episode"]["settle_s"] / dt)
    for _ in range(settle):
        scene.step(render=False)
    # odometry proxy: integrate commanded motion with per-metre drift (VO-like) unless gt_pose
    est = np.array(cfg["rover"]["start_xy"], float)
    prev = scene.rover_pose(0.0)
    t, k, next_frame, wall0 = 0.0, 0, 0.0, time.perf_counter()
    render = frames_dir is not None
    outcome = None
    while outcome is None:
        p = scene.rover_pose(t)
        if cc["gt_pose"]:
            est[:] = (p.x, p.y)
        else:
            d = math.hypot(p.x - prev.x, p.y - prev.y)
            est += np.array([p.x - prev.x, p.y - prev.y]) + rng.normal(0, cc["vo_drift_per_m"] * d, 2)
        v_body = math.hypot(p.x - prev.x, p.y - prev.y) / dt
        prev = p
        v, w = ctl(est[0], est[1], p.yaw)
        scene.drive(v, w)
        slip = scene.wheel_slip(v_body)
        outcome = mon.update(p, slip=slip, visibility=1.0)
        if outcome is None and ctl.halted:
            outcome = mon.policy_halted()
        if outcome is None and scene.body_rock_contact():
            outcome = mon.collision()
        if render and t >= next_frame:
            scene.capture_frame(frames_dir / f"f{k:05d}.png")
            next_frame += cfg["camera"]["frame_every_s"]
        scene.step(render=render and (k % max(int(cfg["app"]["rendering_dt"] / dt), 1) == 0))
        t += dt
        k += 1
        if time.perf_counter() - wall0 > camp["episode"]["wall_timeout_s"]:
            outcome = mon._end("timeout")
    m = mon.metrics()
    vo_err = None if cc["gt_pose"] else float(math.hypot(est[0] - prev.x, est[1] - prev.y))
    return m, vo_err


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(REPO / "isaac/configs/lunar_scene.yaml"))
    ap.add_argument("--campaign", default=str(REPO / "isaac/configs/campaign.yaml"))
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--mode", help="proposal | prior:P2 | replay:runs/c1.npz (overrides campaign.yaml)")
    ap.add_argument("--csv")
    ap.add_argument("--no-headless", action="store_true")
    ap.add_argument("--write-frames", action="store_true")
    ap.add_argument("--gt-pose", action="store_true")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(a.scene).read_text())
    camp = yaml.safe_load(Path(a.campaign).read_text())
    if a.mode:
        camp["sampling"]["mode"] = a.mode
    if a.gt_pose:
        cfg["controller"]["gt_pose"] = True
    n = a.episodes or camp["sampling"]["episodes"]
    seed = a.seed if a.seed is not None else camp["sampling"]["seed"]
    X, seeds, source = sample_plan(camp, n, seed)
    csv_path = Path(a.csv or camp["logging"]["csv"])
    frames_root = Path(camp["logging"]["frames_dir"]) if (a.write_frames or cfg["camera"]["write_frames"]) else None

    scene = IsaacScene(cfg, headless=not a.no_headless)
    log = EpisodeLog(csv_path, simulator="isaac-standalone/rigid",
                     controller="goalseek-P" + ("/gt" if cfg["controller"]["gt_pose"] else "/vo-proxy"),
                     gt_pose=cfg["controller"]["gt_pose"], batch_id=f"standalone-{seed}")
    start = log.index
    print(f"[astraeus-isaac] {n} episodes, mode={camp['sampling']['mode']} source={source}, resuming at {start} -> {csv_path}")
    try:
        for i in range(start, n):
            x, s = X[i], int(seeds[i])
            fdir = (frames_root / f"ep{i:05d}") if frames_root else None
            if fdir:
                fdir.mkdir(parents=True, exist_ok=True)
            w0 = time.perf_counter()
            m, vo = run_episode(scene, x, s, cfg, camp, fdir)
            row = log.write(x, s, m, source=source, vo_error=vo, wall_s=time.perf_counter() - w0)
            print(f"  ep {i:4d} seed {s:<11d} {m['outcome']:<9s} t={m['t_end']:6.1f}s dist={m['final_dist']:5.2f} "
                  f"tilt={m['max_tilt_deg']:4.1f} wall={row['wall_s']}s")
    finally:
        log.close()
        scene.close()
    print(f"done. ingest with:  python -m astraeus ingest runs/isaac.npz {csv_path}"
          + ("" if source == 0 else " --not-from-proposal"))


if __name__ == "__main__":
    main()
