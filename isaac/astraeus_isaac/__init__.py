"""Isaac-Sim-independent helpers shared by the standalone environment and the
ROS 2 / OmniLRS bridge.

Nothing here imports `omni`, `pxr`, `isaacsim` or `rclpy`, so this package is
importable (and unit-tested) on a machine without Isaac Sim. The two entry
points that do need the simulator import it lazily *after* SimulationApp has
been created, as Isaac Sim requires.

    scene.py     x (8 disturbance dims) -> concrete scene parameters
    monitor.py   pose stream -> outcome (same rules as astraeus.rover)
    log.py       CSV rows in the exact schema `python -m astraeus ingest` reads
    export.py    astraeus terrain -> heightfield / OBJ mesh / rock list
"""
from .scene import SceneParams, scene_from_x           # noqa: F401
from .monitor import EpisodeMonitor, Pose, OUTCOMES     # noqa: F401
from .log import EpisodeLog, CSV_FIELDS                 # noqa: F401
