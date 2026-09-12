"""Append-only CSV log in the schema `python -m astraeus ingest` consumes.

Every row is flushed immediately so an overnight run that dies keeps what it
produced. `resume_index()` lets a restarted batch continue numbering."""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

from .scene import DIMS

CSV_FIELDS = list(DIMS) + [
    "index", "seed", "outcome", "source",
    "t_end", "final_dist", "max_tilt_deg", "max_slip", "max_sinkage",
    "min_visibility", "mean_visibility", "vo_error", "severity",
    # provenance (ignored by ingest, kept for audit)
    "simulator", "controller", "gt_pose", "wall_s", "batch_id",
]


class EpisodeLog:
    def __init__(self, path: Path, simulator: str, controller: str, gt_pose: bool, batch_id: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.simulator, self.controller, self.gt_pose, self.batch_id = simulator, controller, gt_pose, batch_id
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            self._w.writeheader()
            self._fh.flush()
        self.index = self.resume_index(self.path)

    @staticmethod
    def resume_index(path: Path) -> int:
        path = Path(path)
        if not path.exists():
            return 0
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        return max((int(r["index"]) for r in rows if r.get("index")), default=-1) + 1

    def write(self, x: Sequence[float], seed: int, metrics: Dict[str, object], source: int = 0,
              vo_error: Optional[float] = None, wall_s: float = 0.0) -> Dict[str, object]:
        row: Dict[str, object] = dict(zip(DIMS, (float(v) for v in x)))
        row.update(metrics)
        row.update({
            "index": self.index, "seed": int(seed), "source": int(source),
            "vo_error": "" if vo_error is None else round(float(vo_error), 4),
            "simulator": self.simulator, "controller": self.controller,
            "gt_pose": int(self.gt_pose), "wall_s": round(wall_s, 2), "batch_id": self.batch_id,
        })
        self._w.writerow(row)
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.index += 1
        return row

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "EpisodeLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
