"""World-model providers: turn a recorded episode + frame index into an image.

Two providers ship:

* ``LocalProvider`` — the NumPy software renderer in ``astraeus.render``. This is
  the default and needs no network, key, or GPU. Every frame is deterministic
  in (x, seed, frame, camera).
* ``ReactorProvider`` — streams frames from a Reactor (reactor.inc) world model
  using ``reactor-sdk``. It is **opt-in** (``ASTRAEUS_PROVIDER=reactor`` plus
  ``REACTOR_API_KEY``) and it never replaces the local render silently: when
  no live frame is available it returns the local frame tagged ``live: False``
  with the reason, so the UI can say so.

What leaves the machine with Reactor: a text prompt built from the eight
disturbance parameters and the rover state, plus (optionally) the *synthetic*
local render as a conditioning image. Nothing in Astraeus is a real photograph,
so there is no privacy exposure beyond the parameters themselves — but the
badge in the UI still tells you which provider produced each frame.

The SDK surface used here (``Reactor(model_name, api_key, api_url)``,
``connect``, ``on_track`` / ``track.on_frame``, ``request_schema``,
``send_command``, ``upload_file``) is what reactor-sdk 1.5.x exposes. The model
slug and the command names a given model accepts are only knowable from its
own ``request_schema()``; they are environment-overridable rather than
hard-coded for that reason. Run ``python -m astraeus.providers --probe`` with a
key to print them.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from . import render as R
from . import rover as RV
from .priors import DIMS

log = logging.getLogger("astraeus.providers")


@dataclass
class Frame:
    """One rendered frame; ``data_url`` is directly usable as an <img> src."""
    data_url: str
    provider: str
    latency_ms: float
    frame: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"data_url": self.data_url, "provider": self.provider,
                "latency_ms": round(self.latency_ms, 2), "frame": self.frame, "meta": self.meta}


def _png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class WorldModelProvider(ABC):
    name: str = "base"
    live: bool = False

    @abstractmethod
    def render(self, res: RV.EpisodeResult, frame: int = -1, camera: str = "chase",
               size: tuple = (960, 540), quality: int = 1) -> Frame: ...

    def health(self) -> Dict[str, Any]:
        return {"provider": self.name, "live": self.live, "ok": True}

    def close(self) -> None:
        pass


class LocalProvider(WorldModelProvider):
    name = "local"

    def render(self, res, frame=-1, camera="chase", size=(960, 540), quality=1) -> Frame:
        t0 = time.perf_counter()
        scene = R.Scene.from_result(res, frame)
        im = R.render_frame(scene, camera, size, hud=True, quality=quality)
        return Frame(_png_data_url(R.to_png_bytes(im)), self.name,
                     (time.perf_counter() - t0) * 1000.0, scene.frame,
                     {"live": False, "synthetic": True, "camera": camera, "outcome": scene.outcome,
                      "t": scene.t, "n_frames": scene.n_frames})


# ---------------------------------------------------------------------- reactor
DEFAULT_MODEL = os.getenv("REACTOR_MODEL", "reactor/helios")
DEFAULT_API_URL = os.getenv("REACTOR_API_URL", "https://api.reactor.inc")
PROMPT_COMMAND = os.getenv("REACTOR_PROMPT_COMMAND", "set_prompt")
PROMPT_FIELD = os.getenv("REACTOR_PROMPT_FIELD", "prompt")
START_COMMAND = os.getenv("REACTOR_START_COMMAND", "start")
IMAGE_COMMAND = os.getenv("REACTOR_IMAGE_COMMAND", "set_image")
IMAGE_FIELD = os.getenv("REACTOR_IMAGE_FIELD", "image")
CONDITION_ON_LOCAL_RENDER = os.getenv("REACTOR_CONDITION_LOCAL", "1") not in ("0", "false", "")
FRAME_STALE_SECONDS = float(os.getenv("REACTOR_FRAME_STALE_S", "2.0"))


def scene_to_prompt(scene: R.Scene) -> str:
    """The eight disturbance parameters and the rover state as text. Text is a
    lossy channel for geometry; the local render is uploaded as the anchor."""
    p = scene.params
    light = ("grazing polar sunlight, very long shadows" if p["sun_e"] < 3
             else "low sun, long shadows" if p["sun_e"] < 10 else "high sun")
    dust = ("clear vacuum" if p["tau_dust"] < 0.05
            else "thin suspended dust haze" if p["tau_dust"] < 0.25 else "dense regolith dust haze, washed-out contrast")
    rocks = ("almost no rocks" if p["rho_rock"] < 1 else "scattered rocks" if p["rho_rock"] < 5 else "dense rock field")
    soil = ("firm compacted regolith" if p["k_soil"] > 2 else "soft, powdery regolith with deep wheel tracks"
            if p["k_soil"] < 1 else "loose regolith")
    relief = "flat" if p["r_terrain"] < 0.8 else "rolling" if p["r_terrain"] < 1.4 else "rugged, cratered"
    return (f"Photoreal lunar south-pole surface, {relief} terrain on a {p['slope_deg']:.0f} degree slope, "
            f"{soil}, {rocks}, {light} from azimuth {p['sun_psi']:.0f} degrees at {p['sun_e']:.1f} degrees "
            f"elevation, {dust}. A four-wheeled VIPER-class rover with a camera mast and solar panel, "
            f"pitched {scene.telemetry['pitch_deg']:.0f} and rolled {scene.telemetry['roll_deg']:.0f} degrees, "
            f"driving at {scene.telemetry['v']:.2f} m/s with wheel slip {scene.telemetry['slip']:.2f}. "
            f"Black sky, stars, no atmosphere. Status: {scene.outcome}.")


class ReactorProvider(WorldModelProvider):
    """Reactor stream with loud local fallback. The SDK is asyncio/push; our API
    is a synchronous pull, so the SDK runs on a loop we own on a daemon thread
    and ``render`` returns the most recent frame if it is fresh."""

    name = "reactor"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self.api_key = os.getenv("REACTOR_API_KEY", "")
        self.api_url = DEFAULT_API_URL
        self._local = LocalProvider()
        self._latest_png: Optional[bytes] = None
        self._latest_at = 0.0
        self._frames_in = 0
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reactor: Any = None
        self._last_prompt = ""
        self._schema: Optional[Dict[str, Any]] = None
        self._uploads = 0
        self.live = False
        self.status = "not started"
        if not self.api_key:
            self.status = "no REACTOR_API_KEY — serving local frames"
            log.warning(self.status)
            return
        try:
            import reactor_sdk  # noqa: F401
        except ImportError:
            self.status = "reactor-sdk not installed — serving local frames"
            log.warning(self.status)
            return
        threading.Thread(target=self._run_loop, daemon=True, name="reactor-stream").start()
        self.status = "connecting"

    # connection -----------------------------------------------------------
    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main())
        except Exception as exc:                       # noqa: BLE001 — never kill the app
            self.live = False
            self.status = f"error: {exc}"
            log.exception("Reactor stream failed; serving local frames")
        finally:
            loop.close()
            self._loop = None

    async def _main(self) -> None:
        from reactor_sdk import Reactor, TrackKind
        reactor = Reactor(model_name=self.model, api_key=self.api_key, api_url=self.api_url)
        self._reactor = reactor

        @reactor.on_track
        def _on_track(track: Any) -> None:
            if str(track.kind) != str(TrackKind.VIDEO):
                return

            @track.on_frame
            def _on_frame(frame: Any, *_: Any) -> None:
                png = self._encode(frame)
                if png is None:
                    return
                with self._lock:
                    self._latest_png, self._latest_at = png, time.time()
                    self._frames_in += 1
                self.live, self.status = True, "streaming"

        @reactor.on_status
        def _on_status(status: Any) -> None:
            if not self.live:
                self.status = f"status: {status}"

        @reactor.on_error
        def _on_error(err: Any) -> None:
            self.status = f"error: {err}"

        await reactor.connect()
        self.status = "connected"
        try:
            self._schema = await reactor.request_schema()
        except Exception as exc:                       # noqa: BLE001
            log.warning("request_schema failed (continuing): %s", exc)
        try:
            await reactor.send_command(START_COMMAND, {})
        except Exception as exc:                       # noqa: BLE001
            log.warning("%s failed (continuing): %s", START_COMMAND, exc)
        await asyncio.Event().wait()

    @staticmethod
    def _encode(frame: Any) -> Optional[bytes]:
        try:
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:                       # noqa: BLE001
            log.error("frame encode failed: %s", exc)
            return None

    def _submit(self, coro) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed():
            asyncio.run_coroutine_threadsafe(coro, loop)

    async def _condition(self, png: bytes) -> None:
        """Upload the synthetic local render as the geometric anchor."""
        path = os.path.join(tempfile.gettempdir(), "astraeus_reactor_ref.png")
        with open(path, "wb") as f:
            f.write(png)
        ref = await self._reactor.upload_file(path)
        await self._reactor.send_command(IMAGE_COMMAND, {IMAGE_FIELD: ref})
        self._uploads += 1

    # render ---------------------------------------------------------------
    def render(self, res, frame=-1, camera="chase", size=(960, 540), quality=1) -> Frame:
        t0 = time.perf_counter()
        local = self._local.render(res, frame, camera, size, quality)
        scene = R.Scene.from_result(res, frame)
        prompt = scene_to_prompt(scene)
        if self._reactor is not None and prompt != self._last_prompt:
            self._last_prompt = prompt
            self._submit(self._reactor.send_command(PROMPT_COMMAND, {PROMPT_FIELD: prompt}))
            if CONDITION_ON_LOCAL_RENDER:
                self._submit(self._condition(base64.b64decode(local.data_url.split(",", 1)[1])))
        with self._lock:
            png, at, n = self._latest_png, self._latest_at, self._frames_in
        age = time.time() - at if png is not None else None
        if png is not None and age is not None and age < FRAME_STALE_SECONDS:
            return Frame(_png_data_url(png), self.name, (time.perf_counter() - t0) * 1000.0, scene.frame,
                         {"live": True, "synthetic": False, "model": self.model, "age_s": round(age, 3),
                          "frames_in": n, "camera": camera, "outcome": scene.outcome, "t": scene.t,
                          "n_frames": scene.n_frames, "prompt": prompt})
        local.provider = "reactor(fallback:local)"
        local.meta["reason"] = ("no frame yet — " + self.status if png is None
                                else f"last frame is {age:.1f}s old — {self.status}")
        local.meta["prompt"] = prompt
        return local

    def health(self) -> Dict[str, Any]:
        with self._lock:
            n, at = self._frames_in, self._latest_at
        return {"provider": self.name, "live": self.live, "ok": True, "status": self.status,
                "model": self.model, "api_url": self.api_url, "key_present": bool(self.api_key),
                "frames_in": n, "last_frame_age_s": round(time.time() - at, 2) if at else None,
                "schema_known": self._schema is not None, "conditioning_uploads": self._uploads,
                "sends_parameters": True, "sends_real_pixels": False}

    def close(self) -> None:
        loop, reactor = self._loop, self._reactor
        if loop is not None and reactor is not None and not loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(reactor.disconnect(), loop).result(5)
            except Exception:                          # noqa: BLE001
                pass


def make_provider(name: Optional[str] = None) -> WorldModelProvider:
    name = (name or os.getenv("ASTRAEUS_PROVIDER", "local")).lower()
    if name == "reactor":
        return ReactorProvider()
    if name != "local":
        log.warning("unknown provider %r; using local", name)
    return LocalProvider()


if __name__ == "__main__":                             # pragma: no cover - manual probe
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="connect to Reactor and print schema/health")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    prov = ReactorProvider() if a.probe else make_provider()
    if a.probe:
        for _ in range(60):
            if prov.live or prov.status.startswith("error") or prov.status.startswith("no "):
                break
            time.sleep(1)
        print(json.dumps(prov.health(), indent=2))
        if prov._schema:
            print(json.dumps(prov._schema, indent=2)[:4000])
    else:
        print(json.dumps(prov.health(), indent=2), DIMS)
