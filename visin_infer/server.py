"""The HTTP side: three routes, CORS for one origin, and nothing else.

    GET  /info      what is loaded and on what
    POST /detect    image bytes -> boxes
    POST /segment   JSON {imageId, image?, points, box?} -> a mask

Standard library only (`http.server`), so installing this adds nothing beside
the model stack, and the protocol can be tested without it. It serves one
annotator on one machine; it is not built for load.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import __version__
from .backends import Detector, ImageNeeded, Prompt, Segmenter, device_name
from .rle import encode_mask

MAX_BODY = 64 * 1024 * 1024
DEFAULT_MIN_SCORE = 0.25


def load_image(data: bytes) -> np.ndarray:
    """Image bytes (JPEG, PNG...) as an RGB array."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"))
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise BadRequest("body is not a readable image") from error


def finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BadRequest("prompt coordinates must be finite numbers")
    try:
        number = float(value)
    except OverflowError:
        raise BadRequest("prompt coordinates must be finite numbers") from None
    if not math.isfinite(number):
        raise BadRequest("prompt coordinates must be finite numbers")
    return number


class Inference:
    """What the routes call. Models run one request at a time: a GPU gains nothing from two."""

    def __init__(self, detector: Detector | None, segmenter: Segmenter | None, device: str | None = None):
        self.detector = detector
        self.segmenter = segmenter
        self.device = device or device_name()
        self._lock = threading.Lock()

    def info(self) -> dict[str, Any]:
        return {
            "name": "visin-infer",
            "version": __version__,
            "device": self.device,
            "classes": list(self.detector.names) if self.detector else [],
            "supports": {"detect": self.detector is not None, "segment": self.segmenter is not None},
        }

    def detect(self, image_bytes: bytes, min_score: float) -> dict[str, Any]:
        if not self.detector:
            raise Unsupported("no detector is loaded")
        image = load_image(image_bytes)
        with self._lock:
            boxes = self.detector.detect(image, min_score)
        return {
            "boxes": [
                {
                    "bbox": [round(b.x, 2), round(b.y, 2), round(b.w, 2), round(b.h, 2)],
                    "class": b.name,
                    "score": round(b.score, 4),
                }
                for b in boxes
            ]
        }

    def segment(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.segmenter:
            raise Unsupported("no segmenter is loaded")
        if not isinstance(payload, dict):
            raise BadRequest("body must be a JSON object")
        image_id = payload.get("imageId")
        if not isinstance(image_id, str) or not image_id:
            raise BadRequest("imageId is required")
        raw_points = payload.get("points", [])
        if not isinstance(raw_points, list):
            raise BadRequest("points must be a list")
        points = []
        for point in raw_points:
            if not isinstance(point, dict) or point.get("label", 1) not in (0, 1):
                raise BadRequest("each point needs x, y and label 0 or 1")
            points.append(
                Prompt(finite_number(point.get("x")), finite_number(point.get("y")), point.get("label", 1))
            )
        raw_box = payload.get("box")
        if raw_box is not None and (not isinstance(raw_box, list) or len(raw_box) != 4):
            raise BadRequest("box is [x, y, width, height]")
        box = tuple(finite_number(v) for v in raw_box) if raw_box is not None else None
        if box is not None and (box[2] <= 0 or box[3] <= 0):
            raise BadRequest("box width and height must be positive")
        if not points and box is None:
            raise BadRequest("send at least one point or a box")
        image = None
        if "image" in payload:
            try:
                if not isinstance(payload["image"], str):
                    raise ValueError("not a string")
                image_bytes = base64.b64decode(payload["image"], validate=True)
            except (ValueError, binascii.Error):
                raise BadRequest("image must be base64") from None
            image = load_image(image_bytes)
        with self._lock:
            try:
                mask, score = self.segmenter.segment(image_id, image, points, box)  # type: ignore[arg-type]
            except ImageNeeded:
                raise NeedsImage() from None
        ys, xs = np.nonzero(mask)
        bbox = (
            [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
            if xs.size
            else [0, 0, 0, 0]
        )
        return {
            "rle": base64.b64encode(encode_mask(mask.astype(np.uint8))).decode("ascii"),
            "width": int(mask.shape[1]),
            "height": int(mask.shape[0]),
            "bbox": bbox,
            "score": round(float(score), 4),
        }


class BadRequest(Exception):
    pass


class Unsupported(Exception):
    pass


class NeedsImage(Exception):
    """The editor has to send the image: this server holds none for that id."""


def make_handler(inference: Inference, allowed_origins: set[str], token: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "visin-infer"

        # --- plumbing ---------------------------------------------------

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        def _origin_ok(self) -> str | None:
            origin = self.headers.get("Origin")
            return origin if origin and origin in allowed_origins else None

        def _cors(self) -> None:
            origin = self._origin_ok()
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                # Chrome's Local Network Access: a public page may reach this
                # machine only if the server says it agrees to be reached.
                self.send_header("Access-Control-Allow-Private-Network", "true")

        def _json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read(self) -> bytes:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise BadRequest("Content-Length must be a non-negative integer") from None
            if length < 0:
                raise BadRequest("Content-Length must be a non-negative integer")
            if length > MAX_BODY:
                raise BadRequest("body is too large")
            return self.rfile.read(length)

        def _allowed(self) -> bool:
            # A request with no Origin is not a browser (curl, a script on this
            # machine) and is let through; one from a page must be an allowed page.
            if self.headers.get("Origin") and not self._origin_ok():
                self._json(403, {"error": "origin not allowed"})
                return False
            if token and self.headers.get("X-Visin-Token") != token:
                self._json(401, {"error": "token required"})
                return False
            return True

        def _handle(self, run) -> None:
            try:
                if self._allowed():
                    self._json(200, run())
            except BadRequest as error:
                self._json(400, {"error": str(error)})
            except Unsupported as error:
                self._json(501, {"error": str(error)})
            except NeedsImage:
                self._json(409, {"error": "image_needed"})
            except Exception as error:  # a model failing must not take the server with it
                self._json(500, {"error": f"{type(error).__name__}: {error}"})

        # --- routes -----------------------------------------------------

        def do_OPTIONS(self) -> None:
            origin = self._origin_ok()
            self.send_response(204 if origin else 403)
            if origin:
                self._cors()
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Visin-Token")
                self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            if urlparse(self.path).path != "/info":
                self._json(404, {"error": "not found"})
                return
            self._handle(inference.info)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/detect":
                query = parse_qs(parsed.query)
                try:
                    min_score = float(query.get("minScore", [DEFAULT_MIN_SCORE])[0])
                except ValueError:
                    self._json(400, {"error": "minScore must be a number"})
                    return
                if not math.isfinite(min_score):
                    self._json(400, {"error": "minScore must be a finite number"})
                    return
                self._handle(lambda: inference.detect(self._read(), min(max(min_score, 0.0), 1.0)))
            elif parsed.path == "/segment":

                def run() -> dict[str, Any]:
                    try:
                        payload = json.loads(self._read() or b"{}")
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        raise BadRequest("body is not JSON") from None
                    return inference.segment(payload)

                self._handle(run)
            else:
                self._json(404, {"error": "not found"})

    return Handler


def create_server(
    inference: Inference,
    allowed_origins: list[str],
    host: str = "127.0.0.1",
    port: int = 8000,
    token: str | None = None,
) -> ThreadingHTTPServer:
    """The server, not yet serving. Bound to the loopback by default: anyone who
    can reach it can use the GPU."""
    origins = {origin.rstrip("/") for origin in allowed_origins}
    return ThreadingHTTPServer((host, port), make_handler(inference, origins, token))
