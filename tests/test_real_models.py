"""The real models, on the real GPU: YOLO26 for boxes, SAM 2.1 for click-to-segment.

Opt in with VISIN_INFER_TEST_MODELS=1. Skipped unless ultralytics is installed and the weights are in `models/` (or
$VISIN_INFER_MODELS); the protocol tests in test_server.py need neither. Run
them after `visin-infer serve` has fetched the weights once, or after
`visin-infer fetch`:

    VISIN_INFER_TEST_MODELS=1 .venv/bin/python -m unittest tests.test_real_models -v

The pictures are the two that ship with ultralytics: a bus with people beside
it, and two people in front of a wall.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np

MODELS = Path(os.environ.get("VISIN_INFER_MODELS", Path(__file__).resolve().parent.parent / "models"))
DETECTOR = MODELS / "yolo26s.pt"
SEGMENTER = MODELS / "sam2.1_b.pt"

ENABLED = os.environ.get("VISIN_INFER_TEST_MODELS") == "1"

try:
    if not ENABLED:
        raise ImportError("real-model tests are opt-in")
    import ultralytics  # noqa: F401
    from ultralytics.utils import ASSETS

    HAVE_ULTRALYTICS = True
except ImportError:  # pragma: no cover - depends on the machine
    HAVE_ULTRALYTICS = False

READY = HAVE_ULTRALYTICS and DETECTOR.exists() and SEGMENTER.exists()
if ENABLED and not READY:
    raise RuntimeError("real-model tests require ultralytics and both cached model weights")

REASON = "set VISIN_INFER_TEST_MODELS=1; needs ultralytics and models/yolo26s.pt + models/sam2.1_b.pt"


def iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    inter_w = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    inter_h = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


@unittest.skipUnless(READY, REASON)
class RealModelsTest(unittest.TestCase):
    detector = None
    segmenter = None

    @classmethod
    def setUpClass(cls):
        from visin_infer.backends import UltralyticsDetector, UltralyticsSegmenter
        from visin_infer.server import load_image

        cls.detector = UltralyticsDetector(str(DETECTOR))
        cls.segmenter = UltralyticsSegmenter(str(SEGMENTER))
        cls.bus_bytes = (ASSETS / "bus.jpg").read_bytes()
        cls.bus = load_image(cls.bus_bytes)
        cls.zidane = load_image((ASSETS / "zidane.jpg").read_bytes())

    def bus_box(self):
        boxes = self.detector.detect(self.bus, 0.25)
        buses = [b for b in boxes if b.name == "bus"]
        self.assertTrue(buses, f"no bus among {[b.name for b in boxes]}")
        return max(buses, key=lambda b: b.score)

    def test_detects_the_bus_and_the_people(self):
        boxes = self.detector.detect(self.bus, 0.25)
        names = [b.name for b in boxes]
        self.assertIn("bus", names)
        self.assertGreaterEqual(names.count("person"), 3)
        height, width = self.bus.shape[:2]
        for b in boxes:
            self.assertGreaterEqual(b.x, -1)
            self.assertGreaterEqual(b.y, -1)
            self.assertLessEqual(b.x + b.w, width + 1)
            self.assertLessEqual(b.y + b.h, height + 1)
            self.assertGreaterEqual(b.score, 0.25)
        bus = self.bus_box()
        self.assertGreater(bus.score, 0.7)
        # The bus fills most of the picture's width.
        self.assertGreater(bus.w, width * 0.6)

    def test_the_threshold_drops_unsure_boxes(self):
        loose = self.detector.detect(self.bus, 0.1)
        strict = self.detector.detect(self.bus, 0.8)
        self.assertLess(len(strict), len(loose))
        self.assertTrue(all(b.score >= 0.8 for b in strict))

    def test_sees_colours_the_right_way_round(self):
        # A red-blue swap leaves a bus detectable, but costs confidence; the
        # detector must see the picture as it is.
        from visin_infer.backends import Box  # noqa: F401

        straight = self.bus_box().score
        swapped = max(
            (b.score for b in self.detector.detect(self.bus[..., ::-1].copy(), 0.25) if b.name == "bus"),
            default=0,
        )
        self.assertGreaterEqual(straight, swapped - 0.02)

    def test_a_click_on_the_bus_segments_the_bus(self):
        from visin_infer.backends import Prompt

        bus = self.bus_box()
        x, y = bus.x + bus.w * 0.5, bus.y + bus.h * 0.4
        mask, score = self.segmenter.segment("bus", self.bus, [Prompt(x, y, 1)], None)
        self.assertEqual(mask.shape, self.bus.shape[:2])
        self.assertTrue(mask[int(y), int(x)], "the clicked pixel is not in the mask")
        ys, xs = np.nonzero(mask)
        mask_box = [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]
        self.assertGreater(iou(mask_box, [bus.x, bus.y, bus.w, bus.h]), 0.6)
        self.assertGreater(score, 0.5)

    def test_a_box_prompt_segments_what_is_inside_it(self):
        bus = self.bus_box()
        mask, _ = self.segmenter.segment("bus-box", self.bus, [], (bus.x, bus.y, bus.w, bus.h))
        ys, xs = np.nonzero(mask)
        mask_box = [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]
        self.assertGreater(iou(mask_box, [bus.x, bus.y, bus.w, bus.h]), 0.7)

    def test_a_negative_click_is_read_together_with_the_positive_one(self):
        # SAM reads points as clicks on one object only when they are nested; a
        # flat list gives exactly the positive click's mask, the negative ignored.
        # So: the same clicks, with and without a negative one, must differ.
        from visin_infer.backends import Prompt

        bus = self.bus_box()
        person = max(
            (b for b in self.detector.detect(self.bus, 0.5) if b.name == "person"), key=lambda b: b.x
        )
        inside = Prompt(bus.x + bus.w * 0.5, bus.y + bus.h * 0.4, 1)
        not_this = Prompt(person.x + person.w / 2, person.y + person.h * 0.4, 0)
        first, _ = self.segmenter.segment("neg", self.bus, [inside], None)
        second, _ = self.segmenter.segment("neg", None, [inside, not_this], None)
        self.assertFalse(np.array_equal(first, second), "the negative click was ignored")

    def test_a_click_on_a_person_stays_on_that_person(self):
        from visin_infer.backends import Prompt

        person = max(
            (b for b in self.detector.detect(self.bus, 0.5) if b.name == "person"), key=lambda b: b.score
        )
        mask, _ = self.segmenter.segment(
            "person", self.bus, [Prompt(person.x + person.w / 2, person.y + person.h * 0.4, 1)], None
        )
        ys, xs = np.nonzero(mask)
        self.assertGreater(xs.size, 1000)
        # Everything SAM picked is within (a little around) that person's box.
        self.assertGreaterEqual(xs.min(), person.x - 10)
        self.assertLessEqual(xs.max(), person.x + person.w + 10)

    def test_keeps_the_embedding_for_the_frame_and_asks_for_a_new_one(self):
        from visin_infer.backends import ImageNeeded, Prompt

        click = [Prompt(600, 350, 1)]
        started = time.perf_counter()
        self.segmenter.segment("zidane", self.zidane, click, None)
        first = time.perf_counter() - started
        started = time.perf_counter()
        self.segmenter.segment("zidane", None, [Prompt(600, 350, 1), Prompt(300, 400, 1)], None)
        again = time.perf_counter() - started
        # Only the first click on a frame pays for the encoder.
        self.assertLess(again, first)
        with self.assertRaises(ImageNeeded):
            self.segmenter.segment("never-sent", None, click, None)


@unittest.skipUnless(READY, REASON)
class RealServerTest(unittest.TestCase):
    """The same models behind the HTTP server, called the way the editor calls them."""

    ORIGIN = "http://localhost:3008"

    @classmethod
    def setUpClass(cls):
        from visin_infer.backends import UltralyticsDetector, UltralyticsSegmenter
        from visin_infer.server import Inference, create_server

        cls.server = create_server(
            Inference(UltralyticsDetector(str(DETECTOR)), UltralyticsSegmenter(str(SEGMENTER))),
            [cls.ORIGIN],
            port=0,
        )
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.bus = (ASSETS / "bus.jpg").read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, path, body, content_type):
        request = urllib.request.Request(
            self.base + path,
            data=body,
            method="POST",
            headers={"Content-Type": content_type, "Origin": self.ORIGIN},
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], self.ORIGIN)
            return json.loads(response.read())

    def test_info_names_the_gpu_and_the_classes(self):
        info = json.loads(urllib.request.urlopen(self.base + "/info").read())
        self.assertEqual(info["supports"], {"detect": True, "segment": True})
        self.assertIn("bus", info["classes"])
        self.assertEqual(len(info["classes"]), 80)
        self.assertNotEqual(info["device"], "")

    def test_detect_and_segment_over_http(self):
        from visin_infer.rle import decode_mask

        boxes = self.post("/detect?minScore=0.5", self.bus, "image/jpeg")["boxes"]
        bus = next(b for b in boxes if b["class"] == "bus")
        x, y, w, h = bus["bbox"]

        payload = {
            "imageId": "http-bus",
            "image": base64.b64encode(self.bus).decode(),
            "points": [{"x": x + w / 2, "y": y + h * 0.4, "label": 1}],
        }
        result = self.post("/segment", json.dumps(payload).encode(), "application/json")
        mask = decode_mask(base64.b64decode(result["rle"]))
        self.assertEqual(mask.shape, (result["height"], result["width"]))
        self.assertGreater(iou(result["bbox"], bus["bbox"]), 0.6)
        self.assertEqual(int(mask.sum()) > 10_000, True)


if __name__ == "__main__":
    unittest.main()
