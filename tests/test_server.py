import base64
import io
import json
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from visin_infer.backends import Box, ImageNeeded
from visin_infer.rle import decode_mask, encode_mask
from visin_infer.server import Inference, create_server

ORIGIN = "https://label.example.test"


class FakeDetector:
    names = ["car", "person"]

    def __init__(self):
        self.calls = []

    def detect(self, image, min_score):
        self.calls.append((image.shape, min_score))
        return [Box(10, 20, 30, 40, "car", 0.91), Box(1, 2, 3, 4, "person", 0.5)]


class FakeSegmenter:
    """Paints a 4x4 square at the first point, and remembers which image it holds."""

    def __init__(self):
        self.held = None
        self.encoded = 0

    def segment(self, image_id, image, points, box):
        if image_id != self.held:
            if image is None:
                raise ImageNeeded(image_id)
            self.held = image_id
            self.encoded += 1
        mask = np.zeros((20, 30), dtype=bool)
        if points:
            x, y = int(points[0].x), int(points[0].y)
        else:
            x, y = int(box[0]), int(box[1])
        mask[y : y + 4, x : x + 4] = True
        return mask, 0.8


def png(width=30, height=20):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class ServerTest(unittest.TestCase):
    token = None

    @classmethod
    def setUpClass(cls):
        cls.detector = FakeDetector()
        cls.segmenter = FakeSegmenter()
        cls.server = create_server(
            Inference(cls.detector, cls.segmenter, device="Test GPU"), [ORIGIN + "/"], port=0, token=cls.token
        )
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, method, path, body=None, headers=None):
        request = urllib.request.Request(self.base + path, data=body, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            with error:
                return error.code, dict(error.headers), error.read()

    def json(self, method, path, body=None, headers=None):
        status, response_headers, data = self.call(method, path, body, headers)
        return status, response_headers, json.loads(data or b"{}")

    def post_json(self, path, payload, headers=None):
        return self.json(
            "POST",
            path,
            json.dumps(payload).encode(),
            {"Content-Type": "application/json", **(headers or {})},
        )


class InfoTest(ServerTest):
    def test_says_what_is_loaded(self):
        status, _, body = self.json("GET", "/info")
        self.assertEqual(status, 200)
        self.assertEqual(body["device"], "Test GPU")
        self.assertEqual(body["classes"], ["car", "person"])
        self.assertEqual(body["supports"], {"detect": True, "segment": True})

    def test_unknown_routes_are_404(self):
        self.assertEqual(self.json("GET", "/nope")[0], 404)
        self.assertEqual(self.json("POST", "/nope", b"")[0], 404)


class DetectTest(ServerTest):
    def test_returns_boxes_in_image_pixels(self):
        status, _, body = self.json("POST", "/detect?minScore=0.4", png(), {"Content-Type": "image/png"})
        self.assertEqual(status, 200)
        self.assertEqual(body["boxes"][0], {"bbox": [10, 20, 30, 40], "class": "car", "score": 0.91})
        self.assertEqual(self.detector.calls[-1], ((20, 30, 3), 0.4))

    def test_clamps_and_defaults_the_threshold(self):
        self.json("POST", "/detect", png())
        self.assertEqual(self.detector.calls[-1][1], 0.25)
        self.json("POST", "/detect?minScore=7", png())
        self.assertEqual(self.detector.calls[-1][1], 1.0)
        self.assertEqual(self.json("POST", "/detect?minScore=high", png())[0], 400)

    def test_refuses_nonfinite_thresholds_without_running_the_model(self):
        before = len(self.detector.calls)
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                self.assertEqual(self.json("POST", f"/detect?minScore={value}", png())[0], 400)
        self.assertEqual(len(self.detector.calls), before)

    def test_refuses_a_negative_content_length_without_waiting_for_eof(self):
        self.assertEqual(self.json("POST", "/detect", b"", {"Content-Length": "-1"})[0], 400)

    def test_a_bad_image_is_an_error_not_a_crash(self):
        status, _, body = self.json("POST", "/detect", b"not an image")
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        self.assertEqual(self.json("GET", "/info")[0], 200)


class SegmentTest(ServerTest):
    def mask_of(self, body):
        return decode_mask(base64.b64decode(body["rle"]))

    def test_needs_the_image_once_then_remembers_it(self):
        payload = {"imageId": "needs-image", "points": [{"x": 5, "y": 6, "label": 1}]}
        status, _, body = self.post_json("/segment", payload)
        self.assertEqual((status, body), (409, {"error": "image_needed"}))

        with_image = {**payload, "image": base64.b64encode(png()).decode()}
        status, _, body = self.post_json("/segment", with_image)
        self.assertEqual(status, 200)
        self.assertEqual((body["width"], body["height"]), (30, 20))
        self.assertEqual(body["bbox"], [5, 6, 4, 4])
        mask = self.mask_of(body)
        self.assertEqual(int(mask.sum()), 16)
        self.assertEqual(int(mask[6, 5]), 1)

        before = self.segmenter.encoded
        status, _, _ = self.post_json("/segment", payload)
        self.assertEqual(status, 200)
        self.assertEqual(self.segmenter.encoded, before)

    def test_accepts_a_box_prompt(self):
        status, _, body = self.post_json(
            "/segment",
            {"imageId": "frame-1", "image": base64.b64encode(png()).decode(), "box": [2, 3, 10, 10]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["bbox"], [2, 3, 4, 4])

    def test_refuses_what_it_cannot_use(self):
        self.assertEqual(self.post_json("/segment", {"points": [{"x": 1, "y": 1}]})[0], 400)
        self.assertEqual(self.post_json("/segment", {"imageId": "a"})[0], 400)
        self.assertEqual(self.post_json("/segment", {"imageId": "a", "box": [1, 2, 3], "points": []})[0], 400)
        self.assertEqual(self.json("POST", "/segment", b"{nope")[0], 400)

    def test_malformed_prompts_are_client_errors(self):
        invalid = [
            None,
            [],
            "not an object",
            {"imageId": "a", "points": "not a list"},
            {"imageId": "a", "points": [None]},
            {"imageId": "a", "points": [{"x": 1}]},
            {"imageId": "a", "points": [{"x": "bad", "y": 1}]},
            {"imageId": "a", "points": [{"x": float("nan"), "y": 1}]},
            {"imageId": "a", "points": [{"x": 1, "y": 1, "label": 2}]},
            {"imageId": "a", "box": [0, 0, -1, 2]},
            {"imageId": "a", "box": [0, 0, 2, float("inf")]},
            {"imageId": "a", "box": [0, 0, 2, 2], "image": "bad base64!"},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                self.assertEqual(self.post_json("/segment", payload)[0], 400)
        self.assertEqual(self.json("POST", "/segment", b"\xff")[0], 400)
        self.assertEqual(self.json("GET", "/info")[0], 200)


class CorsTest(ServerTest):
    def test_answers_the_preflight_for_the_allowed_origin_only(self):
        status, headers, _ = self.call("OPTIONS", "/detect", headers={"Origin": ORIGIN})
        self.assertEqual(status, 204)
        self.assertEqual(headers["Access-Control-Allow-Origin"], ORIGIN)
        self.assertEqual(headers["Access-Control-Allow-Private-Network"], "true")
        self.assertIn("X-Visin-Token", headers["Access-Control-Allow-Headers"])

        status, headers, _ = self.call("OPTIONS", "/detect", headers={"Origin": "https://evil.example.test"})
        self.assertEqual(status, 403)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_refuses_a_request_from_another_page(self):
        status, headers, _ = self.json("GET", "/info", headers={"Origin": "https://evil.example.test"})
        self.assertEqual(status, 403)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_marks_responses_to_the_allowed_page(self):
        _, headers, _ = self.json("GET", "/info", headers={"Origin": ORIGIN})
        self.assertEqual(headers["Access-Control-Allow-Origin"], ORIGIN)


class TokenTest(ServerTest):
    token = "s3cret"

    def test_wants_the_token_when_one_is_set(self):
        self.assertEqual(self.json("GET", "/info")[0], 401)
        self.assertEqual(self.json("GET", "/info", headers={"X-Visin-Token": "wrong"})[0], 401)
        self.assertEqual(self.json("GET", "/info", headers={"X-Visin-Token": "s3cret"})[0], 200)


class RleTest(unittest.TestCase):
    def test_round_trips_and_writes_runs(self):
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:3, 2:5] = 1
        mask[5, :] = 2
        data = encode_mask(mask)
        self.assertEqual(data[:5], b"VMSK\x01")
        self.assertEqual(np.array_equal(decode_mask(data), mask), True)
        self.assertLess(len(data), 60)

    def test_long_runs_and_edge_cases(self):
        big = np.ones((1000, 1000), dtype=np.uint8)
        self.assertEqual(np.array_equal(decode_mask(encode_mask(big)), big), True)
        with self.assertRaises(ValueError):
            decode_mask(b"nope")
        with self.assertRaises(ValueError):
            encode_mask(np.zeros(5, dtype=np.uint8))


class UnsupportedTest(unittest.TestCase):
    def test_a_server_with_no_models_says_so(self):
        server = create_server(Inference(None, None, device="CPU"), [ORIGIN], port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            info = json.loads(urllib.request.urlopen(base + "/info").read())
            self.assertEqual(info["supports"], {"detect": False, "segment": False})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(urllib.request.Request(base + "/detect", data=png(), method="POST"))
            self.assertEqual(caught.exception.code, 501)
            caught.exception.close()
        finally:
            server.shutdown()
            server.server_close()


class BackendHelpersTest(unittest.TestCase):
    def test_hands_ultralytics_its_bgr_order(self):
        from visin_infer.backends import to_bgr

        rgb = np.zeros((1, 2, 3), dtype=np.uint8)
        rgb[0, 0] = (255, 0, 10)
        bgr = to_bgr(rgb)
        self.assertEqual(bgr[0, 0].tolist(), [10, 0, 255])
        self.assertTrue(bgr.flags["C_CONTIGUOUS"])


if __name__ == "__main__":
    unittest.main()
