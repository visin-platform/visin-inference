"""The models behind the server, and the fakes that stand in for them in tests.

Two small interfaces so the server does not care what runs underneath:

    Detector.detect(image, min_score)   -> [Box(x, y, w, h, name, score)]
    Segmenter.segment(image_id, image, points, box) -> (mask, score)

`UltralyticsDetector` and `UltralyticsSegmenter` import ultralytics (and through
it torch) only when constructed, so the server, its protocol and its tests run
without a GPU stack installed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Protocol

import numpy as np


@dataclasses.dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    name: str
    score: float


@dataclasses.dataclass
class Prompt:
    """One click: where, and whether the pixel belongs to the object (1) or not (0)."""

    x: float
    y: float
    label: int = 1


class Detector(Protocol):
    names: Sequence[str]

    def detect(self, image: np.ndarray, min_score: float) -> list[Box]: ...


class Segmenter(Protocol):
    def segment(
        self,
        image_id: str,
        image: np.ndarray | None,
        points: Sequence[Prompt],
        box: tuple[float, float, float, float] | None,
    ) -> tuple[np.ndarray, float]:
        """Returns a boolean mask the size of the image, and the model's score for it.

        `image` is None when the segmenter already holds this `image_id`; it must
        raise `ImageNeeded` if it does not.
        """
        ...


class ImageNeeded(Exception):
    """The segmenter has no image for this id, so the caller must send it."""


def device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "Apple GPU (mps)"
    except Exception:  # torch absent or misbehaving: say so rather than fail /info
        pass
    return "CPU"


def to_bgr(image: np.ndarray) -> np.ndarray:
    """The server decodes images as RGB; ultralytics, like OpenCV, reads a numpy
    image as BGR. Without this every frame would be seen with red and blue swapped."""
    return np.ascontiguousarray(image[..., ::-1])


class UltralyticsDetector:
    """YOLO through ultralytics: `yolo26s.pt` by default, any ultralytics detector works."""

    def __init__(self, weights: str = "yolo26s.pt") -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self.names: Sequence[str] = list(self._model.names.values())

    def detect(self, image: np.ndarray, min_score: float) -> list[Box]:
        result = self._model.predict(to_bgr(image), conf=min_score, verbose=False)[0]
        boxes: list[Box] = []
        for xyxy, score, cls in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist(), strict=True
        ):
            x1, y1, x2, y2 = xyxy
            boxes.append(Box(x1, y1, x2 - x1, y2 - y1, self.names[int(cls)], float(score)))
        return boxes


class UltralyticsSegmenter:
    """SAM through ultralytics' predictor.

    The image embedding is the slow part (a second or so on a mid-range GPU) and
    the prompt decoder is not, so the embedding is kept for the frame being
    worked on: only the first click on a frame pays for the encoder.
    """

    def __init__(self, weights: str = "sam2.1_b.pt") -> None:
        # SAM 2 weights need their own predictor; the original SAM's will not load them.
        if "sam2" in weights.lower():
            from ultralytics.models.sam import SAM2Predictor as PredictorClass
        else:
            from ultralytics.models.sam import Predictor as PredictorClass

        # save=False: ultralytics otherwise writes every result under ./runs, a file per click.
        overrides = dict(
            conf=0.25, task="segment", mode="predict", imgsz=1024, model=weights, verbose=False, save=False
        )
        self._predictor = PredictorClass(overrides=overrides)
        self._image_id: str | None = None

    def segment(self, image_id, image, points, box):
        if image_id != self._image_id:
            if image is None:
                raise ImageNeeded(image_id)
            self._predictor.set_image(to_bgr(image))
            self._image_id = image_id
        kwargs: dict = {}
        if points:
            # One more level of nesting than looks necessary: a flat list of
            # points asks for one object per point; nested, they are all
            # clicks on the same object, which is what the editor sends.
            kwargs["points"] = [[[p.x, p.y] for p in points]]
            kwargs["labels"] = [[p.label for p in points]]
        if box is not None:
            x, y, w, h = box
            kwargs["bboxes"] = [x, y, x + w, y + h]
        results = self._predictor(**kwargs)
        masks = results[0].masks.data
        mask = masks[0].cpu().numpy() > 0.5
        scores = getattr(results[0].boxes, "conf", None)
        score = float(scores[0]) if scores is not None and len(scores) else 1.0
        return mask, score
