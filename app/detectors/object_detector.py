"""YOLOv8 generic object detector (80 COCO classes)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from app import config

_COCO_FALLBACK = {i: n for i, n in enumerate(config.COCO_LABELS_80)}


class ObjectDetector:
    """Thread-safe lazy singleton around ultralytics.YOLO."""

    _instance: "ObjectDetector | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.YOLO_MODEL_NAME
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._load_error: str | None = None

    @classmethod
    def instance(cls) -> "ObjectDetector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # -- lifecycle -----------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                self._load_error = (
                    "ultralytics is not installed. Run: pip install -r requirements.txt "
                    f"({exc})"
                )
                raise RuntimeError(self._load_error)
            try:
                self._model = YOLO(self.model_name)  # auto-downloads on 1st run
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"Failed to load YOLO model '{self.model_name}': {exc}"
                raise RuntimeError(self._load_error)
            self._load_error = None
            return self._model

    def switch_model(self, model_name: str) -> str:
        """Point at a different checkpoint; weights lazy-load on next predict."""
        with self._model_lock:
            self.model_name = model_name
            self._model = None
            self._load_error = None
        return self.model_name

    @property
    def class_names(self) -> dict[int, str]:
        if self.loaded:
            try:
                names = self._model.names  # type: ignore[union-attr]
                return {int(k): str(v) for k, v in dict(names).items()}
            except Exception:  # noqa: BLE001
                pass
        return dict(_COCO_FALLBACK)

    # -- inference -----------------------------------------------------
    def predict(
        self,
        image_bgr: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
    ) -> list[dict]:
        model = self.load()
        conf = config.YOLO_CONF if conf is None else float(conf)
        iou = config.YOLO_IOU if iou is None else float(iou)

        results = model.predict(image_bgr, conf=conf, iou=iou, verbose=False)
        if not results:
            return []
        r = results[0]
        names = self.class_names
        detections: list[dict] = []
        if r.boxes is None:
            return detections
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for box, c, k in zip(boxes, confs, clss):
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(c),
                    "class_id": int(k),
                    "class_name": names.get(int(k), str(int(k))),
                }
            )
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def status(self) -> dict:
        return {
            "model": self.model_name,
            "loaded": self.loaded,
            "error": self._load_error,
            "classes": len(self.class_names),
            "device": self._device_str(),
        }

    def _device_str(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"


def get_object_detector() -> ObjectDetector:
    return ObjectDetector.instance()


def download_yolo_weights(model_name: str | None = None) -> Path:
    """Pre-download YOLO weights (used by scripts/download_models.py)."""
    from ultralytics import YOLO

    name = model_name or config.YOLO_MODEL_NAME
    model = YOLO(name)
    # ultralytics stores weights under ~/.cache or weights dir; return best-effort path
    try:
        ckpt = getattr(model, "ckpt_path", None) or getattr(model, "weights", name)
        return Path(str(ckpt))
    except Exception:  # noqa: BLE001
        return Path(name)
