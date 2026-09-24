"""Face detection (YuNet) + recognition (SFace) via OpenCV DNN.

Why YuNet+SFace?
- Ships as small ONNX files, runs on CPU, no dlib/conda pain.
- Accurate enough for a college/industry image-processing project
  and genuinely deployable (Docker / Render / Hugging Face).
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from app import config


class FaceEngineError(RuntimeError):
    pass


def _require_model(path: Path, kind: str) -> Path:
    if not path.exists():
        raise FaceEngineError(
            f"Missing {kind} model: {path}. Run `python scripts/download_models.py` "
            "or start the server once with internet access (auto-download)."
        )
    return path


class FaceEngine:
    """Owns YuNet detector + SFace recognizer instances.

    OpenCV's FaceDetectorYN needs setInputSize() per image size, so we guard
    the detector with a lock to stay thread-safe under uvicorn.
    """

    def __init__(
        self,
        det_model: Path | None = None,
        rec_model: Path | None = None,
        score_thresh: float | None = None,
        nms_thresh: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self.det_model = det_model or config.FACE_DETECTION_MODEL
        self.rec_model = rec_model or config.FACE_RECOGNITION_MODEL
        self.score_thresh = config.FACE_SCORE_THRESH if score_thresh is None else score_thresh
        self.nms_thresh = config.FACE_NMS_THRESH if nms_thresh is None else nms_thresh
        self.top_k = config.FACE_TOP_K if top_k is None else top_k
        self._detector = None
        self._recognizer = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    # -- lifecycle -----------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._detector is not None and self._recognizer is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> "FaceEngine":
        if self.loaded:
            return self
        with self._lock:
            if self.loaded:
                return self
            try:
                _require_model(Path(self.det_model), "YuNet face detection")
                _require_model(Path(self.rec_model), "SFace face recognition")
                backend = cv2.dnn.DNN_BACKEND_OPENCV
                target = cv2.dnn.DNN_TARGET_CPU
                try:
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        target = cv2.dnn.DNN_TARGET_CUDA
                except Exception:  # noqa: BLE001
                    pass
                self._detector = cv2.FaceDetectorYN_create(
                    str(self.det_model),
                    "",
                    (320, 320),
                    self.score_thresh,
                    self.nms_thresh,
                    self.top_k,
                    backend,
                    target,
                )
                self._recognizer = cv2.FaceRecognizerSF_create(
                    str(self.rec_model), "", backend, target
                )
                self._load_error = None
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                raise FaceEngineError(self._load_error)
        return self

    def status(self) -> dict:
        return {
            "detection_model": str(self.det_model),
            "recognition_model": str(self.rec_model),
            "loaded": self.loaded,
            "error": self._load_error,
            "det_exists": Path(self.det_model).exists(),
            "rec_exists": Path(self.rec_model).exists(),
        }

    # -- detection -----------------------------------------------------
    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """Return faces: {bbox[x1,y1,x2,y2], confidence, landmarks, raw}."""
        self.load()
        h, w = image_bgr.shape[:2]
        with self._lock:
            assert self._detector is not None
            self._detector.setInputSize((w, h))
            _, faces = self._detector.detect(image_bgr)
        out: list[dict] = []
        if faces is None:
            return out
        for f in faces:
            x, y, ww, hh = (float(v) for v in f[:4])
            # landmarks: right eye, left eye, nose, right mouth, left mouth
            lm = []
            try:
                coords = f[4:14].reshape(5, 2)
                lm = [[float(a), float(b)] for a, b in coords]
            except Exception:  # noqa: BLE001
                lm = []
            conf = float(f[14]) if len(f) > 14 else 1.0
            out.append(
                {
                    "bbox": [x, y, x + ww, y + hh],
                    "confidence": conf,
                    "landmarks": lm,
                    "raw": np.asarray(f, dtype=np.float32),
                }
            )
        out.sort(key=lambda d: d["confidence"], reverse=True)
        return out

    # -- recognition ---------------------------------------------------
    def embedding(self, image_bgr: np.ndarray, face_raw: np.ndarray) -> np.ndarray:
        self.load()
        with self._lock:
            assert self._recognizer is not None
            aligned = self._recognizer.alignCrop(image_bgr, face_raw)
            feat = self._recognizer.feature(aligned)
        return np.asarray(feat, dtype=np.float32).flatten()

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).flatten()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)


# Shared singleton used by the API
_engine: FaceEngine | None = None
_engine_lock = threading.Lock()


def get_face_engine() -> FaceEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = FaceEngine()
    return _engine
