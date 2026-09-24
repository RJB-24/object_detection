"""Central configuration (env-overridable)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


# --- Object detection (YOLOv8) ---
YOLO_MODEL_NAME: str = _env("YOLO_MODEL_NAME", "yolov8n.pt")
# Swap to yolov8s.pt / yolov8m.pt / yolov8l.pt for higher accuracy (slower).
YOLO_CONF: float = _env_float("YOLO_CONF", 0.25)
YOLO_IOU: float = _env_float("YOLO_IOU", 0.45)

# --- Face detection (YuNet) + recognition (SFace) ---
FACE_DETECTION_MODEL: Path = ROOT / _env(
    "FACE_DETECTION_MODEL", "models/face_detection_yunet_2023mar.onnx"
)
FACE_RECOGNITION_MODEL: Path = ROOT / _env(
    "FACE_RECOGNITION_MODEL", "models/face_recognition_sface_2021dec.onnx"
)
FACE_SCORE_THRESH: float = _env_float("FACE_SCORE_THRESH", 0.6)
FACE_NMS_THRESH: float = _env_float("FACE_NMS_THRESH", 0.3)
FACE_TOP_K: int = _env_int("FACE_TOP_K", 5000)
# Cosine-similarity match threshold for SFace (OpenCV default: 0.363).
FACE_MATCH_THRESH: float = _env_float("FACE_MATCH_THRESH", 0.363)

# --- Face database ---
FACE_DB_PATH: Path = ROOT / _env("FACE_DB_PATH", "data/face_db.pkl")
KNOWN_FACES_DIR: Path = ROOT / _env("KNOWN_FACES_DIR", "data/known_faces")

# --- API limits ---
MAX_FILE_MB: int = _env_int("MAX_FILE_MB", 10)
MAX_IMAGE_SIDE: int = _env_int("MAX_IMAGE_SIDE", 1280)

# --- Model download URLs (OpenCV Zoo) ---
YUNET_URL: str = _env(
    "YUNET_URL",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
SFACE_URL: str = _env(
    "SFACE_URL",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
)

# --- Server ---
HOST: str = _env("HOST", "0.0.0.0")
PORT: int = _env_int("PORT", 8000)
CORS_ORIGINS: str = _env("CORS_ORIGINS", "*")

COCO_LABELS_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
