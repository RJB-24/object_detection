"""Image decoding / encoding helpers (bytes <-> OpenCV BGR)."""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from app import config


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to BGR numpy array. Fixes EXIF rotation, caps size."""
    if not data:
        raise ValueError("Empty image data")
    max_bytes = config.MAX_FILE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise ValueError(f"Image exceeds {config.MAX_FILE_MB} MB limit")

    pil = Image.open(io.BytesIO(data))
    pil = ImageOps.exif_transpose(pil).convert("RGB")

    # Downscale huge images to keep inference fast
    side = max(pil.size)
    if side > config.MAX_IMAGE_SIDE:
        scale = config.MAX_IMAGE_SIDE / float(side)
        pil = pil.resize(
            (int(pil.width * scale), int(pil.height * scale)), Image.LANCZOS
        )

    rgb = np.asarray(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode_jpeg(bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def to_base64_jpeg(bgr: np.ndarray, quality: int = 88) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(
        encode_jpeg(bgr, quality)
    ).decode("ascii")
