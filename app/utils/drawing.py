"""Annotated-image rendering."""
from __future__ import annotations

import cv2
import numpy as np

# Distinct BGR palette
_PALETTE = [
    (56, 189, 248), (52, 211, 153), (251, 191, 36), (244, 114, 182),
    (167, 139, 250), (251, 113, 133), (45, 212, 191), (253, 224, 71),
    (96, 165, 250), (248, 113, 113),
]
_GREEN = (52, 211, 153)
_RED = (251, 113, 133)
_LABEL_BG = (15, 23, 42)


def _boxed_label(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 label: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y0 = max(0, y1 - th - 8)
    cv2.rectangle(img, (x1, y0), (x1 + tw + 8, y1), color, -1)
    cv2.putText(img, label, (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _LABEL_BG, 2, cv2.LINE_AA)


def draw_detections(bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = bgr.copy()
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        label = f"{d.get('class_name', '?')} {d.get('confidence', 0.0):.2f}"
        _boxed_label(out, x1, y1, x2, y2, label,
                     _PALETTE[d.get("class_id", i) % len(_PALETTE)])
    return out


def draw_faces(bgr: np.ndarray, faces: list[dict]) -> np.ndarray:
    """Draw face boxes. Recognition results ({name, score, matched}) render
    green/red; plain detections ({confidence}) render in palette colors."""
    out = bgr.copy()
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = (int(v) for v in f["bbox"])
        if "matched" in f:
            label = f"{f['name']} {f['score']:.2f}"
            color = _GREEN if f["matched"] else _RED
        else:
            label = f"face {f['confidence']:.2f}"
            color = _PALETTE[i % len(_PALETTE)]
        _boxed_label(out, x1, y1, x2, y2, label, color)
    return out
