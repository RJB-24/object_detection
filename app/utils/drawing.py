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


def _color(i: int):
    return _PALETTE[i % len(_PALETTE)]


def draw_detections(
    bgr: np.ndarray,
    detections: list[dict],
    label_key: str = "class_name",
) -> np.ndarray:
    out = bgr.copy()
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        conf = d.get("confidence", 0.0)
        label = f"{d.get(label_key, '?')} {conf:.2f}"
        color = _color(d.get("class_id", i))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y0 = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, y0), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2,
                    cv2.LINE_AA)
    return out


def draw_faces(bgr: np.ndarray, faces: list[dict]) -> np.ndarray:
    """Faces: {bbox, confidence, name?, score?, matched?}"""
    out = bgr.copy()
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = (int(v) for v in f["bbox"])
        matched = f.get("matched", False)
        name = f.get("name") or "Unknown"
        score = f.get("score")
        if score is not None:
            label = f"{name} {score:.2f}"
        else:
            label = f"{name} {f.get('confidence', 0):.2f}"
        color = (52, 211, 153) if matched else (251, 113, 133)
        if f.get("name") is None and "confidence" in f and score is None:
            color = _color(i)  # plain detection mode
            label = f"face {f['confidence']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y0 = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, y0), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2,
                    cv2.LINE_AA)
    return out
