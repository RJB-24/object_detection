"""Face-match threshold calibration ("fine-tuning" the recognizer).

Given the registered gallery, this module:
  1. Computes genuine scores (same-person embedding pairs) and impostor
     scores (cross-person pairs) using cosine similarity.
  2. Sweeps candidate thresholds and reports FAR / FRR at each.
  3. Recommends the threshold maximizing Youden's J (= TPR - FPR),
     plus the EER operating point.

For a meaningful calibration you want >= 2 identities with >= 2 photos
each. With less data the module says so explicitly and keeps the safe
default (0.363, OpenCV's SFace recommendation).
"""
from __future__ import annotations

import numpy as np

from app import config
from app.detectors.face import FaceEngine
from app.services.face_db import FaceDatabase

DEFAULT_THRESHOLD = config.FACE_MATCH_THRESH


def _pair_scores(embs_a: list[np.ndarray], embs_b: list[np.ndarray]) -> list[float]:
    out = []
    for a in embs_a:
        for b in embs_b:
            out.append(FaceEngine.cosine_similarity(a, b))
    return out


def calibrate(db: FaceDatabase | None = None) -> dict:
    db = db or FaceDatabase()
    snap = db.snapshot()
    names = sorted(snap.keys())

    genuine: list[float] = []
    for embs in snap.values():
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                genuine.append(FaceEngine.cosine_similarity(embs[i], embs[j]))

    impostor: list[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            impostor.extend(_pair_scores(snap[names[i]], snap[names[j]]))

    per_identity: dict[str, dict] = {}
    for name, embs in snap.items():
        if len(embs) >= 2:
            intra = [
                FaceEngine.cosine_similarity(embs[i], embs[j])
                for i in range(len(embs)) for j in range(i + 1, len(embs))
            ]
            per_identity[name] = {
                "photos": len(embs),
                "mean_intra_similarity": round(float(np.mean(intra)), 4),
                "min_intra_similarity": round(float(np.min(intra)), 4),
                "quality": (
                    "good" if float(np.min(intra)) > 0.55
                    else "mixed" if float(np.min(intra)) > 0.4 else "poor"
                ),
            }
        else:
            per_identity[name] = {
                "photos": len(embs),
                "mean_intra_similarity": None,
                "min_intra_similarity": None,
                "quality": "need-more-photos",
            }

    enough = len(genuine) >= 1 and len(impostor) >= 1

    thresholds: list[dict] = []
    roc: list[dict] = []
    suggested = DEFAULT_THRESHOLD
    eer_thr = DEFAULT_THRESHOLD
    best_j = -2.0
    best_eer_gap = 2.0

    if enough:
        g = np.asarray(genuine)
        m = np.asarray(impostor)
        for thr in np.arange(0.15, 0.751, 0.01):
            t = round(float(thr), 2)
            far = float(np.mean(m >= thr))
            frr = float(np.mean(g < thr))
            youden = (1.0 - frr) - far
            thresholds.append(
                {"threshold": t, "far": round(far, 4), "frr": round(frr, 4),
                 "youden": round(youden, 4)}
            )
            roc.append({"fpr": round(far, 4), "tpr": round(1.0 - frr, 4),
                        "threshold": t})
            if youden > best_j:
                best_j = youden
                suggested = t
            gap = abs(far - frr)
            if gap < best_eer_gap:
                best_eer_gap = gap
                eer_thr = t
        # Perfectly separable gallery? Prefer the max-margin midpoint over
        # the first threshold that happens to achieve Youden J = 1.
        try:
            lo, hi = float(np.max(m)), float(np.min(g))
            if lo < hi:
                suggested = round((lo + hi) / 2, 3)
                eer_thr = suggested
        except ValueError:
            pass

    return {
        "identities": names,
        "counts": {k: len(v) for k, v in snap.items()},
        "per_identity": per_identity,
        "genuine_pairs": len(genuine),
        "impostor_pairs": len(impostor),
        "genuine_scores": [round(float(s), 4) for s in sorted(genuine, reverse=True)[:500]],
        "impostor_scores": [round(float(s), 4) for s in sorted(impostor, reverse=True)[:500]],
        "genuine_mean": round(float(np.mean(genuine)), 4) if genuine else None,
        "impostor_mean": round(float(np.mean(impostor)), 4) if impostor else None,
        "sufficient_data": enough,
        "default_threshold": DEFAULT_THRESHOLD,
        "suggested_threshold": suggested,
        "eer_threshold": eer_thr,
        "thresholds": thresholds,
        "roc": roc,
        "guidance": (
            "Calibration based on your gallery."
            if enough else
            "Need at least 2 identities with 2+ photos each for calibration. "
            f"Keeping default {DEFAULT_THRESHOLD}. Register more photos, then re-run."
        ),
    }
