"""High-level inference orchestrating YOLO + YuNet/SFace + face DB."""
from __future__ import annotations

import threading
import time

import numpy as np

from app import config
from app.detectors.face import get_face_engine
from app.detectors.object_detector import get_object_detector
from app.services.face_db import FaceDatabase
from app.services import history as history_store
from app.services.narration import (
    describe_combined,
    describe_faces_sentence,
    describe_objects_sentence,
)
from app.utils.drawing import draw_detections, draw_faces
from app.utils.image import encode_jpeg, to_base64_jpeg


class InferenceService:
    def __init__(self) -> None:
        self.objects = get_object_detector()
        self.faces = get_face_engine()
        self.db = FaceDatabase()

    # -- objects -------------------------------------------------------
    def detect_objects(
        self,
        image_bgr: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
        return_image: bool = True,
        log_history: bool = True,
        lang: str = "en",
    ) -> dict:
        t0 = time.perf_counter()
        dets = self.objects.predict(image_bgr, conf=conf, iou=iou)
        dt = (time.perf_counter() - t0) * 1000
        out: dict = {
            "count": len(dets),
            "detections": [
                {
                    "bbox": [round(float(v), 1) for v in d["bbox"]],
                    "confidence": round(d["confidence"], 4),
                    "class_id": d["class_id"],
                    "class_name": d["class_name"],
                }
                for d in dets
            ],
            "inference_ms": round(dt, 1),
            "model": self.objects.model_name,
        }
        if return_image:
            out["annotated_image"] = to_base64_jpeg(draw_detections(image_bgr, dets))
        if log_history:
            out["run_id"] = self._log("objects", image_bgr, len(dets), 0, 0,
                                      round(dt, 1), out["detections"][:10])
        out["narration"] = describe_objects_sentence(out["detections"], lang=lang)
        return out

    # -- faces ---------------------------------------------------------
    def detect_faces(self, image_bgr: np.ndarray, return_image: bool = True,
                       lang: str = "en") -> dict:
        t0 = time.perf_counter()
        faces = self.faces.detect(image_bgr)
        dt = (time.perf_counter() - t0) * 1000
        payload = [
            {
                "bbox": [round(float(v), 1) for v in f["bbox"]],
                "confidence": round(float(f["confidence"]), 4),
                "landmarks": [[round(float(a), 1), round(float(b), 1)] for a, b in f["landmarks"]],
            }
            for f in faces
        ]
        out: dict = {"count": len(payload), "faces": payload,
                     "inference_ms": round(dt, 1)}
        if return_image:
            out["annotated_image"] = to_base64_jpeg(draw_faces(image_bgr, faces))
        out["run_id"] = self._log("faces", image_bgr, 0, len(payload), 0,
                                  round(dt, 1), None)
        out["narration"] = describe_faces_sentence(payload, lang=lang)
        return out

    def recognize_faces(
        self,
        image_bgr: np.ndarray,
        threshold: float | None = None,
        return_image: bool = True,
        log_history: bool = False,
        lang: str = "en",
    ) -> dict:
        thr = config.FACE_MATCH_THRESH if threshold is None else float(threshold)
        t0 = time.perf_counter()
        raw_faces = self.faces.detect(image_bgr)
        results: list[dict] = []
        draw_list: list[dict] = []
        for f in raw_faces:
            try:
                emb = self.faces.embedding(image_bgr, f["raw"])
            except Exception:  # noqa: BLE001
                continue
            name, score, matched = self.db.best_match(emb, threshold=thr)
            item = {
                "bbox": [round(float(v), 1) for v in f["bbox"]],
                "confidence": round(float(f["confidence"]), 4),
                "name": name if matched else "Unknown",
                "score": round(float(score), 4),
                "matched": bool(matched),
            }
            results.append(item)
            draw_list.append(
                {"bbox": f["bbox"], "confidence": f["confidence"],
                 "name": item["name"], "score": item["score"],
                 "matched": item["matched"]}
            )
        dt = (time.perf_counter() - t0) * 1000
        out: dict = {
            "count": len(results),
            "faces": results,
            "inference_ms": round(dt, 1),
            "threshold": thr,
            "known_identities": self.db.names(),
        }
        if return_image:
            out["annotated_image"] = to_base64_jpeg(draw_faces(image_bgr, draw_list))
        if log_history:
            matched = sum(1 for r in results if r["matched"])
            out["run_id"] = self._log("faces", image_bgr, 0, len(results),
                                      matched, round(dt, 1), None)
        out["narration"] = describe_faces_sentence(results, lang=lang)
        return out

    # -- combined ------------------------------------------------------
    def analyze(
        self,
        image_bgr: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
        threshold: float | None = None,
        return_image: bool = True,
        log_history: bool = True,
        lang: str = "en",
    ) -> dict:
        # Each branch degrades gracefully so a face-only (or object-only)
        # deployment still returns useful results instead of a 500.
        try:
            objects = self.detect_objects(image_bgr, conf=conf, iou=iou,
                                          return_image=False, log_history=False,
                                          lang=lang)
            obj_err = None
        except Exception as exc:  # noqa: BLE001
            objects = {"count": 0, "detections": [], "inference_ms": 0.0,
                       "model": self.objects.model_name}
            obj_err = str(exc)
            objects["error"] = obj_err
        try:
            faces = self.recognize_faces(image_bgr, threshold=threshold,
                                         return_image=False, log_history=False,
                                         lang=lang)
            face_err = None
        except Exception as exc:  # noqa: BLE001
            faces = {"count": 0, "faces": [], "inference_ms": 0.0,
                     "threshold": config.FACE_MATCH_THRESH,
                     "known_identities": []}
            face_err = str(exc)
            faces["error"] = face_err
        out: dict = {
            "objects": objects, "faces": faces,
            "inference_ms": round(objects["inference_ms"] + faces["inference_ms"], 1),
        }
        if obj_err or face_err:
            out["warnings"] = [e for e in (obj_err, face_err) if e]
        if return_image:
            tmp = draw_detections(
                image_bgr,
                [
                    {"bbox": d["bbox"], "confidence": d["confidence"],
                     "class_id": d["class_id"], "class_name": d["class_name"]}
                    for d in objects["detections"]
                ],
            )
            tmp = draw_faces(
                tmp,
                [
                    {"bbox": f["bbox"], "confidence": f["confidence"],
                     "name": f["name"], "score": f["score"], "matched": f["matched"]}
                    for f in faces["faces"]
                ],
            )
            out["annotated_image"] = to_base64_jpeg(tmp)
        out["narration"] = describe_combined(objects["detections"], faces["faces"],
                                                  lang=lang)
        if log_history:
            matched = sum(1 for f in faces["faces"] if f.get("matched"))
            out["run_id"] = self._log(
                "analyze", image_bgr, objects["count"], faces["count"],
                matched, out["inference_ms"],
                {"top_objects": objects["detections"][:5],
                 "people": [f["name"] for f in faces["faces"]]},
            )
        return out

    # -- history -------------------------------------------------------
    @staticmethod
    def _log(kind: str, image_bgr, objects: int, faces: int, matched: int,
             ms: float, summary) -> int | None:
        try:
            return history_store.log_run(
                kind, objects, faces, matched, ms,
                summary=summary if isinstance(summary, dict)
                else {"detections": summary},
                image_bgr=image_bgr,
            )
        except Exception:  # noqa: BLE001
            return None

    # -- registration --------------------------------------------------
    def register_face(self, name: str, images_bgr: list[np.ndarray]) -> dict:
        name = name.strip()
        if not name:
            raise ValueError("Name must not be empty")
        if not images_bgr:
            raise ValueError("No images provided")
        added = 0
        per_image: list[dict] = []
        for idx, img in enumerate(images_bgr):
            faces = self.faces.detect(img)
            if not faces:
                per_image.append({"image": idx, "faces_found": 0, "added": 0})
                continue
            # Register the most confident face per image (standard practice)
            best = max(faces, key=lambda f: f["confidence"])
            emb = self.faces.embedding(img, best["raw"])
            total = self.db.add_embedding(name, emb)
            added += 1
            per_image.append(
                {"image": idx, "faces_found": len(faces), "added": 1,
                 "confidence": round(float(best["confidence"]), 4),
                 "embeddings_for_name": total}
            )
        # Persist a reference crop for the gallery (best-effort)
        try:
            ref_dir = config.KNOWN_FACES_DIR / name
            ref_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(images_bgr):
                with open(ref_dir / f"{idx}.jpg", "wb") as fh:
                    fh.write(encode_jpeg(img))
        except Exception:  # noqa: BLE001
            pass
        return {
            "name": name,
            "images_received": len(images_bgr),
            "embeddings_added": added,
            "total_for_name": self.db.counts().get(name, 0),
            "details": per_image,
        }


_service: InferenceService | None = None
_service_lock = threading.Lock()


def get_inference_service() -> InferenceService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = InferenceService()
    return _service
