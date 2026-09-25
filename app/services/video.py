"""Video analysis worker: objects and/or faces across sampled frames.

Reads an uploaded video, runs inference on every Nth frame, writes an
annotated MP4 and returns aggregate stats + timeline data for charts.
Executed inside a background Job (see jobs.py) with progress updates.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app import config
from app.detectors.face import FaceEngineError
from app.services.inference import get_inference_service
from app.services.jobs import Job
from app.services.narration import describe_video
from app.utils.drawing import draw_detections, draw_faces
from app.utils.image import to_base64_jpeg

VIDEO_OUT_DIR = config.ROOT / "outputs" / "videos"


def process_video(
    job: Job,
    src_path: str,
    mode: str = "combined",  # objects | faces | combined
    frame_stride: int = 5,
    conf: float | None = None,
    threshold: float | None = None,
    max_frames: int | None = None,
) -> dict:
    svc = get_inference_service()
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file (unsupported codec?)")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    # Cap width for speed; keep aspect.
    scale = min(1.0, 960.0 / max(width, height))
    out_w, out_h = int(width * scale), int(height * scale)

    VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_OUT_DIR / f"{job.id}.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / max(frame_stride, 1),
        (out_w, out_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video writer")

    max_frames = config.MAX_VIDEO_FRAMES if max_frames is None else int(max_frames)
    do_objects = mode in ("objects", "combined")
    do_faces = mode in ("faces", "combined")

    job.log(f"Video: {width}x{height} @ {fps:.1f}fps, {total} frames; "
            f"sampling every {frame_stride}")
    if do_objects:
        try:
            svc.objects.load()
        except RuntimeError as exc:
            writer.release()
            cap.release()
            raise RuntimeError(f"Object model unavailable: {exc}")
    if do_faces:
        try:
            svc.faces.load()
        except FaceEngineError as exc:
            writer.release()
            cap.release()
            raise RuntimeError(f"Face models unavailable: {exc}")

    class_counts: dict[str, int] = {}
    people: dict[str, int] = {}
    timeline: list[dict] = []
    thumb: np.ndarray | None = None
    processed = 0
    read = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        if (read - 1) % max(frame_stride, 1) != 0:
            continue
        if processed >= max_frames:
            job.log(f"Reached frame cap ({max_frames}); stopping early")
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (out_w, out_h))
        t_sec = round(read / fps, 1)

        n_obj = 0
        if do_objects:
            dets = svc.objects.predict(frame, conf=conf)
            n_obj = len(dets)
            for d in dets:
                class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1
            frame = draw_detections(frame, dets)

        n_face = 0
        if do_faces:
            raw_faces = svc.faces.detect(frame)
            draw_list = []
            for f in raw_faces:
                try:
                    emb = svc.faces.embedding(frame, f["raw"])
                except Exception:  # noqa: BLE001
                    continue
                name, score, matched = svc.db.best_match(emb, threshold=threshold)
                label = name if matched else "Unknown"
                people[label] = people.get(label, 0) + 1
                n_face += 1
                draw_list.append({"bbox": f["bbox"], "confidence": f["confidence"],
                                  "name": label, "score": score, "matched": matched})
            frame = draw_faces(frame, draw_list)

        writer.write(frame)
        if thumb is None and (n_obj or n_face):
            thumb = frame.copy()
        processed += 1
        timeline.append({"t": t_sec, "objects": n_obj, "faces": n_face})

        if total:
            job.set_progress(read / total, f"Frame {read}/{total} ({processed} analyzed)")
        elif processed % 10 == 0:
            job.set_progress(0.0, f"{processed} frames analyzed…")

    cap.release()
    writer.release()

    thumb_b64 = to_base64_jpeg(thumb) if thumb is not None else None

    job.log(f"Done: {processed} frames analyzed")
    top_classes = dict(sorted(class_counts.items(), key=lambda kv: -kv[1])[:15])
    top_people = dict(sorted(people.items(), key=lambda kv: -kv[1]))
    return {
        "mode": mode,
        "src_fps": round(fps, 2),
        "src_frames": total,
        "frames_analyzed": processed,
        "frame_stride": frame_stride,
        "output_size": [out_w, out_h],
        "class_counts": top_classes,
        "people_counts": top_people,
        "narration": describe_video(processed, top_classes, top_people),
        "timeline": timeline[:: max(1, len(timeline) // 120)],  # <= ~120 points for charts
        "thumbnail": thumb_b64,
        "download": f"/api/jobs/{job.id}/download",
    }
