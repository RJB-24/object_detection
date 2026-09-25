"""Image inference endpoints: objects, faces, combined analysis."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api import deps

router = APIRouter(tags=["inference"])


@router.post("/api/detect/objects")
async def detect_objects(
    file: UploadFile = File(...),
    conf: float | None = Query(default=None, ge=0.0, le=1.0),
    iou: float | None = Query(default=None, ge=0.0, le=1.0),
    return_image: bool = Query(default=True),
    lang: str = Query(default="en"),
):
    svc = deps.get_service()
    img = deps.decode_or_400(await deps.read_upload(file))
    try:
        return svc.detect_objects(img, conf=conf, iou=iou, return_image=return_image,
                                  lang=lang)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/detect/faces")
async def detect_faces(
    file: UploadFile = File(...),
    return_image: bool = Query(default=True),
    lang: str = Query(default="en"),
):
    svc = deps.get_service()
    img = deps.decode_or_400(await deps.read_upload(file))
    try:
        return svc.detect_faces(img, return_image=return_image, lang=lang)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/recognize/faces")
async def recognize_faces(
    file: UploadFile = File(...),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    return_image: bool = Query(default=True),
    lang: str = Query(default="en"),
):
    svc = deps.get_service()
    img = deps.decode_or_400(await deps.read_upload(file))
    try:
        return svc.recognize_faces(img, threshold=threshold, return_image=return_image,
                                   lang=lang)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    conf: float | None = Query(default=None, ge=0.0, le=1.0),
    iou: float | None = Query(default=None, ge=0.0, le=1.0),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    return_image: bool = Query(default=True),
    lang: str = Query(default="en"),
):
    """One call: generic objects (YOLOv8) + recognized faces (YuNet+SFace)."""
    svc = deps.get_service()
    img = deps.decode_or_400(await deps.read_upload(file))
    try:
        return svc.analyze(img, conf=conf, iou=iou, threshold=threshold,
                           return_image=return_image, lang=lang)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
