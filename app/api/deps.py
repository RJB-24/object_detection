"""Shared API dependencies and request helpers."""
from __future__ import annotations

from fastapi import HTTPException, UploadFile

from app import config
from app.services.inference import InferenceService, get_inference_service
from app.utils.image import decode_image


def get_service() -> InferenceService:
    try:
        return get_inference_service()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


async def read_upload(file: UploadFile, max_mb: int | None = None) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    limit = (config.MAX_FILE_MB if max_mb is None else max_mb) * 1024 * 1024
    if len(data) > limit:
        raise HTTPException(status_code=413, detail="File exceeds size limit")
    return data


def decode_or_400(data: bytes):
    try:
        return decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file")
