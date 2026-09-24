"""Face gallery CRUD + threshold calibration."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api import deps
from app.services.calibration import calibrate

router = APIRouter(tags=["faces"])


@router.get("/api/faces")
def list_faces():
    svc = deps.get_service()
    return {"identities": svc.db.names(), "counts": svc.db.counts(),
            "total_embeddings": svc.db.total_embeddings()}


@router.post("/api/faces/register")
async def register_face(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    svc = deps.get_service()
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name must not be empty")
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one face image")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 images per request")
    images = [deps.decode_or_400(await deps.read_upload(f)) for f in files]
    try:
        result = svc.register_face(name, images)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if result["embeddings_added"] == 0:
        raise HTTPException(
            status_code=422,
            detail="No faces found in the uploaded images. Use clear, front-facing photos.",
        )
    return result


@router.delete("/api/faces/{name}")
def delete_face(name: str):
    svc = deps.get_service()
    if svc.db.remove(name):
        return {"deleted": name}
    raise HTTPException(status_code=404, detail=f"Unknown identity: {name}")


@router.delete("/api/faces")
def clear_faces():
    deps.get_service().db.clear()
    return {"cleared": True}


@router.post("/api/faces/calibrate")
def calibrate_threshold():
    """Recommend the optimal face-match threshold from gallery statistics."""
    return calibrate(deps.get_service().db)
