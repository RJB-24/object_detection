"""FastAPI app: object detection + face recognition API + web demo UI."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.services.inference import get_inference_service
from app.utils.image import decode_base64, decode_image

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="VisionAI — Object Detection + Face Recognition",
    description=(
        "Deployable image-processing API: YOLOv8 object detection (80 classes) + "
        "YuNet face detection with SFace face recognition against your own gallery."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.CORS_ORIGINS.strip() == "*" else [
        o.strip() for o in config.CORS_ORIGINS.split(",")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _svc():
    try:
        return get_inference_service()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    max_bytes = config.MAX_FILE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"File exceeds {config.MAX_FILE_MB} MB limit"
        )
    return data


def _decode_or_400(data: bytes):
    try:
        return decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file")


# ------------------------------------------------------------------ UI
@app.get("/", include_in_schema=False)
def index():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return JSONResponse({"message": "API running. See /docs for the interactive API."})


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -------------------------------------------------------------- health
@app.get("/api/health")
def health():
    svc = _svc()
    return {
        "status": "ok",
        "version": "1.0.0",
        "models": {
            "yolo": svc.objects.status(),
            "face": svc.faces.status(),
        },
        "face_identities": svc.db.names(),
        "face_embeddings": svc.db.total_embeddings(),
    }


@app.get("/api/models/status")
def model_status():
    return health()


@app.get("/api/labels/objects")
def object_labels():
    svc = _svc()
    return {"labels": svc.objects.class_names}


# ------------------------------------------------------ object detect
@app.post("/api/detect/objects")
async def detect_objects(
    file: UploadFile | None = File(default=None),
    image_base64: str | None = Form(default=None),
    conf: float | None = Query(default=None, ge=0.0, le=1.0),
    iou: float | None = Query(default=None, ge=0.0, le=1.0),
    return_image: bool = Query(default=True),
):
    svc = _svc()
    img = await _image_from_inputs(file, image_base64)
    try:
        return svc.detect_objects(img, conf=conf, iou=iou, return_image=return_image)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# -------------------------------------------------------- face detect
@app.post("/api/detect/faces")
async def detect_faces(
    file: UploadFile | None = File(default=None),
    image_base64: str | None = Form(default=None),
    return_image: bool = Query(default=True),
):
    svc = _svc()
    img = await _image_from_inputs(file, image_base64)
    try:
        return svc.detect_faces(img, return_image=return_image)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------- face recog
@app.post("/api/recognize/faces")
async def recognize_faces(
    file: UploadFile | None = File(default=None),
    image_base64: str | None = Form(default=None),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    return_image: bool = Query(default=True),
):
    svc = _svc()
    img = await _image_from_inputs(file, image_base64)
    try:
        return svc.recognize_faces(img, threshold=threshold, return_image=return_image)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ----------------------------------------------------------- combined
@app.post("/api/analyze")
async def analyze(
    file: UploadFile | None = File(default=None),
    image_base64: str | None = Form(default=None),
    conf: float | None = Query(default=None, ge=0.0, le=1.0),
    iou: float | None = Query(default=None, ge=0.0, le=1.0),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    return_image: bool = Query(default=True),
):
    """One call: generic objects (YOLOv8) + recognized faces (YuNet+SFace)."""
    svc = _svc()
    img = await _image_from_inputs(file, image_base64)
    try:
        return svc.analyze(img, conf=conf, iou=iou, threshold=threshold,
                           return_image=return_image)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


async def _image_from_inputs(file, image_base64):
    if file is not None and getattr(file, "filename", None):
        return _decode_or_400(await _read_upload(file))
    if image_base64:
        try:
            return decode_base64(image_base64)
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="Invalid base64 image")
    raise HTTPException(status_code=400, detail="Provide 'file' or 'image_base64'")


# ------------------------------------------------- face gallery (CRUD)
@app.get("/api/faces")
def list_faces():
    svc = _svc()
    return {"identities": svc.db.names(), "counts": svc.db.counts(),
            "total_embeddings": svc.db.total_embeddings()}


@app.post("/api/faces/register")
async def register_face(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    svc = _svc()
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name must not be empty")
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one face image")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 images per request")
    images = []
    for f in files:
        images.append(_decode_or_400(await _read_upload(f)))
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


@app.delete("/api/faces/{name}")
def delete_face(name: str):
    svc = _svc()
    if svc.db.remove(name):
        return {"deleted": name}
    raise HTTPException(status_code=404, detail=f"Unknown identity: {name}")


@app.delete("/api/faces")
def clear_faces():
    svc = _svc()
    svc.db.clear()
    return {"cleared": True}
