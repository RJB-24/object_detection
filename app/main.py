"""VisionAI platform API + web app.

- Image inference: objects / faces / combined analysis + face gallery CRUD
- Video analysis as background jobs with progress + annotated MP4 download
- Run history (SQLite), threshold calibration, YOLO model registry
- YOLO fine-tuning jobs (custom datasets or synthetic demo data)
- Serves the React dashboard (frontend build output) at /
"""
from __future__ import annotations

from pathlib import Path

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config
from app.services.inference import get_inference_service
from app.utils.image import decode_base64, decode_image

DIST_DIR = Path(__file__).parent / "static" / "dist"
UPLOAD_DIR = config.ROOT / "outputs" / "uploads"

app = FastAPI(
    title="VisionAI — Visual Intelligence Platform",
    description=(
        "Production-style image-processing platform: YOLOv8 object detection, "
        "YuNet + SFace face recognition, video analysis, run history, "
        "threshold calibration and YOLO fine-tuning."
    ),
    version="2.0.0",
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


async def _read_upload(file: UploadFile, max_mb: int | None = None) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    limit = (config.MAX_FILE_MB if max_mb is None else max_mb) * 1024 * 1024
    if len(data) > limit:
        raise HTTPException(status_code=413, detail="File exceeds size limit")
    return data


def _decode_or_400(data: bytes):
    try:
        return decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file")


async def _image_from_inputs(file, image_base64):
    if file is not None and getattr(file, "filename", None):
        return _decode_or_400(await _read_upload(file))
    if image_base64:
        try:
            return decode_base64(image_base64)
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="Invalid base64 image")
    raise HTTPException(status_code=400, detail="Provide 'file' or 'image_base64'")


# ================================================================ health
@app.get("/api/health")
def health():
    svc = _svc()
    return {
        "status": "ok",
        "version": "2.0.0",
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


@app.get("/api/stats")
def platform_stats():
    """Dashboard roll-up: history stats + gallery + active model."""
    from app.services import history as history_store
    from app.services.model_registry import list_models

    svc = _svc()
    h = history_store.stats()
    h["faces"] = {"identities": svc.db.names(),
                  "embeddings": svc.db.total_embeddings()}
    h["active_model"] = list_models()["active"]
    return h


# ======================================================= image inference
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


# =========================================================== face gallery
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


@app.post("/api/faces/calibrate")
def calibrate_threshold():
    """Recommend the optimal face-match threshold from gallery statistics."""
    from app.services.calibration import calibrate

    svc = _svc()
    return calibrate(svc.db)


# ============================================================ run history
@app.get("/api/history")
def get_history(limit: int = Query(default=50, le=200),
                offset: int = Query(default=0, ge=0),
                kind: str | None = Query(default=None)):
    from app.services import history as history_store

    return history_store.list_runs(limit=limit, offset=offset, kind=kind)


@app.get("/api/history/{run_id}")
def get_history_item(run_id: int):
    from app.services import history as history_store

    run = history_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/history/{run_id}/thumb")
def get_history_thumb(run_id: int):
    from app.services import history as history_store

    run = history_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    p = history_store.thumb_path(run)
    if not p:
        raise HTTPException(status_code=404, detail="No thumbnail for this run")
    return FileResponse(p, media_type="image/jpeg")


@app.delete("/api/history/{run_id}")
def delete_history_item(run_id: int):
    from app.services import history as history_store

    if history_store.delete_run(run_id):
        return {"deleted": run_id}
    raise HTTPException(status_code=404, detail="Run not found")


@app.delete("/api/history")
def clear_history():
    from app.services import history as history_store

    return {"cleared": history_store.clear_runs()}


# ======================================================= jobs + video
@app.get("/api/jobs")
def list_jobs(kind: str | None = Query(default=None),
              limit: int = Query(default=50, le=100)):
    from app.services.jobs import get_job_manager

    mgr = get_job_manager()
    return {"jobs": [j.to_dict(include_logs=False) for j in mgr.list(kind, limit)]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    from app.services.jobs import get_job_manager

    job = get_job_manager().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/download")
def download_job_artifact(job_id: str):
    """Download the job artifact (annotated video / trained weights)."""
    from app.services.jobs import get_job_manager

    job = get_job_manager().get(job_id)
    if not job or job.status != "done":
        raise HTTPException(status_code=404, detail="Job not found or not finished")
    if job.kind == "video":
        p = config.ROOT / "outputs" / "videos" / f"{job.id}.mp4"
        if p.exists():
            return FileResponse(p, media_type="video/mp4",
                                filename=f"visionai_{job.id}.mp4")
    elif job.kind == "training" and job.result and job.result.get("promoted_weights"):
        p = config.ROOT / "models" / job.result["promoted_weights"]
        if p.exists():
            return FileResponse(p, media_type="application/octet-stream",
                                filename=job.result["promoted_weights"])
    raise HTTPException(status_code=404, detail="No downloadable artifact for this job")


@app.post("/api/video/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    mode: str = Form(default="combined"),
    frame_stride: int = Form(default=5),
    conf: float | None = Form(default=None),
    threshold: float | None = Form(default=None),
):
    """Submit a video for background analysis. Returns a job to poll."""
    from app.services.jobs import get_job_manager
    from app.services.video import process_video

    if mode not in ("objects", "faces", "combined"):
        raise HTTPException(status_code=400, detail="mode must be objects|faces|combined")
    frame_stride = max(1, min(int(frame_stride), 60))
    data = await _read_upload(file, max_mb=config.MAX_VIDEO_MB)
    if len(data) < 1024:
        raise HTTPException(status_code=400, detail="Video file too small/corrupt")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {ext}")

    mgr = get_job_manager()

    def _run(job) -> dict:
        src = UPLOAD_DIR / f"{job.id}{ext}"
        src.write_bytes(data)
        try:
            return process_video(job, str(src), mode=mode,
                                 frame_stride=frame_stride, conf=conf,
                                 threshold=threshold)
        finally:
            try:
                src.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    job = mgr.submit("video", f"Video analysis ({file.filename})", _run)
    return {"job_id": job.id, "status": job.status,
            "poll": f"/api/jobs/{job.id}"}


# ========================================================= model registry
@app.get("/api/models")
def models_list():
    from app.services.model_registry import list_models

    return list_models()


class SwitchModelBody(BaseModel):
    name: str = Field(..., description="yolov8n.pt / yolov8s.pt / … or custom weights")


@app.post("/api/models/switch")
def models_switch(body: SwitchModelBody):
    from app.services.model_registry import switch_model

    try:
        return switch_model(body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ================================================================ training
@app.get("/api/training/status")
def training_status():
    try:
        import torch  # noqa: F401
        from ultralytics import YOLO  # noqa: F401

        import torch as _t

        device = (
            "cuda" if _t.cuda.is_available()
            else "mps" if getattr(_t.backends, "mps", None)
            and _t.backends.mps.is_available() else "cpu"
        )
        return {"available": True, "torch": _t.__version__, "device": device}
    except ImportError as exc:
        return {"available": False, "error": str(exc)}


class TrainingStartBody(BaseModel):
    data_yaml: str = Field(..., description="Path to data.yaml (server-side)")
    epochs: int = Field(default=20, ge=1, le=300)
    imgsz: int = Field(default=640, ge=320, le=1280)
    base_model: str = Field(default="yolov8n.pt")
    name: str = Field(default="custom")
    batch: int = Field(default=16, ge=1, le=128)


@app.post("/api/training/demo-prepare")
def training_demo_prepare():
    """Generate the tiny synthetic dataset used to smoke-test fine-tuning."""
    from training.dataset import generate_demo_dataset

    try:
        info = generate_demo_dataset()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    info["data_yaml"] = str(config.ROOT / "data" / "demo_dataset" / "data.yaml")
    return info


@app.post("/api/training/validate")
def training_validate(body: SwitchModelBody):
    """Validate a dataset's data.yaml (reuses {name} as the yaml path)."""
    from training.dataset import validate_dataset

    try:
        return validate_dataset(body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/training/start")
def training_start(body: TrainingStartBody, background_tasks: BackgroundTasks):
    from app.services.jobs import get_job_manager
    from training.dataset import validate_dataset
    from training.train import run_training

    try:
        info = validate_dataset(body.data_yaml)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Resolve base weights: allow registry names + local custom files.
    base = body.base_model.strip()
    if "/" not in base and (config.ROOT / "models" / base).exists():
        base = str(config.ROOT / "models" / base)

    def _run(job) -> dict:
        return run_training(
            job, body.data_yaml, epochs=body.epochs, imgsz=body.imgsz,
            base_model=base, name=body.name, batch=body.batch,
        )

    mgr = get_job_manager()
    job = mgr.submit("training", f"Fine-tune '{body.name}' "
                                 f"({info['splits']['train']['images']} imgs, "
                                 f"{body.epochs} epochs)", _run)
    return {"job_id": job.id, "status": job.status,
            "poll": f"/api/jobs/{job.id}"}


# ================================================================ web app
if DIST_DIR.exists():
    assets = DIST_DIR / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(DIST_DIR / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str):
        # API + docs routes are matched before this; anything else that is
        # not a real file falls through to the React app (client routing).
        if path.startswith(("api/", "docs", "openapi.json", "redoc")):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        candidate = DIST_DIR / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST_DIR / "index.html")
else:

    @app.get("/", include_in_schema=False)
    def index():
        return JSONResponse({
            "message": "VisionAI API running. Build the web app with "
                       "`cd frontend && npm install && npm run build`, "
                       "then reload. Interactive API: /docs",
        })
