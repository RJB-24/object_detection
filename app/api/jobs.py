"""Background jobs: video analysis, status polling, artifact download."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app import config
from app.api import deps
from app.services.jobs import get_job_manager
from app.services.video import process_video

router = APIRouter(tags=["jobs"])

UPLOAD_DIR = config.ROOT / "outputs" / "uploads"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@router.get("/api/jobs")
def list_jobs(kind: str | None = Query(default=None),
              limit: int = Query(default=50, le=100)):
    mgr = get_job_manager()
    return {"jobs": [j.to_dict(include_logs=False) for j in mgr.list(kind, limit)]}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = get_job_manager().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.get("/api/jobs/{job_id}/download")
def download_job_artifact(job_id: str):
    """Download the job artifact (annotated video / trained weights)."""
    job = get_job_manager().get(job_id)
    if not job or job.status != "done":
        raise HTTPException(status_code=404, detail="Job not found or not finished")
    if job.kind == "video":
        path = config.ROOT / "outputs" / "videos" / f"{job.id}.mp4"
        if path.exists():
            return FileResponse(path, media_type="video/mp4",
                                filename=f"visionai_{job.id}.mp4")
    elif job.kind == "training" and job.result and job.result.get("promoted_weights"):
        path = config.ROOT / "models" / job.result["promoted_weights"]
        if path.exists():
            return FileResponse(path, media_type="application/octet-stream",
                                filename=job.result["promoted_weights"])
    raise HTTPException(status_code=404, detail="No downloadable artifact for this job")


@router.post("/api/video/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    mode: str = Form(default="combined"),
    frame_stride: int = Form(default=5),
    conf: float | None = Form(default=None),
    threshold: float | None = Form(default=None),
):
    """Submit a video for background analysis. Returns a job to poll."""
    if mode not in ("objects", "faces", "combined"):
        raise HTTPException(status_code=400, detail="mode must be objects|faces|combined")
    frame_stride = max(1, min(int(frame_stride), 60))
    data = await deps.read_upload(file, max_mb=config.MAX_VIDEO_MB)
    if len(data) < 1024:
        raise HTTPException(status_code=400, detail="Video file too small/corrupt")

    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if ext not in VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {ext}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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

    job = get_job_manager().submit("video", f"Video analysis ({file.filename})", _run)
    return {"job_id": job.id, "status": job.status, "poll": f"/api/jobs/{job.id}"}
