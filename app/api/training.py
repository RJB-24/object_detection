"""YOLO fine-tuning: dataset tools and training jobs."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.services.jobs import get_job_manager
from training.dataset import generate_demo_dataset, validate_dataset
from training.train import run_training

router = APIRouter(tags=["training"])


class TrainingStartBody(BaseModel):
    data_yaml: str = Field(..., description="Path to data.yaml (server-side)")
    epochs: int = Field(default=20, ge=1, le=300)
    imgsz: int = Field(default=640, ge=320, le=1280)
    base_model: str = Field(default="yolov8n.pt")
    name: str = Field(default="custom")
    batch: int = Field(default=16, ge=1, le=128)


class ValidateDatasetBody(BaseModel):
    data_yaml: str


@router.get("/api/training/status")
def training_status():
    try:
        import torch
        from ultralytics import YOLO  # noqa: F401
    except ImportError as exc:
        return {"available": False, "error": str(exc)}
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available() else "cpu"
    )
    return {"available": True, "torch": torch.__version__, "device": device}


@router.post("/api/training/demo-prepare")
def training_demo_prepare():
    """Generate the tiny synthetic dataset used to smoke-test fine-tuning."""
    try:
        info = generate_demo_dataset()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    info["data_yaml"] = str(config.ROOT / "data" / "demo_dataset" / "data.yaml")
    return info


@router.post("/api/training/validate")
def training_validate(body: ValidateDatasetBody):
    try:
        return validate_dataset(body.data_yaml)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/training/start")
def training_start(body: TrainingStartBody):
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

    job = get_job_manager().submit(
        "training",
        f"Fine-tune '{body.name}' ({info['splits']['train']['images']} imgs, "
        f"{body.epochs} epochs)",
        _run,
    )
    return {"job_id": job.id, "status": job.status, "poll": f"/api/jobs/{job.id}"}
