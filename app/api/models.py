"""YOLO model registry: list weights, switch the active model."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.model_registry import list_models, switch_model

router = APIRouter(tags=["models"])


class SwitchModelBody(BaseModel):
    name: str


@router.get("/api/models")
def models_list():
    return list_models()


@router.post("/api/models/switch")
def models_switch(body: SwitchModelBody):
    try:
        return switch_model(body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
