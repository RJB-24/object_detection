"""Platform status endpoints: health, stats, labels."""
from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api import deps
from app.services import history as history_store
from app.services.model_registry import list_models

router = APIRouter(tags=["system"])


@router.get("/api/health")
def health():
    svc = deps.get_service()
    return {
        "status": "ok",
        "version": __version__,
        "models": {
            "yolo": svc.objects.status(),
            "face": svc.faces.status(),
        },
        "face_identities": svc.db.names(),
        "face_embeddings": svc.db.total_embeddings(),
    }


@router.get("/api/labels/objects")
def object_labels():
    return {"labels": deps.get_service().objects.class_names}


@router.get("/api/stats")
def platform_stats():
    """Dashboard roll-up: history stats + gallery + active model."""
    svc = deps.get_service()
    stats = history_store.stats()
    stats["faces"] = {"identities": svc.db.names(),
                      "embeddings": svc.db.total_embeddings()}
    stats["active_model"] = list_models()["active"]
    return stats
