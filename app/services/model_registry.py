"""YOLO model registry: discover local weights, track the active model."""
from __future__ import annotations

from pathlib import Path

from app import config
from app.detectors.object_detector import get_object_detector

# Official checkpoints the server can auto-download on demand.
KNOWN_CHECKPOINTS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]


def _size_mb(p: Path) -> float:
    try:
        return round(p.stat().st_size / 1e6, 1)
    except OSError:
        return 0.0


def list_models() -> dict:
    """All known + custom weights with availability and active marker."""
    active = get_object_detector().model_name
    active_base = Path(active).name
    models_dir = config.ROOT / "models"
    local_pts = {p.name: p for p in models_dir.glob("*.pt")} if models_dir.exists() else {}

    # Custom (fine-tuned) weights live in models/ but aren't official checkpoints.
    custom = sorted(n for n in local_pts if n not in KNOWN_CHECKPOINTS)

    items = []
    for name in KNOWN_CHECKPOINTS + custom:
        p = local_pts.get(name)
        items.append({
            "name": name,
            "available_locally": p is not None,
            "size_mb": _size_mb(p) if p else None,
            "custom": name in custom,
            "active": name in (active, active_base),
        })
    return {"active": active, "models": items}


def switch_model(name: str) -> dict:
    """Switch the active YOLO model (lazy-loads on next inference).

    Accepts an official checkpoint name (auto-downloaded unless already local)
    or a custom weights file under models/.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Model name must not be empty")

    if name in KNOWN_CHECKPOINTS:
        # Prefer the local copy when present (works fully offline);
        # otherwise ultralytics auto-downloads the bare checkpoint name.
        local = config.ROOT / "models" / name
        target = str(local) if local.exists() else name
    else:
        # Custom weights must already exist locally (no silent downloads).
        candidate = (config.ROOT / "models" / Path(name).name).resolve()
        models_dir = (config.ROOT / "models").resolve()
        if models_dir not in candidate.parents:
            raise ValueError("Invalid model path")
        if not candidate.exists():
            raise ValueError(f"Custom weights not found: {candidate.name}")
        target = str(candidate)

    get_object_detector().switch_model(target)
    try:
        config.ACTIVE_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.ACTIVE_MODEL_FILE.write_text(target)
    except OSError:
        pass
    return {"active": target}
