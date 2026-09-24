"""VisionAI platform: app factory, router wiring, web-app hosting."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app import __version__, config
from app.api import faces, history, inference, jobs, models, system, training

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="VisionAI — Visual Intelligence Platform",
        description="YOLOv8 object detection, YuNet + SFace face recognition, "
                    "video analysis, run history and model fine-tuning.",
        version=__version__,
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
    for router in (system.router, inference.router, faces.router,
                   history.router, jobs.router, models.router,
                   training.router):
        app.include_router(router)
    _mount_ui(app)
    return app


def _mount_ui(app: FastAPI) -> None:
    """Serve the single-page UI at / (app/static/index.html)."""
    index = STATIC_DIR / "index.html"

    @app.get("/", include_in_schema=False)
    def root():
        if index.exists():
            return FileResponse(index)
        return JSONResponse({"message": "VisionAI API running. See /docs."})


app = create_app()
