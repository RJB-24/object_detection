"""VisionAI platform: app factory, router wiring, web-app hosting."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__, config
from app.api import faces, history, inference, jobs, models, system, training

DIST_DIR = Path(__file__).parent / "static" / "dist"


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
    _mount_spa(app)
    return app


def _mount_spa(app: FastAPI) -> None:
    """Serve the React dashboard (built via `npm run build` in frontend/).

    Client-side routes fall through to index.html; API/docs paths are left
    to return proper 404s.
    """
    if not DIST_DIR.exists():
        @app.get("/", include_in_schema=False)
        def index():
            return JSONResponse({
                "message": "VisionAI API running. Build the web app with "
                           "`cd frontend && npm install && npm run build`, "
                           "then reload. Interactive API: /docs",
            })
        return

    assets = DIST_DIR / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(DIST_DIR / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str):
        if path.startswith(("api/", "docs", "openapi.json", "redoc")):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        candidate = DIST_DIR / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST_DIR / "index.html")


app = create_app()
