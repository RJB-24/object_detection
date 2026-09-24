"""Run history: logged image analyses with thumbnails."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.services import history as history_store

router = APIRouter(tags=["history"])


@router.get("/api/history")
def get_history(limit: int = Query(default=50, le=200),
                offset: int = Query(default=0, ge=0),
                kind: str | None = Query(default=None)):
    return history_store.list_runs(limit=limit, offset=offset, kind=kind)


@router.get("/api/history/{run_id}")
def get_history_item(run_id: int):
    run = history_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/api/history/{run_id}/thumb")
def get_history_thumb(run_id: int):
    run = history_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    path = history_store.thumb_path(run)
    if not path:
        raise HTTPException(status_code=404, detail="No thumbnail for this run")
    return FileResponse(path, media_type="image/jpeg")


@router.delete("/api/history/{run_id}")
def delete_history_item(run_id: int):
    if history_store.delete_run(run_id):
        return {"deleted": run_id}
    raise HTTPException(status_code=404, detail="Run not found")


@router.delete("/api/history")
def clear_history():
    return {"cleared": history_store.clear_runs()}
