"""Persistent run history (SQLite, stdlib only).

Every image analysis can be logged here with a thumbnail + summary so the
dashboard can show recent runs, totals and latency trends.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from app import config

THUMB_DIR = config.ROOT / "outputs" / "thumbs"
HISTORY_DB = config.ROOT / "data" / "history.db"

_lock = threading.Lock()
_init_done = False


def _connect() -> sqlite3.Connection:
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(HISTORY_DB), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    global _init_done
    with _lock:
        if _init_done:
            return
        con = _connect()
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,          -- objects | faces | analyze
                source TEXT NOT NULL DEFAULT 'upload',
                objects INTEGER NOT NULL DEFAULT 0,
                faces INTEGER NOT NULL DEFAULT 0,
                matched INTEGER NOT NULL DEFAULT 0,
                ms REAL NOT NULL DEFAULT 0,
                thumb TEXT,                  -- relative path under outputs/
                summary TEXT                 -- compact JSON
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts)")
        con.commit()
        con.close()
        _init_done = True


def _save_thumb(run_id: int, image_bgr: np.ndarray | None) -> str | None:
    if image_bgr is None:
        return None
    try:
        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        h, w = image_bgr.shape[:2]
        scale = min(1.0, 320.0 / max(h, w))
        small = (
            cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
            if scale < 1.0
            else image_bgr
        )
        rel = f"thumbs/{run_id}.jpg"
        cv2.imwrite(str(THUMB_DIR / f"{run_id}.jpg"), small,
                    [cv2.IMWRITE_JPEG_QUALITY, 75])
        return rel
    except Exception:  # noqa: BLE001
        return None


def log_run(
    kind: str,
    objects: int,
    faces: int,
    matched: int,
    ms: float,
    summary: dict | None = None,
    image_bgr: np.ndarray | None = None,
    source: str = "upload",
) -> int:
    init_db()
    with _lock:
        con = _connect()
        cur = con.execute(
            "INSERT INTO runs (ts, kind, source, objects, faces, matched, ms, summary)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), kind, source, int(objects), int(faces),
             int(matched), float(ms), json.dumps(summary or {})),
        )
        run_id = int(cur.lastrowid)
        con.commit()
        con.close()
    thumb = _save_thumb(run_id, image_bgr)
    if thumb:
        with _lock:
            con = _connect()
            con.execute("UPDATE runs SET thumb = ? WHERE id = ?", (thumb, run_id))
            con.commit()
            con.close()
    return run_id


def list_runs(limit: int = 50, offset: int = 0, kind: str | None = None) -> dict:
    init_db()
    with _lock:
        con = _connect()
        if kind:
            rows = con.execute(
                "SELECT * FROM runs WHERE kind = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (kind, limit, offset),
            ).fetchall()
            total = con.execute(
                "SELECT COUNT(*) FROM runs WHERE kind = ?", (kind,)
            ).fetchone()[0]
        else:
            rows = con.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        con.close()
    return {"total": total, "runs": [dict(r) for r in rows]}


def get_run(run_id: int) -> dict | None:
    init_db()
    with _lock:
        con = _connect()
        row = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        con.close()
    return dict(row) if row else None


def thumb_path(run: dict) -> Path | None:
    if not run.get("thumb"):
        return None
    p = THUMB_DIR / Path(run["thumb"]).name
    return p if p.exists() else None


def delete_run(run_id: int) -> bool:
    run = get_run(run_id)
    if not run:
        return False
    t = thumb_path(run)
    if t:
        try:
            t.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    with _lock:
        con = _connect()
        con.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        con.commit()
        con.close()
    return True


def clear_runs() -> int:
    with _lock:
        con = _connect()
        n = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        con.execute("DELETE FROM runs")
        con.commit()
        con.close()
    try:
        for f in THUMB_DIR.glob("*.jpg"):
            f.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
    return int(n)


def stats() -> dict:
    """Aggregate stats for the dashboard."""
    init_db()
    with _lock:
        con = _connect()
        row = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(objects),0), COALESCE(SUM(faces),0),"
            " COALESCE(SUM(matched),0), COALESCE(AVG(ms),0) FROM runs"
        ).fetchone()
        by_kind = con.execute(
            "SELECT kind, COUNT(*) FROM runs GROUP BY kind"
        ).fetchall()
        # latency trend: last 30 runs, oldest -> newest
        trend = con.execute(
            "SELECT id, ms, objects, faces FROM runs ORDER BY id DESC LIMIT 30"
        ).fetchall()
        con.close()
    return {
        "total_runs": row[0],
        "total_objects": row[1],
        "total_faces": row[2],
        "total_matched": row[3],
        "avg_ms": round(row[4] or 0, 1),
        "by_kind": {k: c for k, c in by_kind},
        "trend": [
            {"id": r[0], "ms": round(r[1] or 0, 1),
             "objects": r[2], "faces": r[3]}
            for r in reversed(trend)
        ],
    }
