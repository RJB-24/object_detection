"""Tests for calibration, history store and model registry."""
import numpy as np


def _seeded_db(path):
    from app.services.face_db import FaceDatabase

    db = FaceDatabase(path=path)
    db.clear()
    rng = np.random.RandomState(11)
    # Well-separated synthetic identities (unit-sphere-ish for realism)
    a = rng.rand(128).astype(np.float32)
    b = rng.rand(128).astype(np.float32)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    for _ in range(3):
        na = a + rng.rand(128).astype(np.float32) * 0.02
        nb = b + rng.rand(128).astype(np.float32) * 0.02
        db.add_embedding("Ada", na / np.linalg.norm(na))
        db.add_embedding("Bob", nb / np.linalg.norm(nb))
    return db


def test_calibrate_separated_gallery(tmp_path):
    from app.services.calibration import calibrate

    out = calibrate(_seeded_db(tmp_path / "db.pkl"))
    assert out["sufficient_data"] is True
    assert out["genuine_pairs"] == 6  # C(3,2) x 2 identities
    assert out["impostor_pairs"] == 9  # 3 x 3
    assert out["genuine_mean"] > out["impostor_mean"]
    # Max-margin suggestion sits strictly between the two distributions.
    assert max(out["impostor_scores"]) < out["suggested_threshold"] < min(out["genuine_scores"])
    assert len(out["roc"]) > 40
    assert out["per_identity"]["Ada"]["quality"] == "good"


def test_calibrate_insufficient_data(tmp_path):
    from app.services.calibration import calibrate
    from app.services.face_db import FaceDatabase

    db = FaceDatabase(path=tmp_path / "db.pkl")
    db.clear()
    db.add_embedding("Solo", np.random.rand(128).astype(np.float32))
    out = calibrate(db)
    assert out["sufficient_data"] is False
    assert out["suggested_threshold"] == out["default_threshold"]


def test_history_crud(tmp_path, monkeypatch):
    from app.services import history as H

    monkeypatch.setattr(H, "HISTORY_DB", tmp_path / "h.db")
    monkeypatch.setattr(H, "THUMB_DIR", tmp_path / "thumbs")
    rid = H.log_run("analyze", 2, 1, 1, 99.0, summary={"k": "v"},
                    image_bgr=np.zeros((32, 32, 3), np.uint8))
    assert rid == 1
    assert H.list_runs()["total"] == 1
    assert H.get_run(1)["kind"] == "analyze"
    assert H.thumb_path(H.get_run(1)) is not None
    s = H.stats()
    assert s["total_runs"] == 1 and s["total_objects"] == 2
    assert H.delete_run(1) is True
    assert H.list_runs()["total"] == 0


def test_registry_lists_known_models():
    from app.services.model_registry import list_models

    out = list_models()
    names = [m["name"] for m in out["models"]]
    assert "yolov8n.pt" in names and "yolov8s.pt" in names
    assert out["active"]


def test_job_manager_lifecycle():
    from app.services.jobs import JobManager

    mgr = JobManager()

    def _fn(job):
        job.log("hello")
        job.set_progress(0.5, "half")
        return {"ok": True}

    job = mgr.submit("video", "test job", _fn)
    import time

    for _ in range(100):
        if job.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert job.status == "done"
    assert job.result == {"ok": True}
    assert any("hello" in line for line in job.logs)
    assert mgr.get(job.id) is job
