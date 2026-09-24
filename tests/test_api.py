"""API smoke tests (mock heavy inference so CI stays light)."""
import io

from fastapi.testclient import TestClient
from PIL import Image


def _fake_jpeg(w=320, h=240, color=(120, 140, 200)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG")
    return buf.getvalue()


def _client_with_mocks(monkeypatch):
    import app.main as main_mod
    from app.api import deps as deps_mod

    class FakeYOLO:
        model_name = "yolov8n.pt"

        def status(self):
            return {"model": "yolov8n.pt", "loaded": True, "error": None}

        class_names = {0: "person"}

    class FakeFace:
        def status(self):
            return {"loaded": True, "error": None,
                    "det_exists": True, "rec_exists": True}

    class FakeDB:
        def names(self): return ["Ada"]
        def counts(self): return {"Ada": 2}
        def total_embeddings(self): return 2
        def remove(self, name): return name == "Ada"
        def clear(self): return None

    class FakeSvc:
        objects = FakeYOLO()
        faces = FakeFace()
        db = FakeDB()

        def detect_objects(self, *a, **k):
            return {"count": 1, "detections": [
                {"bbox": [10, 10, 50, 60], "confidence": 0.9,
                 "class_id": 0, "class_name": "person"}],
                "inference_ms": 5.0, "model": "yolov8n.pt"}

        def detect_faces(self, *a, **k):
            return {"count": 0, "faces": [], "inference_ms": 3.0}

        def recognize_faces(self, *a, **k):
            return {"count": 0, "faces": [], "inference_ms": 3.0,
                    "threshold": 0.363, "known_identities": ["Ada"]}

        def analyze(self, *a, **k):
            return {"objects": self.detect_objects(),
                    "faces": self.recognize_faces(), "inference_ms": 8.0}

        def register_face(self, name, images):
            return {"name": name, "images_received": len(images),
                    "embeddings_added": len(images), "total_for_name": 2, "details": []}

    monkeypatch.setattr(deps_mod, "get_service", lambda: FakeSvc())
    return TestClient(main_mod.app)


def test_health(monkeypatch):
    c = _client_with_mocks(monkeypatch)
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_detect_objects(monkeypatch):
    c = _client_with_mocks(monkeypatch)
    r = c.post("/api/detect/objects", files={"file": ("t.jpg", _fake_jpeg(), "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_analyze(monkeypatch):
    c = _client_with_mocks(monkeypatch)
    r = c.post("/api/analyze", files={"file": ("t.jpg", _fake_jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert "objects" in body and "faces" in body


def test_register_and_list(monkeypatch):
    c = _client_with_mocks(monkeypatch)
    r = c.post("/api/faces/register",
               data={"name": "Ada"},
               files=[("files", ("a.jpg", _fake_jpeg(), "image/jpeg"))])
    assert r.status_code == 200
    assert r.json()["name"] == "Ada"
    r = c.get("/api/faces")
    assert "Ada" in r.json()["identities"]


def test_empty_upload_rejected(monkeypatch):
    c = _client_with_mocks(monkeypatch)
    r = c.post("/api/detect/objects", files={"file": ("e.jpg", b"", "image/jpeg")})
    assert r.status_code in (400, 413)


def test_missing_file_rejected(monkeypatch):
    c = _client_with_mocks(monkeypatch)
    r = c.post("/api/detect/objects", files={})
    assert r.status_code == 422
