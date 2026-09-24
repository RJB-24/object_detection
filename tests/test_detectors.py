"""Lightweight unit tests that don't need model weights."""
import numpy as np


def test_cosine_similarity_identity():
    from app.detectors.face import FaceEngine

    a = np.random.rand(128).astype(np.float32)
    assert abs(FaceEngine.cosine_similarity(a, a) - 1.0) < 1e-5


def test_cosine_similarity_orthogonal():
    from app.detectors.face import FaceEngine

    a = np.zeros(128, dtype=np.float32); a[0] = 1.0
    b = np.zeros(128, dtype=np.float32); b[1] = 1.0
    assert abs(FaceEngine.cosine_similarity(a, b)) < 1e-6


def test_face_db_roundtrip(tmp_path):
    from app.services.face_db import FaceDatabase

    db = FaceDatabase(path=tmp_path / "db.pkl")
    emb = np.random.rand(128).astype(np.float32)
    assert db.add_embedding("Ada", emb) == 1
    assert db.add_embedding("Ada", emb) == 2
    name, score, matched = db.best_match(emb, threshold=0.3)
    assert name == "Ada" and matched and score > 0.99
    # A far-away vector should not match a strict threshold
    far = -emb
    _, _, matched2 = db.best_match(far, threshold=0.99)
    assert matched2 is False
    # Persistence
    db2 = FaceDatabase(path=tmp_path / "db.pkl")
    assert db2.names() == ["Ada"]
    assert db.remove("Ada") is True
    assert db.names() == []


def test_image_decode_roundtrip():
    import io
    from PIL import Image
    from app.utils.image import decode_image, to_base64_jpeg
    import numpy as np

    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (10, 200, 30)).save(buf, "JPEG")
    bgr = decode_image(buf.getvalue())
    assert bgr.shape == (48, 64, 3)
    assert to_base64_jpeg(bgr).startswith("data:image/jpeg;base64,")
