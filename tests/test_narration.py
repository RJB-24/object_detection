"""Tests for result narration (pure functions, no models needed)."""
from app.services.narration import (
    describe_combined,
    describe_faces,
    describe_objects,
    describe_video,
    plural,
)


def _det(name, conf=0.9):
    return {"class_name": name, "confidence": conf}


def _face(name=None, matched=False, score=0.0):
    return {"name": name, "matched": matched, "score": score}


def test_plural():
    assert plural("person", 1) == "person"
    assert plural("person", 2) == "people"
    assert plural("dog", 2) == "dogs"
    assert plural("bus", 2) == "buses"
    assert plural("mouse", 3) == "mice"


def test_describe_objects():
    dets = [_det("person"), _det("person"), _det("dog")]
    assert describe_objects(dets) == "2 people and 1 dog"
    assert describe_objects([_det("cat")]) == "1 cat"
    assert describe_objects([]) == ""


def test_describe_objects_ranks_by_count():
    dets = [_det("dog", 0.99), _det("person", 0.5), _det("person", 0.5)]
    assert describe_objects(dets) == "2 people and 1 dog"


def test_describe_faces_recognized():
    faces = [_face("Messi", True, 0.9), _face("Unknown", False, 0.1)]
    assert describe_faces(faces) == "Messi and 1 unknown person"


def test_describe_faces_plain_detection():
    faces = [{"bbox": [0, 0, 1, 1], "confidence": 0.9}]
    assert describe_faces(faces) == "1 human face"
    assert describe_faces(faces * 3) == "3 human faces"


def test_describe_combined():
    out = describe_combined([_det("dog")], [_face("Messi", True, 0.9)])
    assert out == "Found 1 dog. Found Messi."
    assert describe_combined([], []) == "No objects or faces detected."
    assert describe_combined([_det("car")], []) == "Found 1 car."
    assert describe_combined([], [_face("Unknown", False, 0.1)]) == \
        "Found 1 unknown person."


def test_describe_video():
    out = describe_video(80, {"person": 40, "car": 10}, {"Messi": 30})
    assert "Analyzed 80 frames" in out
    assert "people" in out and "Messi" in out


def test_hindi_combined():
    out = describe_combined([_det("dog")], [_face("Messi", True, 0.9)], lang="hi")
    assert out == "1 dog मिला। Messi मिला।"


def test_hindi_plurals_and_unknown():
    faces = [_face("Messi", True, 0.9), _face("Unknown", False, 0.1)]
    out = describe_combined([_det("person"), _det("person"), _det("dog")],
                            faces, lang="hi")
    assert out == "2 people और 1 dog मिले। Messi और 1 अज्ञात व्यक्ति मिले।"


def test_hindi_empty():
    assert describe_combined([], [], lang="hi") == "कोई वस्तु या चेहरा नहीं मिला।"


def test_hindi_plain_faces():
    assert describe_faces([{"bbox": [0, 0, 1, 1]}], lang="hi") == "1 मानव चेहरा"


def test_tamil_combined():
    out = describe_combined([_det("dog")], [_face("Messi", True, 0.9)], lang="ta")
    assert out == "1 dog கண்டறியப்பட்டது. Messi கண்டறியப்பட்டது."


def test_tamil_unknown_and_empty():
    out = describe_combined([], [_face("Unknown", False, 0.1)], lang="ta")
    assert out == "1 அறியப்படாத நபர் கண்டறியப்பட்டது."
    assert describe_combined([], [], lang="ta") == \
        "பொருட்களோ முகங்களோ கண்டறியப்படவில்லை."


def test_unknown_lang_falls_back_to_english():
    assert describe_combined([_det("dog")], [], lang="xx") == "Found 1 dog."
    assert describe_video(10, {}, {}, lang="xx") == "Analyzed 10 frames."


def test_video_multilingual():
    hi = describe_video(80, {"person": 40}, {"Messi": 30}, lang="hi")
    assert hi.startswith("80 फ्रेमों का विश्लेषण")
    assert "Messi" in hi
    ta = describe_video(80, {"person": 40}, {}, lang="ta")
    assert ta.startswith("80 பிரேம்கள் பகுப்பாய்வு")
