"""Natural-language narration of inference results.

Every image endpoint returns a `narration` string summarizing what was found,
so UIs can speak it aloud (see the Speak toggle in app/static/index.html)
and API consumers get a human-readable summary for free.

`lang` selects the sentence frame: "en" (default), "hi" (Hindi), "ta"
(Tamil). Object class names stay in English in every language (code-mixed,
e.g. "2 people और 1 dog मिले।") — this matches how the terms are spoken
and keeps TTS pronunciation reliable. Unknown langs fall back to "en".
"""
from __future__ import annotations

LANGS = ("en", "hi", "ta")

_IRREGULAR = {
    "person": "people",
    "mouse": "mice",
    "sheep": "sheep",
    "deer": "deer",
    "fish": "fish",
    "child": "children",
    "toothbrush": "toothbrushes",
}

_AND = {"en": "and", "hi": "और", "ta": "மற்றும்"}

_NO_OBJECTS = {
    "en": "No objects detected.",
    "hi": "कोई वस्तु नहीं मिली।",
    "ta": "எந்தப் பொருளும் கண்டறியப்படவில்லை.",
}
_NO_FACES = {
    "en": "No faces detected.",
    "hi": "कोई चेहरा नहीं मिला।",
    "ta": "முகங்கள் எதுவும் கண்டறியப்படவில்லை.",
}
_NO_EITHER = {
    "en": "No objects or faces detected.",
    "hi": "कोई वस्तु या चेहरा नहीं मिला।",
    "ta": "பொருட்களோ முகங்களோ கண்டறியப்படவில்லை.",
}


def normalize_lang(lang: str | None) -> str:
    lang = (lang or "en").lower()
    return lang if lang in LANGS else "en"


def plural(word: str, n: int) -> str:
    if n == 1:
        return word
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def _join(parts: list[str], lang: str) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    and_word = _AND[lang]
    if len(parts) == 2:
        return f"{parts[0]} {and_word} {parts[1]}"
    return ", ".join(parts[:-1]) + f" {and_word} {parts[-1]}"


def _more(n: int, lang: str) -> str:
    if lang == "hi":
        return f"{n} और"
    if lang == "ta":
        return f"மேலும் {n}"
    return f"{n} more"


def _unknown_person(n: int, lang: str) -> str:
    if lang == "hi":
        return f"{n} अज्ञात व्यक्ति"
    if lang == "ta":
        return f"{n} அறியப்படாத {'நபர்' if n == 1 else 'நபர்கள்'}"
    return f"{n} unknown {plural('person', n)}"


def _human_face(n: int, lang: str) -> str:
    if lang == "hi":
        return f"{n} मानव {'चेहरा' if n == 1 else 'चेहरे'}"
    if lang == "ta":
        return f"{n} மனித {'முகம்' if n == 1 else 'முகங்கள்'}"
    return f"{n} human {'face' if n == 1 else 'faces'}"


def _found(phrase: str, n: int, lang: str) -> str:
    if lang == "hi":
        return f"{phrase} {'मिला' if n == 1 else 'मिले'}।"
    if lang == "ta":
        return f"{phrase} கண்டறியப்பட்டது."
    return f"Found {phrase}."


def describe_objects(detections: list[dict], limit: int = 5,
                     lang: str = "en") -> str:
    """E.g. '2 people and 1 dog'. Empty list -> ''."""
    lang = normalize_lang(lang)
    if not detections:
        return ""
    counts: dict[str, int] = {}
    best: dict[str, float] = {}
    for d in detections:
        name = str(d.get("class_name", "object"))
        counts[name] = counts.get(name, 0) + 1
        best[name] = max(best.get(name, 0.0), float(d.get("confidence", 0.0)))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -best[kv[0]]))
    parts = [f"{n} {plural(name, n)}" for name, n in ranked[:limit]]
    if len(ranked) > limit:
        parts.append(_more(sum(n for _, n in ranked[limit:]), lang))
    return _join(parts, lang)


def describe_faces(faces: list[dict], limit: int = 5, lang: str = "en") -> str:
    """E.g. 'Messi and 1 unknown person'. Recognition results carry names;
    plain detections just count human faces."""
    lang = normalize_lang(lang)
    if not faces:
        return ""
    if all("matched" not in f for f in faces):
        return _human_face(len(faces), lang)
    named = [str(f["name"]) for f in faces[:limit]
             if f.get("matched") and f.get("name")]
    unknown = sum(1 for f in faces[:limit] if not f.get("matched"))
    parts = list(named)
    if unknown:
        parts.append(_unknown_person(unknown, lang))
    extra = len(faces) - limit
    if extra > 0:
        parts.append(_more(extra, lang))
    return _join(parts, lang)


def describe_objects_sentence(detections: list[dict], lang: str = "en") -> str:
    lang = normalize_lang(lang)
    phrase = describe_objects(detections, lang=lang)
    return _found(phrase, len(detections), lang) if phrase else _NO_OBJECTS[lang]


def describe_faces_sentence(faces: list[dict], lang: str = "en") -> str:
    lang = normalize_lang(lang)
    phrase = describe_faces(faces, lang=lang)
    return _found(phrase, len(faces), lang) if phrase else _NO_FACES[lang]


def describe_combined(objects: list[dict], faces: list[dict],
                      lang: str = "en") -> str:
    lang = normalize_lang(lang)
    o = describe_objects(objects, lang=lang)
    f = describe_faces(faces, lang=lang)
    if o and f:
        return (f"{_found(o, len(objects), lang)} "
                f"{_found(f, len(faces), lang)}")
    if o:
        return _found(o, len(objects), lang)
    if f:
        return _found(f, len(faces), lang)
    return _NO_EITHER[lang]


def describe_video(frames: int, class_counts: dict[str, int],
                   people_counts: dict[str, int], lang: str = "en") -> str:
    lang = normalize_lang(lang)
    if lang == "hi":
        bits = [f"{frames} फ्रेमों का विश्लेषण"]
    elif lang == "ta":
        bits = [f"{frames} பிரேம்கள் பகுப்பாய்வு"]
    else:
        bits = [f"Analyzed {frames} frames"]
    top_classes = sorted(class_counts.items(), key=lambda kv: -kv[1])[:3]
    if top_classes:
        items = _join([plural(n, c) for n, c in top_classes], lang)
        bits.append({"en": f"mostly {items}",
                     "hi": f"मुख्य रूप से: {items}",
                     "ta": f"முக்கியமாக: {items}"}[lang])
    people = [n for n in people_counts if n != "Unknown"]
    if people:
        names = _join(sorted(people)[:5], lang)
        bits.append({"en": f"people seen: {names}",
                     "hi": f"दिखे लोग: {names}",
                     "ta": f"காணப்பட்டவர்கள்: {names}"}[lang])
    elif people_counts.get("Unknown"):
        n = people_counts["Unknown"]
        bits.append({"en": f"{n} unknown faces seen",
                     "hi": f"{n} अज्ञात चेहरे दिखे",
                     "ta": f"{n} அறியப்படாத முகங்கள்"}[lang])
    sep, end = ("। ", "।") if lang == "hi" else (". ", ".")
    return sep.join(bits) + end
