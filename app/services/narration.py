"""Natural-language narration of inference results.

Every image endpoint returns a `narration` string summarizing what was found,
so UIs can speak it aloud (see the Speak toggle in app/static/index.html)
and API consumers get a human-readable summary for free.
"""
from __future__ import annotations

_IRREGULAR = {
    "person": "people",
    "mouse": "mice",
    "sheep": "sheep",
    "deer": "deer",
    "fish": "fish",
    "child": "children",
    "toothbrush": "toothbrushes",
}


def plural(word: str, n: int) -> str:
    if n == 1:
        return word
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def describe_objects(detections: list[dict], limit: int = 5) -> str:
    """E.g. '2 people and 1 dog'. Empty list -> ''."""
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
        parts.append(f"{sum(n for _, n in ranked[limit:])} more")
    return _join(parts)


def describe_faces(faces: list[dict], limit: int = 5) -> str:
    """E.g. 'Messi and 1 unknown person'. Recognition results carry names;
    plain detections just count human faces."""
    if not faces:
        return ""
    if all("matched" not in f for f in faces):
        n = len(faces)
        return f"{n} human {'face' if n == 1 else 'faces'}"
    named = [str(f["name"]) for f in faces[:limit]
             if f.get("matched") and f.get("name")]
    unknown = sum(1 for f in faces[:limit] if not f.get("matched"))
    parts = list(named)
    if unknown:
        parts.append(f"{unknown} unknown {plural('person', unknown)}")
    extra = len(faces) - limit
    if extra > 0:
        parts.append(f"{extra} more")
    return _join(parts)


def describe_combined(objects: list[dict], faces: list[dict]) -> str:
    o, f = describe_objects(objects), describe_faces(faces)
    if o and f:
        return f"Found {o}. Found {f}."
    if o:
        return f"Found {o}."
    if f:
        return f"Found {f}."
    return "No objects or faces detected."


def describe_video(frames: int, class_counts: dict[str, int],
                   people_counts: dict[str, int]) -> str:
    bits = [f"Analyzed {frames} frames"]
    top_classes = sorted(class_counts.items(), key=lambda kv: -kv[1])[:3]
    if top_classes:
        bits.append("mostly " + _join(
            [f"{plural(n, c)}" for n, c in top_classes]))
    people = [n for n in people_counts if n != "Unknown"]
    if people:
        bits.append("people seen: " + _join(sorted(people)[:5]))
    elif people_counts.get("Unknown"):
        bits.append(f"{people_counts['Unknown']} unknown faces seen")
    return ". ".join(bits) + "."
