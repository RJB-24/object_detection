"""Bulk-register known faces from a folder structure (CLI alternative to the web UI).

Folder layout:
    data/known_faces/
        Ada/
            1.jpg  2.jpg  ...
        Bob/
            1.jpg  ...

Usage:
    python scripts/register_faces.py
    python scripts/register_faces.py --dir my_photos --clear
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.services.inference import get_inference_service  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(config.KNOWN_FACES_DIR))
    ap.add_argument("--clear", action="store_true", help="Clear DB before registering")
    args = ap.parse_args()

    svc = get_inference_service()
    if args.clear:
        svc.db.clear()
        print("Cleared face database.")

    base = Path(args.dir)
    if not base.exists():
        print(f"Directory not found: {base}")
        sys.exit(1)

    total = 0
    for person_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        imgs = [cv2.imread(str(f)) for f in sorted(person_dir.iterdir())
                if f.suffix.lower() in IMG_EXTS]
        imgs = [i for i in imgs if i is not None]
        if not imgs:
            print(f"  - {person_dir.name}: no images, skipped")
            continue
        try:
            res = svc.register_face(person_dir.name, imgs)
            print(f"  ✓ {person_dir.name}: +{res['embeddings_added']}/{len(imgs)} "
                  f"(total {res['total_for_name']})")
            total += res["embeddings_added"]
        except Exception as exc:  # noqa: BLE001
            print(f"  ✕ {person_dir.name}: {exc}")
    print(f"\nDone. {total} embeddings registered. Identities: {svc.db.names()}")


if __name__ == "__main__":
    main()
