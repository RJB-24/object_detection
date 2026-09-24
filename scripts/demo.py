"""Local CLI demo (no server needed): run detection on an image file.

Usage:
    python scripts/demo.py photo.jpg
    python scripts/demo.py photo.jpg --faces-only --save out.jpg
    python scripts/demo.py photo.jpg --conf 0.4
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="VisionAI local demo")
    ap.add_argument("image", help="Input image path")
    ap.add_argument("--faces-only", action="store_true")
    ap.add_argument("--objects-only", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.363)
    ap.add_argument("--save", default="outputs/annotated.jpg")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read image: {args.image}")
        sys.exit(1)

    from app.services.inference import get_inference_service

    svc = get_inference_service()
    if args.faces_only:
        res = svc.recognize_faces(img, threshold=args.threshold)
        print(f"Faces: {res['count']}")
        for f in res["faces"]:
            print(f"  - {f['name']} (score {f['score']}, det {f['confidence']}) bbox {f['bbox']}")
    elif args.objects_only:
        res = svc.detect_objects(img, conf=args.conf)
        print(f"Objects: {res['count']}  ({res['inference_ms']} ms)")
        for d in res["detections"]:
            print(f"  - {d['class_name']} {d['confidence']:.2f} bbox {d['bbox']}")
    else:
        res = svc.analyze(img, conf=args.conf, threshold=args.threshold)
        print(f"Objects: {res['objects']['count']}  Faces: {res['faces']['count']}  "
              f"({res['inference_ms']} ms)")
        for d in res["objects"]["detections"]:
            print(f"  [obj] {d['class_name']} {d['confidence']:.2f}")
        for f in res["faces"]["faces"]:
            print(f"  [face] {f['name']} score={f['score']} matched={f['matched']}")

    if res.get("annotated_image"):
        raw = res["annotated_image"].split(",", 1)[-1]
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(raw))
        print(f"Annotated image -> {out}")


if __name__ == "__main__":
    main()
