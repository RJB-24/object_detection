"""Download all model weights: YuNet + SFace (ONNX) + YOLOv8 (.pt).

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --yolo yolov8s.pt
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  ✓ already exists: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    print(f"  ↓ downloading {url.split('/')[-1]} -> {dest} ...")

    def _hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, blocks * block_size * 100 / total)
            print(f"\r    {pct:5.1f}%  ({blocks*block_size/1e6:.1f}/{total/1e6:.1f} MB)", end="")

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()
    print(f"  ✓ saved: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default=config.YOLO_MODEL_NAME,
                    help="YOLO weights, e.g. yolov8n.pt / yolov8s.pt / yolov8m.pt")
    ap.add_argument("--skip-yolo", action="store_true")
    args = ap.parse_args()

    print("== VisionAI model download ==")
    print("1/3 YuNet face detection ...")
    download(config.YUNET_URL, Path(config.FACE_DETECTION_MODEL))
    print("2/3 SFace face recognition ...")
    download(config.SFACE_URL, Path(config.FACE_RECOGNITION_MODEL))
    if args.skip_yolo:
        print("3/3 YOLO skipped (--skip-yolo).")
    else:
        print(f"3/3 YOLO weights ({args.yolo}) ...")
        try:
            from app.detectors.object_detector import download_yolo_weights

            p = download_yolo_weights(args.yolo)
            print(f"  ✓ YOLO ready: {p}")
        except ImportError:
            print("  ! ultralytics not installed — run: pip install -r requirements.txt")
            print("    YOLO weights will auto-download on first inference instead.")
    print("\nAll done. Start the server with:  uvicorn app.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
