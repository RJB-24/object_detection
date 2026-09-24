"""Dataset utilities for YOLO fine-tuning.

- `generate_demo_dataset()` builds a tiny synthetic 2-class dataset so you
  can prove the training pipeline works without downloading anything.
- `write_data_yaml()` + `validate_dataset()` help with REAL custom datasets
  in standard YOLO format:

      dataset/
        data.yaml            # train/val paths, nc, names
        train/images/*.jpg + train/labels/*.txt
        val/images/*.jpg   + val/labels/*.txt
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["box", "ball"]


def write_data_yaml(
    root: Path, train_dir: str = "train/images", val_dir: str = "val/images",
    names: list[str] | None = None,
) -> Path:
    names = names or CLASSES
    yaml_path = root / "data.yaml"
    lines = [
        f"path: {root.resolve()}",
        f"train: {train_dir}",
        f"val: {val_dir}",
        f"nc: {len(names)}",
        "names:",
    ]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def validate_dataset(data_yaml: str | Path) -> dict:
    """Check a data.yaml + folders; return counts or raise ValueError."""
    import yaml

    yp = Path(data_yaml)
    if not yp.exists():
        raise ValueError(f"data.yaml not found: {yp}")
    cfg = yaml.safe_load(yp.read_text())
    root = Path(cfg.get("path", yp.parent))
    if not root.is_absolute():
        root = (yp.parent / root).resolve()
    out = {"yaml": str(yp), "root": str(root), "nc": cfg.get("nc"),
           "names": cfg.get("names"), "splits": {}}
    for split in ("train", "val"):
        img_dir = root / cfg.get(split, f"{split}/images")
        lbl_dir = img_dir.parent.parent / split / "labels"
        if not img_dir.exists():
            raise ValueError(f"Missing image folder: {img_dir}")
        imgs = sorted(p for p in img_dir.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not imgs:
            raise ValueError(f"No images in {img_dir}")
        missing = [i for i in imgs if not (lbl_dir / f"{i.stem}.txt").exists()]
        out["splits"][split] = {
            "images": len(imgs),
            "labels_dir": str(lbl_dir),
            "missing_labels": len(missing),
        }
    return out


def generate_demo_dataset(root: str | Path = "data/demo_dataset",
                          n_train: int = 24, n_val: int = 8,
                          img_size: int = 320, seed: int = 7) -> dict:
    """Create a tiny synthetic dataset (red boxes vs blue balls)."""
    rng = random.Random(seed)
    root = Path(root)
    for split, n in (("train", n_train), ("val", n_val)):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.full((img_size, img_size, 3), 245, np.uint8)
            labels: list[str] = []
            for _ in range(rng.randint(1, 3)):
                cls = rng.randint(0, 1)
                cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
                w, h = rng.uniform(0.12, 0.35), rng.uniform(0.12, 0.35)
                x1, y1 = int((cx - w / 2) * img_size), int((cy - h / 2) * img_size)
                x2, y2 = int((cx + w / 2) * img_size), int((cy + h / 2) * img_size)
                if cls == 0:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 220), -1)
                else:
                    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2),
                               (x2 - x1) // 2, (220, 120, 30), -1)
                labels.append(f"{cls} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")
            cv2.imwrite(str(root / split / "images" / f"{split}_{i:03d}.jpg"), img)
            (root / split / "labels" / f"{split}_{i:03d}.txt").write_text(
                "\n".join(labels) + "\n")
    yaml_path = write_data_yaml(root)
    info = validate_dataset(yaml_path)
    info["demo"] = True
    return info
