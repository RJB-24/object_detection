"""YOLO fine-tuning runner (used by the API training job + CLI).

Usage (CLI):
    python -m training.train --data data/my_dataset/data.yaml --epochs 50
    python -m training.train --demo --epochs 3        # synthetic smoke test
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_training(
    job,  # app.services.jobs.Job (duck-typed to avoid import cycles in CLI)
    data_yaml: str,
    epochs: int = 20,
    imgsz: int = 640,
    base_model: str = "yolov8n.pt",
    name: str = "custom",
    batch: int = 16,
    device: str = "",
    lr0: float = 0.01,
) -> dict:
    from ultralytics import YOLO

    from training.dataset import validate_dataset

    t0 = time.time()
    info = validate_dataset(data_yaml)
    job.log(f"Dataset OK: {info['splits']['train']['images']} train / "
            f"{info['splits']['val']['images']} val, nc={info['nc']}")
    job.log(f"Base model={base_model} epochs={epochs} imgsz={imgsz} "
            f"batch={batch} device={device or 'auto'}")

    model = YOLO(base_model)

    state = {"epoch": 0}

    def _on_epoch_end(trainer) -> None:
        try:
            ep = int(trainer.epoch) + 1
            state["epoch"] = ep
            loss = ""
            try:
                m = trainer.metrics
                loss = f" metrics={ {k: round(float(v), 4) for k, v in dict(m).items() if isinstance(v, (int, float))} }" if isinstance(m, dict) else ""
            except Exception:  # noqa: BLE001
                pass
            job.set_progress(ep / max(epochs, 1), f"Epoch {ep}/{epochs}{loss}")
            job.log(f"Epoch {ep}/{epochs} done")
        except Exception:  # noqa: BLE001
            pass

    try:
        model.add_callback("on_train_epoch_end", _on_epoch_end)
    except Exception:  # noqa: BLE001
        job.log("Note: epoch callbacks unavailable in this ultralytics version")

    results = model.train(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=device or None,
        lr0=float(lr0),
        project=str(ROOT / "models" / "runs"),
        name=name,
        exist_ok=True,
        verbose=False,
        plots=True,
    )
    job.log("Training finished, collecting metrics…")

    metrics: dict = {}
    try:
        # Validate on the val split for final numbers.
        val = model.val(data=str(data_yaml), verbose=False)
        box = val.box
        metrics = {
            "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
        }
    except Exception as exc:  # noqa: BLE001
        job.log(f"Validation skipped: {exc}")

    # Promote best.pt into models/ so the registry + UI pick it up.
    run_dir = ROOT / "models" / "runs" / name
    best_src = run_dir / "weights" / "best.pt"
    promoted = None
    if best_src.exists():
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "custom"
        dest = ROOT / "models" / f"{safe}.pt"
        shutil.copy2(best_src, dest)
        promoted = dest.name
        job.log(f"Saved weights: models/{dest.name} ({dest.stat().st_size/1e6:.1f} MB)")

    elapsed = round(time.time() - t0, 1)
    return {
        "run_dir": str(run_dir),
        "epochs_done": state["epoch"] or epochs,
        "elapsed_s": elapsed,
        "metrics": metrics,
        "promoted_weights": promoted,
        "dataset": info,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune YOLOv8")
    ap.add_argument("--data", default=None, help="Path to data.yaml")
    ap.add_argument("--demo", action="store_true", help="Train on synthetic demo data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--name", default="custom")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from app.services.jobs import Job

    from training.dataset import generate_demo_dataset

    data_yaml = args.data
    if args.demo or not data_yaml:
        print("Generating synthetic demo dataset…")
        info = generate_demo_dataset()
        data_yaml = str(ROOT / "data" / "demo_dataset" / "data.yaml")
        print(f"  train={info['splits']['train']['images']} "
              f"val={info['splits']['val']['images']}")

    job = Job(id="cli", kind="training", label=args.name)
    # Mirror logs to stdout for CLI usage.
    orig_log = job.log

    def _tee(msg: str) -> None:
        print(msg)
        orig_log(msg)

    job.log = _tee  # type: ignore[method-assign]
    out = run_training(job, data_yaml, epochs=args.epochs, imgsz=args.imgsz,
                       base_model=args.base, name=args.name, batch=args.batch)
    print("\nDone:", out)


if __name__ == "__main__":
    main()
