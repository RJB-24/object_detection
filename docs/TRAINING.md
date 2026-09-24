# Fine-tuning playbook — train YOLOv8 on your own data 🧪

This project supports full transfer-learning fine-tuning of the object detector,
from the **Fine-tune page** in the dashboard or from the terminal. Trained weights
are promoted into `models/`, appear in the registry, and can be activated with one
click (or API call) — no restart required.

## 1. Prepare a dataset (YOLO format)

```
my_dataset/
  data.yaml
  train/images/*.jpg      val/images/*.jpg
  train/labels/*.txt      val/labels/*.txt
```

- One `.txt` per image, one line per object: `<class> <cx> <cy> <w> <h>` (normalized 0–1).
- `data.yaml`:
  ```yaml
  path: /abs/path/to/my_dataset
  train: train/images
  val: val/images
  nc: 3
  names: {0: helmet, 1: vest, 2: person}
  ```
- Label with **labelImg**, **CVAT** or **Roboflow** (export "YOLOv8" format).
- Guidance: ≥ 100 images/class for real tasks (more variety > more images);
  keep a clean held-out `val/` split; 640px imgsz is the standard starting point.

## 2. Validate

- UI: paste the `data.yaml` path → **Validate custom** (checks folders, counts, missing labels).
- CLI: `python -c "from training.dataset import validate_dataset; print(validate_dataset('...'))"`.

## 3. Train

- UI: choose run name, base weights (`yolov8n.pt` … or another custom `.pt` for
  continued training), epochs / imgsz / batch → **Start fine-tuning job**.
  Epoch logs and metrics stream live; final validation (mAP@0.5, mAP@0.5:0.95,
  precision, recall) is reported on completion.
- CLI: `python -m training.train --data .../data.yaml --epochs 50 --name ppe_v1 --batch 16`

Recommended starting points (single GPU or strong CPU):
| Scenario | epochs | imgsz | batch | base |
|---|---|---|---|---|
| Smoke test (synthetic demo) | 10–50 | 320 | 8–16 | yolov8n.pt |
| Small custom set (<1k imgs) | 50–100 | 640 | 16 | yolov8s.pt |
| Larger set / accuracy push | 100–200 | 640–960 | 16–32 | yolov8m.pt |

## 4. Evaluate & iterate

- Metrics live in the job result; full curves (PR, confusion matrix) under
  `models/runs/<name>/`.
- mAP@0.5 stuck low? Add harder negatives, fix mislabeled boxes, raise `imgsz`,
  train longer, or step up the base model (n → s → m).
- Overfitting (train ≫ val)? More augmentation is on by default; add data variety,
  use smaller base, or early-stop (take an earlier `best.pt`).

## 5. Deploy the tuned weights

- Already promoted: `models/<name>.pt` → registry → **Activate**.
- API: `POST /api/models/switch {"name": "<name>.pt"}`.
- Download from the job card (`GET /api/jobs/{id}/download`) to ship elsewhere.
- Roll back anytime by activating a stock checkpoint.

## 6. Provenance & reproducibility

- Every training job records dataset counts, hyperparams, elapsed time and metrics.
- `models/runs/<name>/args.yaml` stores the exact ultralytics config — commit the
  run name + metrics into your report for a reproducible claim.

## FAQ

- **CPU training too slow?** Use imgsz 320–480, yolov8n base, fewer epochs for
  iteration; run the final long training on a GPU machine (RunPod/Colab/Kaggle),
  then drop the resulting `.pt` into `models/`.
- **Class names wrong after switching?** Custom weights carry their own label map;
  the API returns `class_name` from the active model automatically.
- **Can I fine-tune the face recognizer?** SFace is a fixed embedding model —
  instead, "tune" recognition via gallery quality + the **calibration** tool
  (Face Gallery page), which sets the optimal threshold from your data.
