# VisionAI

Visual intelligence platform: YOLOv8 object detection, YuNet + SFace face
recognition, video analysis, run history, gallery calibration, and model
fine-tuning — one FastAPI service with a React dashboard.

## Features

| Area | Details |
|---|---|
| Object detection | YOLOv8, 80 COCO classes; switchable checkpoints (n/s/m/l + custom) |
| Face recognition | YuNet detection, SFace embeddings, persistent gallery, cosine matching |
| Video analysis | Background jobs with progress/logs, annotated MP4, timeline charts |
| Fine-tuning | Dataset validation, training jobs, metrics, one-click weight activation |
| Calibration | Data-driven match threshold (FAR/FRR, ROC, max-margin suggestion) |
| History | SQLite-backed run log with thumbnails, stats, latency trends |

## Quickstart

Docker (recommended):

```bash
docker compose up --build
# dashboard + API at http://localhost:8000 (weights download on first run)
```

Local development:

```bash
pip install -r requirements-dev.txt
python scripts/download_models.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive API docs: `/docs`.

## Web UI

A single dependency-free page at `/`: analyze images (combined/objects/faces),
enroll and manage known faces, one-click threshold calibration, and recent
runs. Video, training, and registry operations stay on the API (see `/docs`).

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health`, `GET /api/stats` | Liveness, dashboard roll-up |
| `POST /api/analyze` | Objects + faces in one call (logged to history) |
| `POST /api/detect/objects`, `/api/detect/faces`, `/api/recognize/faces` | Single-task inference |
| `GET/POST/DELETE /api/faces`, `POST /api/faces/calibrate` | Gallery CRUD, calibration |
| `POST /api/video/analyze`, `GET /api/jobs…` | Video jobs, progress, artifacts |
| `GET/DELETE /api/history…` | Run log, thumbnails |
| `GET /api/models`, `POST /api/models/switch` | Weight registry, activation |
| `GET /api/training/status`, `POST /api/training/{demo-prepare,validate,start}` | Fine-tuning |

Uploads are multipart `file` (images ≤ 10 MB, video ≤ 100 MB).

## Fine-tuning

See [docs/TRAINING.md](docs/TRAINING.md) for the full playbook. Summary:
prepare a YOLO-format dataset, validate it via the API, start a
training job, and activate the promoted weights from the registry — or run
`python -m training.train --data .../data.yaml --epochs 50 --name v1`.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `YOLO_MODEL_NAME` | `yolov8s.pt` | Base checkpoint (`n` is faster, `m/l` more accurate) |
| `YOLO_CONF` / `YOLO_IOU` | `0.25` / `0.45` | Detection thresholds |
| `FACE_MATCH_THRESH` | `0.363` | SFace cosine default (calibrate per gallery) |
| `MAX_FILE_MB` / `MAX_VIDEO_MB` | `10` / `100` | Upload caps |
| `MAX_VIDEO_FRAMES` | `900` | Analyzed-frame cap per video |
| `CORS_ORIGINS` | `*` | Allowed origins |

## Layout

```
app/            FastAPI service (api/ routers, detectors/, services/, utils/)
app/static/     Single-page UI (no build step)
training/       dataset utils + fine-tuning runner (API and CLI)
scripts/        model download, local CLI demo
tests/          pytest suite (mocked API + service unit tests)
docs/           fine-tuning playbook
```

## Development

```bash
pytest -q                     # backend tests
docker compose up --build     # full-stack check
```

CI runs the backend suite on every push.

## Validation

Spot-checked on CPU with release weights: YOLOv8s detects the expected
objects with fewer false positives than the nano model (~170 ms/image on CPU);
face enroll-then-recognize scores 1.00 (same photo) and 0.91 (degraded copy)
with unknown faces correctly rejected; a 795-frame video job produces counts
plus an annotated MP4; a 50-epoch fine-tune on the bundled synthetic set
reaches mAP@0.5 0.995; gallery calibration separates genuine (mean 0.94)
from impostor (mean 0.14) pairs.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Missing YuNet/SFace weights | `python scripts/download_models.py` (one-time download) |
| Slow first request | Expected — models lazy-load, then stay resident |
| Huge `torch` install | Install the CPU wheel first (see pytorch.org), then requirements |

## License

MIT (see LICENSE). Bundled third-party weights keep their upstream licenses:
YOLOv8 (AGPL-3.0, Ultralytics), YuNet/SFace (MIT, OpenCV Zoo).
Sample photos are from the OpenCV sample set, for testing only.
