# VisionAI — Object Detection + Face Recognition 🎯🧑

A **fully deployable image-processing project**: generic **object detection** (YOLOv8, 80 COCO classes)
plus **face detection + face recognition** (YuNet + SFace) behind one FastAPI service with a web demo UI,
Docker support, and one-command cloud deploys.

Built for an image-processing course project: accurate out-of-the-box on CPU, no dlib/conda pain,
clean code you can explain in a viva/report.

---

## ✨ What it does

| Feature | Model | Notes |
|---|---|---|
| 📦 Object detection | **YOLOv8** (`yolov8n/s/m/l`) | 80 classes (person, car, dog, …), boxes + confidence |
| 😐 Face detection | **YuNet** (OpenCV Zoo ONNX) | Fast, accurate, CPU-friendly |
| 🧑 Face recognition | **SFace** embeddings + cosine match | Register your own gallery, tunable threshold |
| ✦ Combined analysis | YOLO + YuNet/SFace | One `/api/analyze` call does everything |
| 🖥️ Web demo | Served at `/` | Upload / drag-drop / webcam / gallery management |
| 🐳 Deploy | Dockerfile + Compose + Render | CPU-only friendly |

## 🗂️ Project structure

```
.
├── app/
│   ├── main.py              # FastAPI app + routes + UI mount
│   ├── config.py            # env-overridable settings
│   ├── schemas.py           # pydantic schemas
│   ├── detectors/
│   │   ├── object_detector.py  # YOLOv8 wrapper (lazy singleton)
│   │   └── face.py             # YuNet detector + SFace recognizer
│   ├── services/
│   │   ├── inference.py     # orchestration: detect/recognize/analyze/register
│   │   └── face_db.py       # persistent known-face embedding store (pickle)
│   ├── utils/               # image decode/encode, annotation drawing
│   └── static/              # demo UI (index.html/app.js/styles.css)
├── scripts/
│   ├── download_models.py   # fetch YuNet + SFace + YOLO weights
│   ├── register_faces.py    # bulk CLI registration from folders
│   └── demo.py              # local CLI inference (no server)
├── tests/                   # pytest (mocked API + real unit tests)
├── data/known_faces/        # <Name>/*.jpg gallery (also via web UI)
├── models/                  # downloaded weights (git-ignored)
├── Dockerfile  docker-compose.yml  render.yaml  Makefile
└── requirements.txt
```

## 🚀 Quickstart (local)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download weights (~50 MB face models + YOLO auto-fetch)
python scripts/download_models.py

# 3. Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- **Demo UI → http://localhost:8000**
- **Interactive API docs → http://localhost:8000/docs**
- Health check → `GET /api/health`

### CLI demo (no server)

```bash
python scripts/demo.py photo.jpg                 # combined
python scripts/demo.py photo.jpg --objects-only
python scripts/demo.py photo.jpg --faces-only --save outputs/out.jpg
```

## 🧑 Face recognition walkthrough

1. **Register** — in the UI's *Gallery* tab (or `POST /api/faces/register`): name + 2–5 clear,
   front-facing photos per person. Different lighting/angles help.
2. **Recognize** — *Analyze* or *Faces* tab, or `POST /api/recognize/faces`.
3. **Tune** — match threshold default `0.363` (OpenCV's SFace cosine recommendation):
   - Too many "Unknown"? lower to ~0.30.
   - Wrong names? raise to ~0.40–0.45 and add more photos per person.

Bulk CLI alternative:

```bash
mkdir -p data/known_faces/Ada data/known_faces/Bob
# ... copy 2-5 jpgs into each ...
python scripts/register_faces.py
```

Embeddings persist in `data/face_db.pkl` (mounted as a volume in Docker so you don't lose them).

## 📡 API reference

| Method & path | Input | Output |
|---|---|---|
| `GET /api/health` | — | status, model load state, identities |
| `GET /api/labels/objects` | — | 80 COCO class names |
| `POST /api/detect/objects?conf=0.25&iou=0.45` | `file` ⬆ | boxes, labels, annotated image (base64) |
| `POST /api/detect/faces` | `file` ⬆ | face boxes + landmarks, annotated image |
| `POST /api/recognize/faces?threshold=0.363` | `file` ⬆ | names + match scores, annotated image |
| `POST /api/analyze` | `file` ⬆ | objects + faces combined |
| `POST /api/faces/register` | `name` + `files[]` | embeddings added |
| `GET /api/faces` | — | identities + counts |
| `DELETE /api/faces/{name}` | — | delete one identity |
| `DELETE /api/faces` | — | clear all |

All image endpoints also accept `image_base64` (form field, plain or `data:` URL) instead of `file`,
and `?return_image=false` to skip the annotated image for speed.

cURL example:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@photo.jpg" | python -m json.tool
```

## 🎛️ Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `YOLO_MODEL_NAME` | `yolov8n.pt` | `yolov8s.pt`/`yolov8m.pt`/`yolov8l.pt` = more accurate, slower |
| `YOLO_CONF` / `YOLO_IOU` | `0.25` / `0.45` | detection thresholds |
| `FACE_SCORE_THRESH` | `0.6` | YuNet min face confidence |
| `FACE_MATCH_THRESH` | `0.363` | SFace cosine match threshold |
| `MAX_FILE_MB` | `10` | upload limit |
| `PORT` | `8000` | server port |

See `.env.example`. For **maximum accuracy** on a good machine: `YOLO_MODEL_NAME=yolov8m.pt`.

## 🐳 Deploy with Docker

```bash
docker compose up --build
# → http://localhost:8000
```

Face DB (`./data`), weights (`./models`) and `outputs/` are volume-mounted so registrations survive restarts.

## ☁️ Deploy to the cloud (free)

**Render** — push this repo to GitHub, then *New → Blueprint* and point at `render.yaml`
(or *New → Web Service*, Docker runtime, health check `/api/health`).

**Hugging Face Spaces** — new Space (Docker SDK) → push this repo; the Dockerfile serves on `$PORT`
(Spaces sets `PORT=7860` automatically if you leave the default CMD override — set Space port to 8000
or change CMD to use `$PORT`). Simplest: keep Docker SDK + port 7860 mapping:

```dockerfile
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

**Any VPS / EC2** — `docker compose up -d --build` behind nginx/Caddy for HTTPS.

## ✅ Verified end-to-end (real weights, CPU)

Sample run on the bundled test photos (`data/samples/`, from OpenCV's sample set):

| Test | Result |
|---|---|
| Face detect (Lena) | 1 face, conf **0.909**, ~109 ms CPU |
| Face register → recognize (same photo) | **score 1.00, matched** |
| Recognize perturbed copy (½ size, darker, blurred) | **score 0.909, matched** (threshold 0.363) — robust ✅ |
| Non-face photo (baboon) | 0 faces — no false positive ✅ |
| YOLOv8n objects (soccer photo) | **sports ball 0.95**, person 0.85 + crowd ✅ |
| Unknown face (Messi, not in gallery) | detected (0.92) → **"Unknown"** (score 0.13 < 0.363) ✅ |
| `pytest` | **9 passed** (mocked API + real unit tests) |

## 🧪 Tests

```bash
pytest -q
```

API tests mock the heavy models (fast, no weights needed); detector tests cover cosine matching,
face-DB persistence, and image codec round-trips for real.

## 📈 Accuracy notes (for your report)

- **YOLOv8n** (default): COCO mAP50 ≈ 37 — great speed/accuracy balance on CPU (~50–150 ms).
  Swap to **YOLOv8s/m** for +5–10 mAP at 2–4× cost. Raise `conf` to cut false positives.
- **YuNet**: state-of-the-art lightweight face detector; `FACE_SCORE_THRESH=0.6` filters junk.
- **SFace**: 128-D embeddings; cosine ≥ 0.363 ≈ same person (OpenCV benchmark threshold).
  Accuracy climbs steeply with 3–5 diverse enrollment photos per identity.
- Preprocessing keeps EXIF orientation correct and caps image side at 1280 px for stable latency.

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `Missing YuNet/SFace model` | `python scripts/download_models.py` (needs internet once) |
| `ultralytics is not installed` | `pip install -r requirements.txt` |
| Slow first request | Normal — models lazy-load; subsequent calls are fast |
| `torch` install is huge | CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu` first, then requirements |
| No faces found at registration | Use clear front-facing photos, ≥100 px face, no heavy filters |

## 📚 Viva / report talking points

- Two-stage face pipeline: **detect → alignCrop → embed → cosine-NN** against gallery.
- Why YOLOv8 single-shot > R-CNN two-stage for real-time; mAP vs latency tradeoff across n/s/m/l.
- Threshold tuning as precision/recall tradeoff (show the sliders in the UI).
- Deployment: ONNXRuntime-free OpenCV DNN + lazy singletons + Docker = reproducible anywhere.

## 📄 License

MIT — models: YOLOv8 (AGPL-3.0, Ultralytics) for research; YuNet/SFace (MIT, OpenCV Zoo).
Check upstream licenses before commercial use.
