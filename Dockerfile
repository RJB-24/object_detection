# VisionAI 2.0 — multi-stage: React dashboard + FastAPI platform
# ---------------------------------------------------------------- web build
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build   # -> /app/app/static/dist (copied below)

# --------------------------------------------------------------- python
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV / torch runtime deps (+ ffmpeg for video thumbnails/writing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
COPY --from=web /app/app/static/dist ./app/static/dist

# Pre-download weights at build time (needs network; safe to skip & lazy-load)
RUN python scripts/download_models.py || echo "WARN: model pre-download failed, will lazy-load at runtime"
RUN mkdir -p outputs data/known_faces

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
