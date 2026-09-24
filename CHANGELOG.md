# Changelog

## [2.0.0] — 2026-09-24 — Platform release
### Added
- React + TypeScript + Tailwind dashboard (6 pages: Dashboard, Image/Video Analysis,
  Face Gallery, Fine-tune, History) served by FastAPI; multi-stage Dockerfile.
- Video analysis as background jobs: progress, live logs, annotated MP4, timeline charts.
- YOLO fine-tuning pipeline: dataset validation, training jobs with epoch callbacks,
  validation metrics, weights promotion, downloadable artifacts.
- Model registry: discover local/custom weights, one-click activation (offline-safe).
- Face-threshold calibration: genuine/impostor analysis, FAR/FRR + ROC, max-margin suggestion.
- Persistent run history (SQLite) with thumbnails and dashboard stats/trends.
- Training playbook (docs/TRAINING.md), LICENSE, CHANGELOG; tests up to 14.

### Changed
- `/api/analyze` degrades gracefully per-branch and logs a single history entry.

## [1.0.0] — 2026-09-24 — Initial release
- FastAPI inference (YOLOv8 objects, YuNet+SFace faces), single-page demo UI,
  Docker/Compose/Render files, download + demo scripts, 9 tests.
