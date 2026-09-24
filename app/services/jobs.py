"""Background job manager for long tasks (video analysis, model training).

Jobs run in worker threads with progress reporting, live logs and a final
result payload. In-memory (single-instance) by design — history of *image*
runs is persisted separately in SQLite (see history.py).
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

MAX_JOBS = 100
MAX_LOG_LINES = 500


@dataclass
class Job:
    id: str
    kind: str  # "video" | "training" | ...
    label: str
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0  # 0..1
    message: str = "Queued"
    logs: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.logs.append(line)
        if len(self.logs) > MAX_LOG_LINES:
            del self.logs[: len(self.logs) - MAX_LOG_LINES]
        self.updated_at = time.time()

    def set_progress(self, frac: float, message: str | None = None) -> None:
        self.progress = max(0.0, min(1.0, float(frac)))
        if message is not None:
            self.message = message
        self.updated_at = time.time()

    def to_dict(self, include_logs: bool = True) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "logs": list(self.logs) if include_logs else [],
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()

    def submit(
        self,
        kind: str,
        label: str,
        fn: Callable[[Job], dict | None],
    ) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_JOBS:
                self._jobs.popitem(last=False)

        def _runner() -> None:
            job.status = "running"
            job.log(f"Started: {label}")
            try:
                result = fn(job)
                job.result = result or {}
                job.status = "done"
                job.set_progress(1.0, "Completed")
                job.log("Completed successfully")
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = str(exc)
                job.message = f"Failed: {exc}"
                job.log(f"ERROR: {exc}\n{traceback.format_exc(limit=5)}")

        threading.Thread(target=_runner, daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, kind: str | None = None, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = list(reversed(self._jobs.values()))
        if kind:
            jobs = [j for j in jobs if j.kind == kind]
        return jobs[:limit]


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = JobManager()
    return _manager
