import type {
  AnalyzeResponse, Calibration, HealthResponse, HistoryRun, Job,
  ModelItem, ObjectsResponse, PlatformStats, RecognizeResponse, TrainingStatus,
} from "./types";

async function handle<T>(res: Response): Promise<T> {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = (data as any)?.detail || `Request failed (${res.status})`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data as T;
}

const q = (p: Record<string, any>) => {
  const s = new URLSearchParams();
  Object.entries(p).forEach(([k, v]) => {
    if (v !== undefined && v !== null) s.set(k, String(v));
  });
  const str = s.toString();
  return str ? `?${str}` : "";
};

export const api = {
  health: () => fetch("/api/health").then((r) => handle<HealthResponse>(r)),
  stats: () => fetch("/api/stats").then((r) => handle<PlatformStats>(r)),

  analyze: (file: File, o: { conf: number; iou: number; threshold: number }) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch(`/api/analyze${q({ conf: o.conf, iou: o.iou, threshold: o.threshold })}`, {
      method: "POST", body: fd,
    }).then((r) => handle<AnalyzeResponse>(r));
  },
  detectObjects: (file: File, o: { conf: number; iou: number }) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch(`/api/detect/objects${q({ conf: o.conf, iou: o.iou })}`, {
      method: "POST", body: fd,
    }).then((r) => handle<ObjectsResponse>(r));
  },
  recognizeFaces: (file: File, o: { threshold: number }) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch(`/api/recognize/faces${q({ threshold: o.threshold })}`, {
      method: "POST", body: fd,
    }).then((r) => handle<RecognizeResponse>(r));
  },

  listFaces: () =>
    fetch("/api/faces").then((r) => handle<{ identities: string[]; counts: Record<string, number>; total_embeddings: number }>(r)),
  registerFace: (name: string, files: File[]) => {
    const fd = new FormData();
    fd.append("name", name);
    files.forEach((f) => fd.append("files", f));
    return fetch("/api/faces/register", { method: "POST", body: fd }).then((r) => handle<any>(r));
  },
  deleteFace: (name: string) =>
    fetch(`/api/faces/${encodeURIComponent(name)}`, { method: "DELETE" }).then((r) => handle<any>(r)),
  clearFaces: () => fetch("/api/faces", { method: "DELETE" }).then((r) => handle<any>(r)),
  calibrate: () => fetch("/api/faces/calibrate", { method: "POST" }).then((r) => handle<Calibration>(r)),

  history: (limit = 50, offset = 0, kind?: string) =>
    fetch(`/api/history${q({ limit, offset, kind: kind || undefined })}`).then((r) =>
      handle<{ total: number; runs: HistoryRun[] }>(r)
    ),
  deleteRun: (id: number) => fetch(`/api/history/${id}`, { method: "DELETE" }).then((r) => handle<any>(r)),
  clearHistory: () => fetch("/api/history", { method: "DELETE" }).then((r) => handle<any>(r)),

  startVideo: (file: File, o: { mode: string; frame_stride: number; conf: number; threshold: number }) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("mode", o.mode);
    fd.append("frame_stride", String(o.frame_stride));
    fd.append("conf", String(o.conf));
    fd.append("threshold", String(o.threshold));
    return fetch("/api/video/analyze", { method: "POST", body: fd }).then((r) =>
      handle<{ job_id: string; status: string; poll: string }>(r)
    );
  },
  job: (id: string) => fetch(`/api/jobs/${id}`).then((r) => handle<Job>(r)),
  jobs: (kind?: string) => fetch(`/api/jobs${q({ kind: kind || undefined })}`).then((r) => handle<{ jobs: Job[] }>(r)),

  models: () => fetch("/api/models").then((r) => handle<{ active: string; models: ModelItem[] }>(r)),
  switchModel: (name: string) =>
    fetch("/api/models/switch", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).then((r) => handle<{ active: string }>(r)),

  trainingStatus: () => fetch("/api/training/status").then((r) => handle<TrainingStatus>(r)),
  demoPrepare: () => fetch("/api/training/demo-prepare", { method: "POST" }).then((r) => handle<any>(r)),
  validateDataset: (dataYaml: string) =>
    fetch("/api/training/validate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_yaml: dataYaml }),
    }).then((r) => handle<any>(r)),
  startTraining: (body: { data_yaml: string; epochs: number; imgsz: number; base_model: string; name: string; batch: number }) =>
    fetch("/api/training/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => handle<{ job_id: string; status: string; poll: string }>(r)),
};

