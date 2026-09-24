export interface Detection {
  bbox: number[];
  confidence: number;
  class_id: number;
  class_name: string;
}

export interface FaceResult {
  bbox: number[];
  confidence: number;
  name?: string;
  score?: number;
  matched?: boolean;
  landmarks?: number[][];
}

export interface AnalyzeResponse {
  objects: { count: number; detections: Detection[]; inference_ms: number; model: string; error?: string };
  faces: { count: number; faces: FaceResult[]; inference_ms: number; threshold: number; known_identities: string[]; error?: string };
  inference_ms: number;
  warnings?: string[];
  annotated_image?: string;
  run_id?: number;
}

export interface ObjectsResponse {
  count: number;
  detections: Detection[];
  inference_ms: number;
  model: string;
  annotated_image?: string;
  run_id?: number;
}

export interface RecognizeResponse {
  count: number;
  faces: FaceResult[];
  inference_ms: number;
  threshold: number;
  known_identities: string[];
  annotated_image?: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  models: {
    yolo: { model: string; loaded: boolean; error: string | null; classes?: number; device?: string };
    face: { loaded: boolean; error: string | null; det_exists: boolean; rec_exists: boolean };
  };
  face_identities: string[];
  face_embeddings: number;
}

export interface PlatformStats {
  total_runs: number;
  total_objects: number;
  total_faces: number;
  total_matched: number;
  avg_ms: number;
  by_kind: Record<string, number>;
  trend: { id: number; ms: number; objects: number; faces: number }[];
  faces: { identities: string[]; embeddings: number };
  active_model: string;
}

export interface HistoryRun {
  id: number;
  ts: number;
  kind: string;
  source: string;
  objects: number;
  faces: number;
  matched: number;
  ms: number;
  thumb: string | null;
  summary: string;
}

export interface Job {
  id: string;
  kind: string;
  label: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  message: string;
  logs: string[];
  result: any;
  error: string | null;
  created_at: number;
  updated_at: number;
}

export interface VideoResult {
  mode: string;
  src_fps: number;
  src_frames: number;
  frames_analyzed: number;
  frame_stride: number;
  class_counts: Record<string, number>;
  people_counts: Record<string, number>;
  timeline: { t: number; objects: number; faces: number }[];
  thumbnail: string | null;
  download: string;
}

export interface Calibration {
  identities: string[];
  counts: Record<string, number>;
  per_identity: Record<string, { photos: number; mean_intra_similarity: number | null; min_intra_similarity: number | null; quality: string }>;
  genuine_pairs: number;
  impostor_pairs: number;
  genuine_mean: number | null;
  impostor_mean: number | null;
  sufficient_data: boolean;
  default_threshold: number;
  suggested_threshold: number;
  eer_threshold: number;
  thresholds: { threshold: number; far: number; frr: number; youden: number }[];
  roc: { fpr: number; tpr: number; threshold: number }[];
  guidance: string;
}

export interface ModelItem {
  name: string;
  available_locally: boolean;
  size_mb: number | null;
  custom: boolean;
  active: boolean;
}

export interface TrainingStatus {
  available: boolean;
  torch?: string;
  device?: string;
  error?: string;
}
