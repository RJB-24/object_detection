import { useEffect, useState } from "react";
import { Play, Cpu, Database, FolderCheck, RefreshCw, Download, Check } from "lucide-react";
import { api } from "../lib/api";
import { shortModel } from "../lib/format";
import type { Job, ModelItem, TrainingStatus } from "../lib/types";
import { PageHeader, Spinner } from "../components/bits";
import { JobLogs, JobProgress, useJobPoll } from "../components/JobMonitor";
import { useToast } from "../components/Toast";

export function FineTune() {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [models, setModels] = useState<ModelItem[]>([]);
  const [active, setActive] = useState("");
  const [dataYaml, setDataYaml] = useState("");
  const [datasetInfo, setDatasetInfo] = useState<any>(null);
  const [epochs, setEpochs] = useState(20);
  const [imgsz, setImgsz] = useState(640);
  const [batch, setBatch] = useState(16);
  const [base, setBase] = useState("yolov8n.pt");
  const [runName, setRunName] = useState("custom");
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const job = useJobPoll(jobId, 2000);
  const { push } = useToast();

  const load = async () => {
    try {
      const [s, m] = await Promise.all([api.trainingStatus(), api.models()]);
      setStatus(s);
      setModels(m.models);
      setActive(m.active);
    } catch (e: any) {
      push("error", e.message);
    }
  };
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const prepareDemo = async () => {
    setBusy(true);
    try {
      const info = await api.demoPrepare();
      setDatasetInfo(info);
      setDataYaml(info.data_yaml);
      push("success", `Demo dataset ready: ${info.splits.train.images} train / ${info.splits.val.images} val.`);
    } catch (e: any) {
      push("error", e.message);
    }
    setBusy(false);
  };

  const validate = async () => {
    if (!dataYaml.trim()) return push("info", "Enter a data.yaml path first.");
    setBusy(true);
    try {
      const info = await api.validateDataset(dataYaml.trim());
      setDatasetInfo(info);
      push("success", `Dataset OK: ${info.splits.train.images} train / ${info.splits.val.images} val, ${info.nc} classes.`);
    } catch (e: any) {
      push("error", e.message);
    }
    setBusy(false);
  };

  const start = async () => {
    if (!dataYaml.trim()) return push("info", "Prepare or validate a dataset first.");
    if (!status?.available) return push("error", "Training stack unavailable (torch/ultralytics missing).");
    setBusy(true);
    try {
      const r = await api.startTraining({
        data_yaml: dataYaml.trim(), epochs, imgsz, base_model: base, name: runName.trim() || "custom", batch,
      });
      setJobId(r.job_id);
      push("success", "Training job submitted — watch the live log below.");
    } catch (e: any) {
      push("error", e.message);
    }
    setBusy(false);
  };

  const switchModel = async (name: string) => {
    try {
      const r = await api.switchModel(name);
      setActive(r.active);
      setModels((ms) => ms.map((m) => ({ ...m, active: m.name === name })));
      push("success", `Active model: ${shortModel(r.active)} (loads on next inference).`);
    } catch (e: any) {
      push("error", e.message);
    }
  };

  const trainResult = job?.status === "done" ? job.result : null;

  return (
    <div>
      <PageHeader title="Fine-tune" sub="Train YOLO on your own data and manage the model registry" />
      {!status ? (
        <Spinner label="Checking training stack…" />
      ) : (
        <div className="grid xl:grid-cols-2 gap-3.5 items-start">
          <div className="flex flex-col gap-3.5">
            <div className="card card-pad">
              <div className="label mb-3 flex items-center gap-1.5"><Cpu size={14} /> Training environment</div>
              {status.available ? (
                <div className="flex items-center gap-2 text-sm flex-wrap">
                  <span className="badge bg-emerald-400/10 text-emerald-300 border border-emerald-400/30">ready</span>
                  <span className="font-mono text-[13px]">torch {status.torch}</span>
                  <span className="badge bg-white/5 border border-white/10">{status.device}</span>
                </div>
              ) : (
                <p className="text-sm text-red-300">Unavailable: {status.error}</p>
              )}
            </div>

            <div className="card card-pad flex flex-col gap-3">
              <div className="label flex items-center gap-1.5"><Database size={14} /> 1 · Dataset</div>
              <div className="flex gap-2">
                <button className="btn-ghost flex-1 !py-2 text-[13px]" onClick={prepareDemo} disabled={busy}>
                  Generate demo dataset
                </button>
                <button className="btn-ghost flex-1 !py-2 text-[13px]" onClick={validate} disabled={busy}>
                  <FolderCheck size={15} /> Validate custom
                </button>
              </div>
              <input
                className="input font-mono !text-[13px]" placeholder="/abs/path/to/data.yaml or data/…"
                value={dataYaml} onChange={(e) => setDataYaml(e.target.value)}
              />
              {datasetInfo && (
                <div className="rounded-xl bg-white/[0.04] border border-white/10 p-3 text-[13px]">
                  <div className="font-semibold mb-1">{datasetInfo.demo ? "Synthetic demo set" : "Custom dataset"} ✓ valid</div>
                  <div className="text-slate-400 font-mono text-xs">
                    train {datasetInfo.splits.train.images} · val {datasetInfo.splits.val.images} · nc={datasetInfo.nc}
                    <br />classes: {(datasetInfo.names ? Object.values(datasetInfo.names) : []).join(", ")}
                  </div>
                </div>
              )}
              <details className="text-xs text-slate-400">
                <summary className="cursor-pointer hover:text-slate-200">How do I prepare a real dataset?</summary>
                <div className="mt-2 leading-relaxed font-mono text-[11px] bg-ink-950/70 border border-white/10 rounded-xl p-3 whitespace-pre-wrap">
{`dataset/
  data.yaml              # path/train/val/nc/names
  train/images/*.jpg
  train/labels/*.txt     # class cx cy w h (normalized)
  val/images/*.jpg
  val/labels/*.txt

Label with labelImg / Roboflow, export
"YOLOv8" format, upload the folder to the
server, paste its data.yaml path above.`}
                </div>
              </details>
            </div>

            <div className="card card-pad flex flex-col gap-3">
              <div className="label flex items-center gap-1.5"><Play size={14} /> 2 · Train</div>
              <div className="grid grid-cols-2 gap-2.5">
                <label className="block"><span className="label">Run name</span>
                  <input className="input mt-1" value={runName} onChange={(e) => setRunName(e.target.value)} />
                </label>
                <label className="block"><span className="label">Base weights</span>
                  <select className="input mt-1" value={base} onChange={(e) => setBase(e.target.value)}>
                    {models.map((m) => (
                      <option key={m.name} value={m.name}>{m.name}{m.custom ? " (custom)" : ""}{m.available_locally ? "" : " · downloads"}</option>
                    ))}
                  </select>
                </label>
                <label className="block"><span className="label">Epochs</span>
                  <input className="input mt-1" type="number" min={1} max={300} value={epochs} onChange={(e) => setEpochs(+e.target.value)} />
                </label>
                <label className="block"><span className="label">Image size</span>
                  <input className="input mt-1" type="number" min={320} max={1280} step={32} value={imgsz} onChange={(e) => setImgsz(+e.target.value)} />
                </label>
                <label className="block col-span-2"><span className="label">Batch size</span>
                  <input className="input mt-1" type="number" min={1} max={128} value={batch} onChange={(e) => setBatch(+e.target.value)} />
                </label>
              </div>
              <button className="btn-primary w-full" onClick={start} disabled={busy || !status.available}>
                <Play size={16} /> Start fine-tuning job
              </button>
            </div>
          </div>

          <div className="flex flex-col gap-3.5">
            <div className="card card-pad">
              <div className="label mb-3">Training job {job ? `#${job.id}` : ""}</div>
              {!job && <p className="text-sm text-slate-500">Configure a dataset + hyperparameters, then start. Live epoch logs appear here.</p>}
              {job && (
                <div className="flex flex-col gap-3">
                  <JobProgress job={job} />
                  <JobLogs job={job} maxH="max-h-[300px]" />
                  {trainResult && (
                    <TrainingResult job={job} result={trainResult} onRefresh={load} />
                  )}
                </div>
              )}
            </div>

            <div className="card card-pad">
              <div className="flex items-center justify-between mb-3">
                <span className="label">Model registry</span>
                <button className="btn-ghost !py-1.5 !px-3 text-xs" onClick={load}>
                  <RefreshCw size={13} /> Refresh
                </button>
              </div>
              <div className="flex flex-col gap-1.5">
                {models.map((m) => (
                  <div key={m.name} className="flex items-center gap-2.5 rounded-xl bg-white/[0.04] border border-white/10 px-3 py-2.5 text-sm">
                    <span className="font-mono text-[13px] truncate flex-1" title={m.name}>{m.name}</span>
                    {m.custom && <span className="badge bg-amber-400/10 text-amber-300 border border-amber-400/30">custom</span>}
                    <span className="text-[11px] text-slate-500 font-mono">
                      {m.available_locally ? `${m.size_mb} MB` : "remote"}
                    </span>
                    {m.active || active.endsWith(m.name) ? (
                      <span className="badge bg-emerald-400/10 text-emerald-300 border border-emerald-400/30">
                        <Check size={12} /> active
                      </span>
                    ) : (
                      <button className="btn-ghost !py-1 !px-2.5 text-xs" onClick={() => switchModel(m.name)}>
                        Activate
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function TrainingResult({ job, result, onRefresh }: { job: Job; result: any; onRefresh: () => void }) {
  const m = result.metrics || {};
  return (
    <div className="rounded-xl border border-emerald-400/30 bg-emerald-400/5 p-3.5">
      <div className="text-sm font-semibold mb-2">Training complete · {result.elapsed_s}s · {result.epochs_done} epochs</div>
      <div className="grid grid-cols-4 gap-2 text-center mb-3">
        {[
          ["mAP@0.5", m.map50],
          ["mAP@.5:.95", m.map50_95],
          ["Precision", m.precision],
          ["Recall", m.recall],
        ].map(([k, v]: any) => (
          <div key={k} className="rounded-lg bg-ink-950/60 border border-white/10 py-2">
            <div className="font-mono font-bold">{v ?? "—"}</div>
            <div className="text-[10px] text-slate-500">{k}</div>
          </div>
        ))}
      </div>
      {result.promoted_weights && (
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-xs">models/{result.promoted_weights}</span>
          <a className="btn-ghost !py-1.5 !px-3 text-xs" href={`/api/jobs/${job.id}/download`}>
            <Download size={13} /> Weights
          </a>
          <button className="btn-ghost !py-1.5 !px-3 text-xs" onClick={onRefresh}>
            <RefreshCw size={13} /> Update registry
          </button>
        </div>
      )}
    </div>
  );
}
