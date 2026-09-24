import { useState } from "react";
import { Play, Boxes, Users, ChevronDown } from "lucide-react";
import clsx from "clsx";
import { api } from "../lib/api";
import type { AnalyzeResponse, Detection, FaceResult } from "../lib/types";
import { PageHeader, Slider, EmptyState } from "../components/bits";
import { Dropzone } from "../components/Dropzone";
import { useToast } from "../components/Toast";

type Mode = "analyze" | "objects" | "faces";

const MODES: { id: Mode; label: string }[] = [
  { id: "analyze", label: "Combined" },
  { id: "objects", label: "Objects only" },
  { id: "faces", label: "Faces only" },
];

export function ImageAnalysis() {
  const [mode, setMode] = useState<Mode>("analyze");
  const [file, setFile] = useState<File | null>(null);
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [thr, setThr] = useState(0.363);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<AnalyzeResponse | null>(null);
  const [showJson, setShowJson] = useState(false);
  const { push } = useToast();

  const run = async () => {
    if (!file) {
      push("info", "Choose an image first.");
      return;
    }
    setBusy(true);
    try {
      let out: AnalyzeResponse;
      if (mode === "analyze") {
        out = await api.analyze(file, { conf, iou, threshold: thr });
      } else if (mode === "objects") {
        const o = await api.detectObjects(file, { conf, iou });
        out = {
          objects: { ...o, error: undefined }, faces: { count: 0, faces: [], inference_ms: 0, threshold: thr, known_identities: [] },
          inference_ms: o.inference_ms, annotated_image: o.annotated_image, run_id: o.run_id,
        };
      } else {
        const f = await api.recognizeFaces(file, { threshold: thr });
        out = {
          objects: { count: 0, detections: [], inference_ms: 0, model: "" },
          faces: { ...f }, inference_ms: f.inference_ms, annotated_image: f.annotated_image,
        };
      }
      setRes(out);
      if (out.warnings?.length) push("info", out.warnings.join(" "));
      else push("success", `Done in ${out.inference_ms} ms — ${out.objects.count} objects, ${out.faces.count} faces.`);
    } catch (e: any) {
      push("error", e.message);
    }
    setBusy(false);
  };

  return (
    <div>
      <PageHeader title="Image Analysis" sub="Detect objects, detect & recognize faces — singly or combined" />
      <div className="grid xl:grid-cols-[380px_1fr] gap-3.5 items-start">
        <div className="card card-pad flex flex-col gap-4">
          <div>
            <div className="label mb-2">Task</div>
            <div className="grid grid-cols-3 gap-1.5">
              {MODES.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setMode(m.id)}
                  className={clsx(
                    "rounded-xl px-2 py-2 text-[13px] font-semibold border transition",
                    mode === m.id
                      ? "bg-gradient-to-r from-cyan-400/20 to-emerald-400/15 border-cyan-400/50 text-white"
                      : "border-white/10 text-slate-400 hover:text-slate-200 hover:bg-white/5"
                  )}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>
          <Dropzone file={file} onFile={setFile} accept="image/*" kind="image" />
          <div className="flex flex-col gap-3">
            {(mode === "analyze" || mode === "objects") && (
              <>
                <Slider label="Object confidence" value={conf} min={0.05} max={0.9} step={0.05} onChange={setConf} />
                <Slider label="NMS IoU" value={iou} min={0.1} max={0.9} step={0.05} onChange={setIou} />
              </>
            )}
            {(mode === "analyze" || mode === "faces") && (
              <Slider label="Face match threshold" value={thr} min={0.2} max={0.7} step={0.005} onChange={setThr} format={(v) => v.toFixed(3)} />
            )}
          </div>
          <button className="btn-primary w-full" onClick={run} disabled={busy || !file}>
            <Play size={16} /> {busy ? "Analyzing…" : "Run analysis"}
          </button>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            Every run is logged to history with a thumbnail. Tune the face threshold on the
            Face Gallery page after calibrating against your gallery.
          </p>
        </div>

        <div className="card card-pad">
          {!res ? (
            <EmptyState title="No result yet" sub="Upload an image on the left and run an analysis. The annotated output and detection tables will appear here." />
          ) : (
            <div>
              <div className="flex items-center gap-2 flex-wrap mb-3">
                <span className="badge bg-emerald-400/10 text-emerald-300 border border-emerald-400/30">{res.inference_ms} ms</span>
                <span className="badge bg-white/5 border border-white/10">{res.objects.count} objects</span>
                <span className="badge bg-white/5 border border-white/10">{res.faces.count} faces</span>
                {res.run_id && <span className="badge bg-white/5 border border-white/10">run #{res.run_id}</span>}
                {res.objects.model && <span className="badge bg-white/5 border border-white/10 font-mono">{res.objects.model.split("/").pop()}</span>}
              </div>
              {res.annotated_image && (
                <img src={res.annotated_image} alt="annotated result" className="w-full max-h-[440px] object-contain rounded-xl border border-white/10 bg-ink-950" />
              )}
              <div className="grid md:grid-cols-2 gap-3.5 mt-4">
                <ResultTable
                  title="Objects" icon={<Boxes size={15} />}
                  rows={res.objects.detections.map((d: Detection) => ({
                    name: d.class_name,
                    meta: `${(d.confidence * 100).toFixed(1)}% · [${d.bbox.map((v) => Math.round(v)).join(", ")}]`,
                    pct: d.confidence,
                  }))}
                  empty="No objects detected."
                />
                <ResultTable
                  title="Faces" icon={<Users size={15} />}
                  rows={res.faces.faces.map((f: FaceResult) => ({
                    name: f.name || "face",
                    meta: f.score != null
                      ? `${f.matched ? "matched" : "unknown"} · ${(f.score * 100).toFixed(1)}%`
                      : `${(f.confidence * 100).toFixed(1)}%`,
                    pct: f.score ?? f.confidence,
                    good: f.matched,
                  }))}
                  empty="No faces detected."
                />
              </div>
              <button onClick={() => setShowJson(!showJson)} className="mt-4 text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1">
                <ChevronDown size={14} className={clsx("transition", showJson && "rotate-180")} /> Raw JSON
              </button>
              {showJson && (
                <pre className="mt-2 bg-ink-950/80 border border-white/10 rounded-xl p-3 text-[11px] font-mono overflow-auto max-h-[260px]">
                  {JSON.stringify(strip(res), null, 2)}
                </pre>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function strip(r: AnalyzeResponse) {
  const c: any = JSON.parse(JSON.stringify(r));
  if (c.annotated_image) c.annotated_image = "<base64 jpeg omitted>";
  return c;
}

function ResultTable({ title, icon, rows, empty }: {
  title: string; icon: React.ReactNode;
  rows: { name: string; meta: string; pct: number; good?: boolean }[];
  empty: string;
}) {
  return (
    <div>
      <div className="label mb-2 flex items-center gap-1.5">{icon} {title} <span className="text-cyan-300">({rows.length})</span></div>
      {rows.length === 0 && <p className="text-xs text-slate-500">{empty}</p>}
      <div className="flex flex-col gap-1.5 max-h-[240px] overflow-y-auto pr-1">
        {rows.map((r, i) => (
          <div key={i} className="rounded-lg bg-white/[0.04] border border-white/10 px-2.5 py-2">
            <div className="flex justify-between gap-2 text-[13px]">
              <b className="truncate">{r.good === false && r.name === "Unknown" ? "❔ Unknown" : r.name}</b>
              <span className="font-mono text-emerald-300 text-xs shrink-0">{(r.pct * 100).toFixed(1)}%</span>
            </div>
            <div className="text-[11px] text-slate-500 font-mono truncate">{r.meta}</div>
            <div className="h-1 rounded-full bg-white/10 mt-1.5 overflow-hidden">
              <div className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-emerald-400" style={{ width: `${Math.round(r.pct * 100)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
