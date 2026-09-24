import { useState } from "react";
import { Play, Download, Film } from "lucide-react";
import clsx from "clsx";
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  BarChart, Bar,
} from "recharts";
import { api } from "../lib/api";
import type { VideoResult } from "../lib/types";
import { PageHeader, Slider } from "../components/bits";
import { Dropzone } from "../components/Dropzone";
import { JobLogs, JobProgress, useJobPoll } from "../components/JobMonitor";
import { useToast } from "../components/Toast";

const MODES = [
  { id: "combined", label: "Objects + Faces" },
  { id: "objects", label: "Objects only" },
  { id: "faces", label: "Faces only" },
];

export function VideoAnalysis() {
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState("combined");
  const [stride, setStride] = useState(5);
  const [conf, setConf] = useState(0.25);
  const [thr, setThr] = useState(0.363);
  const [jobId, setJobId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const job = useJobPoll(jobId);
  const { push } = useToast();

  const start = async () => {
    if (!file) {
      push("info", "Choose a video first.");
      return;
    }
    setStarting(true);
    try {
      const r = await api.startVideo(file, { mode, frame_stride: stride, conf, threshold: thr });
      setJobId(r.job_id);
      push("success", "Video job submitted — progress streams below.");
    } catch (e: any) {
      push("error", e.message);
    }
    setStarting(false);
  };

  const result: VideoResult | null = job?.status === "done" ? job.result : null;
  const classData = result ? Object.entries(result.class_counts).map(([name, value]) => ({ name, value })) : [];
  const peopleData = result ? Object.entries(result.people_counts).map(([name, value]) => ({ name, value })) : [];

  return (
    <div>
      <PageHeader title="Video Analysis" sub="Sampled-frame inference with an annotated MP4 output" />
      <div className="grid xl:grid-cols-[380px_1fr] gap-3.5 items-start">
        <div className="card card-pad flex flex-col gap-4">
          <div>
            <div className="label mb-2">Task</div>
            <div className="grid grid-cols-3 gap-1.5">
              {MODES.map((m) => (
                <button
                  key={m.id} onClick={() => setMode(m.id)}
                  className={clsx(
                    "rounded-xl px-2 py-2 text-[12px] font-semibold border transition",
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
          <Dropzone file={file} onFile={setFile} accept="video/*" kind="video" />
          <div className="flex flex-col gap-3">
            <Slider label="Frame stride (analyze every Nth)" value={stride} min={1} max={30} step={1} onChange={setStride} format={(v) => String(Math.round(v))} />
            {(mode === "combined" || mode === "objects") && (
              <Slider label="Object confidence" value={conf} min={0.05} max={0.9} step={0.05} onChange={setConf} />
            )}
            {(mode === "combined" || mode === "faces") && (
              <Slider label="Face match threshold" value={thr} min={0.2} max={0.7} step={0.005} onChange={setThr} format={(v) => v.toFixed(3)} />
            )}
          </div>
          <button className="btn-primary w-full" onClick={start} disabled={starting || !file}>
            <Play size={16} /> {starting ? "Submitting…" : "Analyze video"}
          </button>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            Higher stride = faster but coarser. Jobs run in the background — you can navigate
            away and come back; up to 900 analyzed frames per video.
          </p>
        </div>

        <div className="flex flex-col gap-3.5">
          <div className="card card-pad">
            <div className="label mb-3 flex items-center gap-1.5"><Film size={14} /> Job {job ? `#${job.id}` : ""}</div>
            {!job && <p className="text-sm text-slate-500">Submit a video to start a background job. Progress and logs stream here live.</p>}
            {job && (
              <div className="flex flex-col gap-3">
                <JobProgress job={job} />
                <JobLogs job={job} />
              </div>
            )}
          </div>

          {result && (
            <>
              <div className="card card-pad">
                <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
                  <span className="label">Result · {result.frames_analyzed} frames analyzed · {result.src_fps} fps source</span>
                  <a className="btn-primary !py-2 text-[13px]" href={result.download}>
                    <Download size={15} /> Annotated MP4
                  </a>
                </div>
                {result.thumbnail && (
                  <img src={result.thumbnail} alt="video thumbnail" className="w-full max-h-[260px] object-contain rounded-xl border border-white/10 bg-ink-950 mb-3" />
                )}
                <div className="label mb-2">Detections over time</div>
                <ResponsiveContainer width="100%" height={190}>
                  <AreaChart data={result.timeline}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                    <XAxis dataKey="t" tickFormatter={(v) => `${v}s`} minTickGap={40} />
                    <YAxis width={30} allowDecimals={false} />
                    <Tooltip contentStyle={{ background: "#0e1529", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10, fontSize: 12 }} />
                    <Area type="monotone" dataKey="objects" stackId="1" stroke="#34d399" fill="#34d39955" />
                    <Area type="monotone" dataKey="faces" stackId="1" stroke="#a78bfa" fill="#a78bfa55" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="grid md:grid-cols-2 gap-3.5">
                <div className="card card-pad">
                  <div className="label mb-2">Top object classes</div>
                  {classData.length ? (
                    <ResponsiveContainer width="100%" height={Math.max(120, classData.length * 30)}>
                      <BarChart data={classData} layout="vertical">
                        <XAxis type="number" hide />
                        <YAxis type="category" dataKey="name" width={90} tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: "#0e1529", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10, fontSize: 12 }} />
                        <Bar dataKey="value" fill="#34d399" radius={[0, 6, 6, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : <p className="text-xs text-slate-500">None.</p>}
                </div>
                <div className="card card-pad">
                  <div className="label mb-2">People seen</div>
                  {peopleData.length ? (
                    <ResponsiveContainer width="100%" height={Math.max(120, peopleData.length * 30)}>
                      <BarChart data={peopleData} layout="vertical">
                        <XAxis type="number" hide />
                        <YAxis type="category" dataKey="name" width={90} tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: "#0e1529", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10, fontSize: 12 }} />
                        <Bar dataKey="value" fill="#a78bfa" radius={[0, 6, 6, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : <p className="text-xs text-slate-500">None.</p>}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
