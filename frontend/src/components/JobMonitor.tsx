import { useEffect, useRef, useState } from "react";
import { CheckCircle2, XCircle, Loader2 } from "lucide-react";
import { api } from "../lib/api";
import type { Job } from "../lib/types";

export function useJobPoll(jobId: string | null, intervalMs = 1500) {
  const [job, setJob] = useState<Job | null>(null);
  useEffect(() => {
    if (!jobId) {
      setJob(null);
      return;
    }
    let alive = true;
    const poll = async () => {
      try {
        const j = await api.job(jobId);
        if (alive) {
          setJob(j);
          if (j.status === "done" || j.status === "error") return false;
        }
      } catch {
        /* keep polling */
      }
      return alive;
    };
    poll();
    const t = setInterval(async () => {
      const cont = await poll();
      if (!cont) clearInterval(t);
    }, intervalMs);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [jobId, intervalMs]);
  return job;
}

export function JobProgress({ job }: { job: Job }) {
  const pct = Math.round(job.progress * 100);
  const done = job.status === "done";
  const err = job.status === "error";
  return (
    <div>
      <div className="flex items-center justify-between text-sm mb-1.5">
        <span className="flex items-center gap-2 font-medium">
          {done ? (
            <CheckCircle2 size={16} className="text-emerald-400" />
          ) : err ? (
            <XCircle size={16} className="text-red-400" />
          ) : (
            <Loader2 size={16} className="animate-spin text-cyan-400" />
          )}
          {job.message}
        </span>
        <span className="font-mono text-xs text-slate-400">{pct}%</span>
      </div>
      <div className="h-2 rounded-full bg-white/10 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${err ? "bg-red-400" : done ? "bg-emerald-400" : "bg-gradient-to-r from-cyan-400 to-emerald-400"}`}
          style={{ width: `${err ? 100 : pct}%` }}
        />
      </div>
      {err && job.error && (
        <p className="text-xs text-red-300 mt-2 font-mono break-words">{job.error}</p>
      )}
    </div>
  );
}

export function JobLogs({ job, maxH }: { job: Job; maxH?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [job.logs.length]);
  return (
    <div ref={ref} className={`log-view bg-ink-950/80 border border-white/10 rounded-xl p-3 overflow-y-auto ${maxH || "max-h-[220px]"}`}>
      {job.logs.length === 0 && <span className="text-slate-600">waiting for output…</span>}
      {job.logs.map((l, i) => (
        <div key={i} className="whitespace-pre-wrap break-words text-slate-300">{l}</div>
      ))}
    </div>
  );
}
