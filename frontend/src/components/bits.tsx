import type { ReactNode } from "react";
import { Loader2, Inbox } from "lucide-react";
import clsx from "clsx";

export function PageHeader({ title, sub, right }: { title: string; sub?: string; right?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3 mb-5 flex-wrap">
      <div>
        <h1 className="text-[22px] font-bold tracking-tight">{title}</h1>
        {sub && <p className="text-sm text-slate-400 mt-0.5">{sub}</p>}
      </div>
      {right && <div className="flex items-center gap-2">{right}</div>}
    </div>
  );
}

export function StatCard({ icon, label, value, sub, accent }: {
  icon: ReactNode; label: string; value: string | number; sub?: string; accent?: string;
}) {
  return (
    <div className="card card-pad flex items-center gap-3.5">
      <div className={clsx("w-11 h-11 rounded-xl grid place-items-center shrink-0", accent || "bg-cyan-400/10 text-cyan-300")}>
        {icon}
      </div>
      <div className="min-w-0">
        <div className="text-[26px] font-bold leading-none tracking-tight">{value}</div>
        <div className="text-xs text-slate-400 mt-1">{label}</div>
        {sub && <div className="text-[11px] text-slate-500">{sub}</div>}
      </div>
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-400">
      <Loader2 size={16} className="animate-spin text-cyan-400" /> {label || "Loading…"}
    </div>
  );
}

export function EmptyState({ title, sub }: { title: string; sub?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-10 text-center">
      <div className="w-12 h-12 rounded-2xl bg-white/5 border border-white/10 grid place-items-center text-slate-500 mb-3">
        <Inbox size={22} />
      </div>
      <div className="font-semibold text-sm">{title}</div>
      {sub && <div className="text-xs text-slate-500 mt-1 max-w-[300px]">{sub}</div>}
    </div>
  );
}

export function Slider({ label, value, min, max, step, onChange, format }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; format?: (v: number) => string;
}) {
  return (
    <label className="block">
      <div className="flex justify-between items-baseline mb-1">
        <span className="label">{label}</span>
        <span className="text-sm font-semibold font-mono">{format ? format(value) : value.toFixed(2)}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

export function KindBadge({ kind }: { kind: string }) {
  const map: Record<string, string> = {
    analyze: "bg-cyan-400/10 text-cyan-300 border border-cyan-400/30",
    objects: "bg-emerald-400/10 text-emerald-300 border border-emerald-400/30",
    faces: "bg-violet-400/10 text-violet-300 border border-violet-400/30",
  };
  return <span className={`badge ${map[kind] || "bg-white/5 text-slate-300 border border-white/10"}`}>{kind}</span>;
}
