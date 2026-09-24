import { useEffect, useState } from "react";
import { Trash2, Ban } from "lucide-react";
import clsx from "clsx";
import { api, fmtTime } from "../lib/api";
import type { HistoryRun } from "../lib/types";
import { PageHeader, Spinner, KindBadge, EmptyState } from "../components/bits";
import { useToast } from "../components/Toast";

const FILTERS = ["all", "analyze", "objects", "faces"];

export function History() {
  const [runs, setRuns] = useState<HistoryRun[]>([]);
  const [total, setTotal] = useState(0);
  const [kind, setKind] = useState("all");
  const [loading, setLoading] = useState(true);
  const { push } = useToast();

  const load = async (k = kind) => {
    setLoading(true);
    try {
      const d = await api.history(100, 0, k === "all" ? undefined : k);
      setRuns(d.runs);
      setTotal(d.total);
    } catch (e: any) {
      push("error", e.message);
    }
    setLoading(false);
  };

  useEffect(() => {
    load(kind);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind]);

  const del = async (id: number) => {
    try {
      await api.deleteRun(id);
      setRuns((r) => r.filter((x) => x.id !== id));
      setTotal((t) => t - 1);
    } catch (e: any) {
      push("error", e.message);
    }
  };

  const clear = async () => {
    if (!confirm("Delete ALL run history?")) return;
    try {
      await api.clearHistory();
      setRuns([]);
      setTotal(0);
      push("success", "History cleared.");
    } catch (e: any) {
      push("error", e.message);
    }
  };

  return (
    <div>
      <PageHeader
        title="Run History"
        sub={`${total} runs recorded`}
        right={
          <div className="flex items-center gap-2">
            <div className="flex gap-1 bg-white/5 border border-white/10 rounded-xl p-1">
              {FILTERS.map((f) => (
                <button
                  key={f}
                  onClick={() => setKind(f)}
                  className={clsx(
                    "px-3 py-1.5 rounded-lg text-xs font-semibold capitalize transition",
                    kind === f ? "bg-cyan-400/20 text-white" : "text-slate-400 hover:text-slate-200"
                  )}
                >
                  {f}
                </button>
              ))}
            </div>
            <button className="btn-danger !py-2 text-[13px]" onClick={clear}>
              <Ban size={14} /> Clear
            </button>
          </div>
        }
      />
      <div className="card card-pad !p-3">
        {loading ? (
          <div className="p-4"><Spinner label="Loading history…" /></div>
        ) : runs.length === 0 ? (
          <EmptyState title="No runs yet" sub="Analyze an image and it will be logged here automatically with a thumbnail." />
        ) : (
          <div className="rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[11px] uppercase tracking-wider text-slate-500">
                  <th className="px-3 py-2 font-semibold">Preview</th>
                  <th className="px-3 py-2 font-semibold">Run</th>
                  <th className="px-3 py-2 font-semibold">Task</th>
                  <th className="px-3 py-2 font-semibold">Objects</th>
                  <th className="px-3 py-2 font-semibold">Faces</th>
                  <th className="px-3 py-2 font-semibold">Matched</th>
                  <th className="px-3 py-2 font-semibold">Latency</th>
                  <th className="px-3 py-2 font-semibold">Time</th>
                  <th className="px-3 py-2 font-semibold"></th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.id} className="table-row">
                    <td className="px-3 py-2">
                      {r.thumb ? (
                        <img src={`/api/history/${r.id}/thumb`} alt="" className="w-[72px] h-[46px] object-cover rounded-lg border border-white/10" />
                      ) : (
                        <div className="w-[72px] h-[46px] rounded-lg bg-white/5 border border-white/10" />
                      )}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">#{r.id}</td>
                    <td className="px-3 py-2"><KindBadge kind={r.kind} /></td>
                    <td className="px-3 py-2 font-mono">{r.objects}</td>
                    <td className="px-3 py-2 font-mono">{r.faces}</td>
                    <td className="px-3 py-2 font-mono">{r.matched}</td>
                    <td className="px-3 py-2 font-mono">{Math.round(r.ms)} ms</td>
                    <td className="px-3 py-2 text-xs text-slate-400 whitespace-nowrap">{fmtTime(r.ts)}</td>
                    <td className="px-3 py-2">
                      <button onClick={() => del(r.id)} className="text-slate-500 hover:text-red-300 transition" title="Delete run">
                        <Trash2 size={15} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
