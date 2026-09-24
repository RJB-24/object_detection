import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  ScanSearch, Boxes, Users, Gauge, ArrowRight, Cpu, UserCheck,
  AlertTriangle, RefreshCw,
} from "lucide-react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, BarChart, Bar, PieChart, Pie, Cell,
} from "recharts";
import { api, fmtTime, shortModel } from "../lib/api";
import type { HealthResponse, HistoryRun, PlatformStats } from "../lib/types";
import { PageHeader, StatCard, Spinner, KindBadge } from "../components/bits";
import { useToast } from "../components/Toast";

const PIE_COLORS = ["#22d3ee", "#34d399", "#a78bfa", "#fbbf24"];

export function Dashboard() {
  const [stats, setStats] = useState<PlatformStats | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [recent, setRecent] = useState<HistoryRun[]>([]);
  const [loading, setLoading] = useState(true);
  const { push } = useToast();

  const load = async () => {
    setLoading(true);
    try {
      const [s, h, hist] = await Promise.all([api.stats(), api.health(), api.history(6)]);
      setStats(s);
      setHealth(h);
      setRecent(hist.runs);
    } catch (e: any) {
      push("error", e.message);
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (loading || !stats) {
    return (
      <div>
        <PageHeader title="Dashboard" sub="Platform overview" />
        <Spinner label="Loading platform stats…" />
      </div>
    );
  }

  const kindData = Object.entries(stats.by_kind).map(([name, value]) => ({ name, value }));
  const yoloErr = health?.models.yolo.error;
  const faceErr = health?.models.face.error;

  return (
    <div>
      <PageHeader
        title="Dashboard"
        sub="Live status, throughput and recent activity"
        right={
          <button className="btn-ghost !py-2" onClick={load}>
            <RefreshCw size={15} /> Refresh
          </button>
        }
      />

      {(yoloErr || faceErr) && (
        <div className="card card-pad !p-3.5 mb-4 flex items-start gap-2.5 text-sm border-l-4 !border-l-amber-400">
          <AlertTriangle size={18} className="text-amber-300 shrink-0 mt-0.5" />
          <div>
            <b>Model warning.</b>{" "}
            {yoloErr ? `YOLO: ${yoloErr} ` : ""}
            {faceErr ? `Faces: ${faceErr}` : ""}
            <span className="text-slate-400"> Run <code className="font-mono text-xs bg-white/10 px-1.5 py-0.5 rounded">python scripts/download_models.py</code> on the server.</span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3.5 mb-4">
        <StatCard icon={<ScanSearch size={20} />} label="Total runs" value={stats.total_runs} accent="bg-cyan-400/10 text-cyan-300" />
        <StatCard icon={<Boxes size={20} />} label="Objects detected" value={stats.total_objects} accent="bg-emerald-400/10 text-emerald-300" />
        <StatCard icon={<Users size={20} />} label="Faces · matched" value={stats.total_faces} sub={`${stats.total_matched} matched`} accent="bg-violet-400/10 text-violet-300" />
        <StatCard icon={<Gauge size={20} />} label="Avg latency" value={`${stats.avg_ms} ms`} accent="bg-amber-400/10 text-amber-300" />
      </div>

      <div className="grid xl:grid-cols-3 gap-3.5 mb-4">
        <div className="card card-pad xl:col-span-2">
          <div className="label mb-3">Latency trend · last {stats.trend.length} runs (ms)</div>
          {stats.trend.length ? (
            <ResponsiveContainer width="100%" height={210}>
              <LineChart data={stats.trend}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                <XAxis dataKey="id" tick={false} />
                <YAxis width={44} />
                <Tooltip
                  contentStyle={{ background: "#0e1529", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10, fontSize: 12 }}
                  labelFormatter={(v) => `run #${v}`}
                />
                <Line type="monotone" dataKey="ms" stroke="#22d3ee" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-slate-500 py-8 text-center">No runs yet — analyze an image to populate this chart.</p>
          )}
        </div>
        <div className="card card-pad">
          <div className="label mb-3">Runs by task</div>
          {kindData.length ? (
            <ResponsiveContainer width="100%" height={210}>
              <PieChart>
                <Pie data={kindData} dataKey="value" nameKey="name" innerRadius={52} outerRadius={80} paddingAngle={3}>
                  {kindData.map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ background: "#0e1529", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10, fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-slate-500 py-8 text-center">No data yet.</p>
          )}
          <div className="flex gap-3 justify-center text-xs text-slate-400">
            {kindData.map((k, i) => (
              <span key={k.name} className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: PIE_COLORS[i % PIE_COLORS.length] }} />
                {k.name} ({k.value})
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="grid xl:grid-cols-3 gap-3.5">
        <div className="card card-pad xl:col-span-2">
          <div className="flex items-center justify-between mb-3">
            <span className="label">Recent runs</span>
            <Link to="/history" className="text-xs text-cyan-300 hover:underline flex items-center gap-1">
              View all <ArrowRight size={13} />
            </Link>
          </div>
          {recent.length === 0 && <p className="text-sm text-slate-500">No runs recorded yet.</p>}
          <div className="flex flex-col">
            {recent.map((r) => (
              <div key={r.id} className="table-row flex items-center gap-3 py-2.5">
                {r.thumb ? (
                  <img src={`/api/history/${r.id}/thumb`} alt="" className="w-14 h-10 object-cover rounded-lg border border-white/10" />
                ) : (
                  <div className="w-14 h-10 rounded-lg bg-white/5 border border-white/10" />
                )}
                <KindBadge kind={r.kind} />
                <span className="text-xs text-slate-400">#{r.id}</span>
                <span className="text-xs ml-auto text-slate-400">
                  {r.objects} obj · {r.faces} faces · {Math.round(r.ms)} ms
                </span>
                <span className="text-xs text-slate-500 hidden sm:inline">{fmtTime(r.ts)}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="flex flex-col gap-3.5">
          <div className="card card-pad">
            <div className="label mb-3">Active models</div>
            <div className="flex flex-col gap-2.5 text-sm">
              <div className="flex items-center gap-2.5">
                <Cpu size={16} className="text-cyan-300" />
                <span className="font-mono text-[13px]">{shortModel(stats.active_model)}</span>
                <span className="badge bg-emerald-400/10 text-emerald-300 border border-emerald-400/30 ml-auto">objects</span>
              </div>
              <div className="flex items-center gap-2.5">
                <UserCheck size={16} className="text-violet-300" />
                <span className="font-mono text-[13px]">YuNet + SFace</span>
                <span className="badge bg-emerald-400/10 text-emerald-300 border border-emerald-400/30 ml-auto">
                  {stats.faces.identities.length} identities
                </span>
              </div>
            </div>
            <Link to="/finetune" className="btn-ghost w-full mt-4 !py-2 text-[13px]">
              Manage models <ArrowRight size={14} />
            </Link>
          </div>
          <div className="card card-pad">
            <div className="label mb-2.5">Detections per run</div>
            <ResponsiveContainer width="100%" height={130}>
              <BarChart data={stats.trend.slice(-12)}>
                <XAxis dataKey="id" tick={false} />
                <Tooltip
                  contentStyle={{ background: "#0e1529", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10, fontSize: 12 }}
                  labelFormatter={(v) => `run #${v}`}
                />
                <Bar dataKey="objects" stackId="a" fill="#34d399" />
                <Bar dataKey="faces" stackId="a" fill="#a78bfa" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
