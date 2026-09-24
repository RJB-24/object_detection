import { useEffect, useState } from "react";
import { UserPlus, Trash2, Crosshair, Users } from "lucide-react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
} from "recharts";
import { api } from "../lib/api";
import type { Calibration } from "../lib/types";
import { PageHeader, EmptyState } from "../components/bits";
import { MultiDropzone } from "../components/Dropzone";
import { useToast } from "../components/Toast";

export function Faces() {
  const [name, setName] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [ids, setIds] = useState<string[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [cal, setCal] = useState<Calibration | null>(null);
  const [calBusy, setCalBusy] = useState(false);
  const { push } = useToast();

  const load = async () => {
    try {
      const d = await api.listFaces();
      setIds(d.identities);
      setCounts(d.counts);
    } catch (e: any) {
      push("error", e.message);
    }
  };
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const register = async () => {
    if (!name.trim()) return push("info", "Enter a name first.");
    if (!files.length) return push("info", "Select at least one face photo.");
    setBusy(true);
    try {
      const r = await api.registerFace(name.trim(), files);
      push("success", `Registered ${r.embeddings_added}/${r.images_received} photo(s) for "${r.name}".`);
      setName("");
      setFiles([]);
      load();
    } catch (e: any) {
      push("error", e.message);
    }
    setBusy(false);
  };

  const remove = async (n: string) => {
    if (!confirm(`Delete "${n}" from the gallery?`)) return;
    try {
      await api.deleteFace(n);
      push("success", `Deleted "${n}".`);
      load();
    } catch (e: any) {
      push("error", e.message);
    }
  };

  const calibrate = async () => {
    setCalBusy(true);
    try {
      const c = await api.calibrate();
      setCal(c);
      push(c.sufficient_data ? "success" : "info", c.guidance);
    } catch (e: any) {
      push("error", e.message);
    }
    setCalBusy(false);
  };

  const qualityBadge = (q: string) => {
    const map: Record<string, string> = {
      good: "bg-emerald-400/10 text-emerald-300 border border-emerald-400/30",
      mixed: "bg-amber-400/10 text-amber-300 border border-amber-400/30",
      poor: "bg-red-400/10 text-red-300 border border-red-400/30",
      "need-more-photos": "bg-white/5 text-slate-400 border border-white/10",
    };
    return <span className={`badge ${map[q] || map["need-more-photos"]}`}>{q.replaceAll("-", " ")}</span>;
  };

  return (
    <div>
      <PageHeader title="Face Gallery" sub="Enroll identities, then calibrate the match threshold against real gallery statistics" />
      <div className="grid xl:grid-cols-[380px_1fr] gap-3.5 items-start">
        <div className="flex flex-col gap-3.5">
          <div className="card card-pad flex flex-col gap-3">
            <div className="label flex items-center gap-1.5"><UserPlus size={14} /> Enroll identity</div>
            <input className="input" placeholder="Person name, e.g. Ada Lovelace" value={name} onChange={(e) => setName(e.target.value)} />
            <MultiDropzone files={files} onFiles={setFiles} accept="image/*" />
            <button className="btn-primary w-full" onClick={register} disabled={busy}>
              {busy ? "Enrolling…" : "Enroll face"}
            </button>
            <p className="text-[11px] text-slate-500 leading-relaxed">
              Use 3–5 varied photos per person (lighting, angle, expression). Quality beats quantity.
            </p>
          </div>
          <div className="card card-pad">
            <div className="label mb-2.5 flex items-center gap-1.5"><Users size={14} /> Identities ({ids.length})</div>
            {ids.length === 0 && <p className="text-xs text-slate-500">Gallery is empty.</p>}
            <div className="flex flex-col gap-1.5 max-h-[260px] overflow-y-auto pr-1">
              {ids.map((n) => (
                <div key={n} className="flex items-center gap-2 rounded-lg bg-white/[0.04] border border-white/10 px-3 py-2 text-sm">
                  <span className="font-medium truncate flex-1">{n}</span>
                  <span className="badge bg-white/5 border border-white/10">×{counts[n] || 0}</span>
                  <button onClick={() => remove(n)} className="text-slate-500 hover:text-red-300 transition" title={`Delete ${n}`}>
                    <Trash2 size={15} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="card card-pad">
          <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
            <span className="label flex items-center gap-1.5"><Crosshair size={14} /> Threshold calibration</span>
            <button className="btn-ghost !py-2 text-[13px]" onClick={calibrate} disabled={calBusy}>
              {calBusy ? "Calibrating…" : "Run calibration"}
            </button>
          </div>
          {!cal && (
            <EmptyState
              title="Not calibrated yet"
              sub="Calibration measures genuine vs impostor score distributions in your gallery and recommends the threshold with the best FAR/FRR trade-off. Needs 2+ identities with 2+ photos each."
            />
          )}
          {cal && (
            <div className="flex flex-col gap-4">
              <div className="grid sm:grid-cols-4 gap-2.5">
                <div className="rounded-xl bg-white/[0.04] border border-white/10 p-3">
                  <div className="label">Suggested</div>
                  <div className="text-2xl font-bold text-emerald-300 font-mono">{cal.suggested_threshold.toFixed(3)}</div>
                </div>
                <div className="rounded-xl bg-white/[0.04] border border-white/10 p-3">
                  <div className="label">EER point</div>
                  <div className="text-2xl font-bold font-mono">{cal.eer_threshold.toFixed(3)}</div>
                </div>
                <div className="rounded-xl bg-white/[0.04] border border-white/10 p-3">
                  <div className="label">Genuine pairs</div>
                  <div className="text-2xl font-bold font-mono">{cal.genuine_pairs}</div>
                  <div className="text-[11px] text-slate-500">mean {cal.genuine_mean ?? "—"}</div>
                </div>
                <div className="rounded-xl bg-white/[0.04] border border-white/10 p-3">
                  <div className="label">Impostor pairs</div>
                  <div className="text-2xl font-bold font-mono">{cal.impostor_pairs}</div>
                  <div className="text-[11px] text-slate-500">mean {cal.impostor_mean ?? "—"}</div>
                </div>
              </div>
              <p className="text-xs text-slate-400">{cal.guidance}</p>

              {cal.sufficient_data && (
                <div className="grid lg:grid-cols-2 gap-3.5">
                  <div>
                    <div className="label mb-2">FAR / FRR vs threshold</div>
                    <ResponsiveContainer width="100%" height={210}>
                      <LineChart data={cal.thresholds}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                        <XAxis dataKey="threshold" tick={{ fontSize: 10 }} minTickGap={30} />
                        <YAxis width={36} tick={{ fontSize: 10 }} />
                        <Tooltip contentStyle={{ background: "#0e1529", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10, fontSize: 12 }} />
                        <ReferenceLine x={cal.suggested_threshold} stroke="#34d399" strokeDasharray="4 4" />
                        <Line type="monotone" dataKey="far" name="FAR" stroke="#fb7185" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="frr" name="FRR" stroke="#22d3ee" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div>
                    <div className="label mb-2">ROC curve (TPR vs FPR)</div>
                    <ResponsiveContainer width="100%" height={210}>
                      <LineChart data={cal.roc}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                        <XAxis dataKey="fpr" name="FPR" tick={{ fontSize: 10 }} type="number" domain={[0, 1]} />
                        <YAxis tick={{ fontSize: 10 }} domain={[0, 1]} width={30} />
                        <Tooltip contentStyle={{ background: "#0e1529", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10, fontSize: 12 }} />
                        <Line type="monotone" dataKey="tpr" name="TPR" stroke="#a78bfa" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              <div>
                <div className="label mb-2">Per-identity enrollment quality</div>
                <div className="rounded-xl border border-white/10 overflow-hidden">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-[11px] uppercase tracking-wider text-slate-500 bg-white/[0.03]">
                        <th className="px-3 py-2 font-semibold">Identity</th>
                        <th className="px-3 py-2 font-semibold">Photos</th>
                        <th className="px-3 py-2 font-semibold">Mean intra-sim</th>
                        <th className="px-3 py-2 font-semibold">Min intra-sim</th>
                        <th className="px-3 py-2 font-semibold">Quality</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(cal.per_identity).map(([n, q]) => (
                        <tr key={n} className="table-row">
                          <td className="px-3 py-2 font-medium">{n}</td>
                          <td className="px-3 py-2 font-mono">{q.photos}</td>
                          <td className="px-3 py-2 font-mono">{q.mean_intra_similarity ?? "—"}</td>
                          <td className="px-3 py-2 font-mono">{q.min_intra_similarity ?? "—"}</td>
                          <td className="px-3 py-2">{qualityBadge(q.quality)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-[11px] text-slate-500 mt-2">
                  Low intra-similarity means inconsistent enrollment photos — replace blurry or occluded shots.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
