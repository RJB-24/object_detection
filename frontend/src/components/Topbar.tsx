import { useEffect, useState } from "react";
import { Activity, Cpu, UserCheck } from "lucide-react";
import { api } from "../lib/api";
import { shortModel } from "../lib/format";
import type { HealthResponse } from "../lib/types";

export function Topbar() {
  const [h, setH] = useState<HealthResponse | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const d = await api.health();
        if (alive) setH(d);
      } catch {
        if (alive) setH(null);
      }
    };
    poll();
    const t = setInterval(poll, 10000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  const yoloOk = h && !h.models.yolo.error;
  const faceOk = h && (h.models.face.loaded || (h.models.face.det_exists && h.models.face.rec_exists));

  return (
    <header className="flex items-center gap-2.5 flex-wrap">
      <span className={`badge ${h ? "bg-emerald-400/10 text-emerald-300 border border-emerald-400/30" : "bg-red-400/10 text-red-300 border border-red-400/30"}`}>
        <Activity size={13} /> {h ? `Online · v${h.version}` : "Offline"}
      </span>
      {h && (
        <>
          <span className={`badge ${yoloOk ? "bg-cyan-400/10 text-cyan-300 border border-cyan-400/30" : "bg-amber-400/10 text-amber-300 border border-amber-400/30"}`}>
            <Cpu size={13} /> {shortModel(h.models.yolo.model)}
            {h.models.yolo.loaded ? "" : " · lazy"}
          </span>
          <span className={`badge ${faceOk ? "bg-violet-400/10 text-violet-300 border border-violet-400/30" : "bg-amber-400/10 text-amber-300 border border-amber-400/30"}`}>
            <UserCheck size={13} /> Faces · {h.face_identities.length} identities
          </span>
        </>
      )}
    </header>
  );
}
