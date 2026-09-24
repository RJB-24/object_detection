import { NavLink } from "react-router-dom";
import {
  LayoutDashboard, ScanSearch, Clapperboard, Users, SlidersHorizontal,
  History, BookOpenText, ScanEye,
} from "lucide-react";
import clsx from "clsx";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/analyze", label: "Image Analysis", icon: ScanSearch },
  { to: "/video", label: "Video Analysis", icon: Clapperboard },
  { to: "/faces", label: "Face Gallery", icon: Users },
  { to: "/finetune", label: "Fine-tune", icon: SlidersHorizontal },
  { to: "/history", label: "Run History", icon: History },
];

export function Sidebar() {
  return (
    <aside className="w-[228px] shrink-0 hidden md:flex flex-col border-r border-white/10 bg-ink-900/60 backdrop-blur min-h-screen sticky top-0 h-screen">
      <div className="flex items-center gap-2.5 px-5 pt-6 pb-5">
        <div className="w-9 h-9 rounded-xl grid place-items-center bg-gradient-to-br from-cyan-400 to-emerald-400 text-ink-950">
          <ScanEye size={20} strokeWidth={2.5} />
        </div>
        <div>
          <div className="font-extrabold tracking-tight text-[17px]">VisionAI</div>
          <div className="text-[11px] text-slate-500 -mt-0.5">Visual Intelligence</div>
        </div>
      </div>
      <nav className="flex flex-col gap-1 px-3">
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.to === "/"}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition",
                isActive
                  ? "bg-gradient-to-r from-cyan-400/15 to-emerald-400/10 text-white border border-cyan-400/25"
                  : "text-slate-400 hover:text-slate-100 hover:bg-white/5 border border-transparent"
              )
            }
          >
            <n.icon size={18} />
            {n.label}
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto p-4">
        <a
          href="/docs"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm text-slate-400 hover:text-slate-100 hover:bg-white/5 transition"
        >
          <BookOpenText size={18} /> API Reference
        </a>
        <div className="mt-2 mx-1 p-3 rounded-xl bg-white/[0.04] border border-white/10 text-[11px] text-slate-500 leading-relaxed">
          YOLOv8 · YuNet · SFace
          <br />
          OpenCV DNN + PyTorch
        </div>
      </div>
    </aside>
  );
}
