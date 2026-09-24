import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import { Menu, X, LayoutDashboard, ScanSearch, Clapperboard, Users, SlidersHorizontal, History as HistoryIcon } from "lucide-react";
import { useState } from "react";
import clsx from "clsx";
import { Sidebar } from "./components/Sidebar";
import { Topbar } from "./components/Topbar";
import { ToastProvider } from "./components/Toast";
import { Dashboard } from "./pages/Dashboard";
import { ImageAnalysis } from "./pages/ImageAnalysis";
import { VideoAnalysis } from "./pages/VideoAnalysis";
import { Faces } from "./pages/Faces";
import { FineTune } from "./pages/FineTune";
import { History } from "./pages/History";

const MOBILE_NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/analyze", label: "Images", icon: ScanSearch },
  { to: "/video", label: "Video", icon: Clapperboard },
  { to: "/faces", label: "Faces", icon: Users },
  { to: "/finetune", label: "Fine-tune", icon: SlidersHorizontal },
  { to: "/history", label: "History", icon: HistoryIcon },
];

function MobileNav() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  return (
    <div className="md:hidden">
      <button onClick={() => setOpen(!open)} className="btn-ghost !p-2.5">
        {open ? <X size={18} /> : <Menu size={18} />}
      </button>
      {open && (
        <div className="absolute left-4 right-4 top-[68px] z-40 card card-pad !p-2 flex flex-col gap-1">
          {MOBILE_NAV.map((n) => (
            <Link
              key={n.to}
              to={n.to}
              onClick={() => setOpen(false)}
              className={clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium",
                loc.pathname === n.to ? "bg-cyan-400/15 text-white" : "text-slate-400"
              )}
            >
              <n.icon size={17} /> {n.label}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

function Shell() {
  return (
    <div className="min-h-screen flex">
      <Sidebar />
      <div className="flex-1 min-w-0">
        <div className="sticky top-0 z-30 bg-ink-950/85 backdrop-blur border-b border-white/10 px-4 sm:px-7 py-3.5 flex items-center gap-3">
          <MobileNav />
          <span className="md:hidden font-extrabold">VisionAI</span>
          <div className="ml-auto">
            <Topbar />
          </div>
        </div>
        <main className="px-4 sm:px-7 py-6 max-w-[1280px] mx-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analyze" element={<ImageAnalysis />} />
            <Route path="/video" element={<VideoAnalysis />} />
            <Route path="/faces" element={<Faces />} />
            <Route path="/finetune" element={<FineTune />} />
            <Route path="/history" element={<History />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
          <footer className="text-center text-[11px] text-slate-600 pt-8 pb-4">
            VisionAI 2.0 · YOLOv8 + YuNet + SFace · FastAPI + React
          </footer>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <Shell />
      </ToastProvider>
    </BrowserRouter>
  );
}
