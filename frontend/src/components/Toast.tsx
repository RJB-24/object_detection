import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { CheckCircle2, AlertCircle, Info, X } from "lucide-react";
import clsx from "clsx";

type Kind = "success" | "error" | "info";
interface Toast { id: number; kind: Kind; text: string }

const Ctx = createContext<{ push: (kind: Kind, text: string) => void }>({ push: () => {} });
export const useToast = () => useContext(Ctx);

let nextId = 1;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const push = useCallback((kind: Kind, text: string) => {
    const id = nextId++;
    setToasts((t) => [...t, { id, kind, text }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4200);
  }, []);
  const dismiss = (id: number) => setToasts((t) => t.filter((x) => x.id !== id));

  return (
    <Ctx.Provider value={{ push }}>
      {children}
      <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 w-[340px]">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={clsx(
              "card card-pad !p-3.5 flex items-start gap-2.5 text-sm shadow-card border-l-4",
              t.kind === "success" && "!border-l-emerald-400",
              t.kind === "error" && "!border-l-red-400",
              t.kind === "info" && "!border-l-cyan-400"
            )}
          >
            {t.kind === "success" ? (
              <CheckCircle2 size={18} className="text-emerald-400 shrink-0 mt-0.5" />
            ) : t.kind === "error" ? (
              <AlertCircle size={18} className="text-red-400 shrink-0 mt-0.5" />
            ) : (
              <Info size={18} className="text-cyan-400 shrink-0 mt-0.5" />
            )}
            <span className="flex-1">{t.text}</span>
            <button onClick={() => dismiss(t.id)} className="text-slate-500 hover:text-slate-300">
              <X size={15} />
            </button>
          </div>
        ))}
      </div>
    </Ctx.Provider>
  );
}
