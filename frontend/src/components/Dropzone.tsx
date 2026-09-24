import { useCallback, useRef, useState } from "react";
import { ImagePlus, Film, X } from "lucide-react";
import clsx from "clsx";

export function Dropzone({ file, onFile, accept, kind }: {
  file: File | null;
  onFile: (f: File | null) => void;
  accept: string;
  kind: "image" | "video";
}) {
  const ref = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [url, setUrl] = useState<string | null>(null);

  const pick = useCallback((f: File | null) => {
    if (url) URL.revokeObjectURL(url);
    onFile(f);
    setUrl(f ? URL.createObjectURL(f) : null);
  }, [onFile, url]);

  return (
    <div
      onClick={() => ref.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0] || null); }}
      className={clsx(
        "relative rounded-xl border-2 border-dashed cursor-pointer overflow-hidden transition",
        drag ? "border-emerald-400 bg-emerald-400/5" : "border-cyan-400/30 bg-cyan-400/[0.03] hover:border-cyan-400/60",
        "min-h-[190px]",
        "grid place-items-center text-center p-4"
      )}
    >
      <input
        ref={ref} type="file" accept={accept} className="hidden"
        onChange={(e) => pick(e.target.files?.[0] || null)}
      />
      {file && url ? (
        <>
          {kind === "image" ? (
            <img src={url} alt="preview" className="absolute inset-0 w-full h-full object-contain bg-ink-950" />
          ) : (
            <video src={url} className="absolute inset-0 w-full h-full object-contain bg-ink-950" muted playsInline />
          )}
          <button
            onClick={(e) => { e.stopPropagation(); pick(null); }}
            className="absolute top-2 right-2 w-7 h-7 rounded-lg bg-ink-950/80 border border-white/15 grid place-items-center hover:border-red-400/60 hover:text-red-300 z-10"
          >
            <X size={14} />
          </button>
          <span className="absolute bottom-2 left-2 badge bg-ink-950/80 border border-white/15 z-10 max-w-[90%] truncate">
            {file.name}
          </span>
        </>
      ) : (
        <div>
          <div className="mx-auto w-11 h-11 rounded-xl bg-white/5 border border-white/10 grid place-items-center text-cyan-300 mb-2">
            {kind === "image" ? <ImagePlus size={20} /> : <Film size={20} />}
          </div>
          <p className="text-sm"><b>Drop a {kind} here</b> <span className="text-slate-500">or click to browse</span></p>
          <p className="text-[11px] text-slate-500 mt-1">{kind === "image" ? "JPG / PNG / WebP · max 10 MB" : "MP4 / AVI / MOV / MKV · max 100 MB"}</p>
        </div>
      )}
    </div>
  );
}

export function MultiDropzone({ files, onFiles, accept }: {
  files: File[]; onFiles: (f: File[]) => void; accept: string;
}) {
  const ref = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  return (
    <div
      onClick={() => ref.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); onFiles([...e.dataTransfer.files].slice(0, 10)); }}
      className={clsx(
        "rounded-xl border-2 border-dashed cursor-pointer p-4 text-center transition",
        drag ? "border-emerald-400 bg-emerald-400/5" : "border-cyan-400/30 bg-cyan-400/[0.03] hover:border-cyan-400/60"
      )}
    >
      <input
        ref={ref} type="file" accept={accept} multiple className="hidden"
        onChange={(e) => onFiles([...(e.target.files || [])].slice(0, 10))}
      />
      <p className="text-sm"><b>Drop 1–10 face photos</b> <span className="text-slate-500">or click to browse</span></p>
      <p className="text-[11px] text-slate-500 mt-1">
        {files.length ? `${files.length} file${files.length > 1 ? "s" : ""} selected` : "clear, front-facing photos work best"}
      </p>
      {!!files.length && (
        <div className="flex flex-wrap gap-1.5 justify-center mt-2">
          {files.map((f, i) => (
            <span key={i} className="badge bg-white/5 border border-white/10 max-w-[160px] truncate">{f.name}</span>
          ))}
        </div>
      )}
    </div>
  );
}
