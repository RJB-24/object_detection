export const fmtTime = (ts: number) => {
  const d = new Date(ts * 1000);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

export const shortModel = (m: string) => m.split("/").pop() || m;
