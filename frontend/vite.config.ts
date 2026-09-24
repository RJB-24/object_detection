import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// Production build drops straight into the FastAPI static dir so the
// whole platform ships as one container: `npm run build` -> serve via uvicorn.
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://localhost:8000", changeOrigin: true } },
  },
  build: {
    outDir: "../app/static/dist",
    emptyOutDir: true,
  },
});
