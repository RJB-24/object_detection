# VisionAI web dashboard

React 18 + TypeScript + Vite + Tailwind CSS + Recharts + React Router.

```bash
npm install
npm run dev      # hot-reload on :5173, /api proxied to localhost:8000
npm run build    # production bundle -> ../app/static/dist (served by FastAPI)
```

No separate deploy needed: the Dockerfile builds this and the FastAPI service
hosts the result at `/` (with client-route fallback).
