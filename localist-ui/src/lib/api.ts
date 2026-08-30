// Every backend call in this app is written as a literal relative
// '/api/...' path. In dev, vite's server.proxy rule (vite.config.ts)
// matches '/api' and rewrites it away before forwarding to the FastAPI
// backend on 127.0.0.1:8001 — the backend's real routes have no '/api'
// prefix at all (e.g. /task, /settings/assistant-name).
//
// That proxy only exists in the vite dev server. A statically-served
// build (adapter-static output, e.g. loaded by Tauri's webview) has no
// proxy — every fetch would 404 against its own static-file origin. This
// module is the one place that knows how to turn a '/api/...' literal
// into a real, reachable URL, so call sites never need to know or care
// which mode they're running in.
//
// 'tauri' mode is a dedicated Vite build mode (`vite build --mode
// tauri`), not the default `vite dev`/`vite build` — those keep behaving
// exactly as before this module existed.
const TAURI_BACKEND_ORIGIN = 'http://127.0.0.1:8001';

export function apiUrl(path: string): string {
  if (import.meta.env.MODE === 'tauri') {
    return TAURI_BACKEND_ORIGIN + path.replace(/^\/api/, '');
  }
  return path;
}
