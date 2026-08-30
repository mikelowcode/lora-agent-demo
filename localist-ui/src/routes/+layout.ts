// Every route in this app is client-fetch driven against the FastAPI
// backend — no server load anywhere. Disabling SSR lets adapter-static
// build a real SPA (fallback: 'index.html' in svelte.config.js) instead of
// requiring every route to be prerenderable, which conversation/[id]
// (a dynamic route) isn't.
export const ssr = false;
