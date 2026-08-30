import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	kit: {
		// SPA mode (fallback: 'index.html') — every route here is client-fetch
		// driven with no server load anywhere in the app (see +layout.ts's
		// ssr = false), and conversation/[id] is a dynamic route that can't be
		// prerendered at build time. Static output is what Tauri's webview
		// needs to load directly (no Node server process); adapter-auto never
		// picked this by default outside a recognized hosting platform.
		adapter: adapter({ fallback: 'index.html' })
	}
};

export default config;
