/**
 * confirmDialog.ts — a confirm() that actually works inside the packaged
 * Tauri app, not just the browser dev flow.
 *
 * Found live (2026-09-05): the packaged .app's WKWebView does not implement
 * window.confirm()/alert()/prompt() — calling confirm() there returns
 * immediately without ever showing a dialog, so every `if (!confirm(...))
 * return;` gate silently no-oped (confirmed by injecting a startup
 * confirm() into app.html and observing the page render straight through
 * it with no dialog and no block). Real browsers (the :5173 dev flow)
 * implement window.confirm() natively and don't need this.
 *
 * `isTauri()` (from @tauri-apps/api/core) distinguishes the two contexts at
 * runtime — same source serves both — routing to the Tauri dialog plugin's
 * confirm() (a real native dialog, permissioned via
 * src-tauri/capabilities/default.json's dialog:allow-confirm) only when
 * actually running inside Tauri.
 */

import { isTauri } from '@tauri-apps/api/core';
import { confirm as tauriConfirm } from '@tauri-apps/plugin-dialog';

export async function confirmDialog(message: string, title = 'Localist'): Promise<boolean> {
  if (isTauri()) {
    return await tauriConfirm(message, { title });
  }
  return window.confirm(message);
}
