/**
 * runtimeBackendSwitch.ts — live runtime-backend switch + chat-model pinning
 *
 * Talks to the three backend endpoints documented in
 * docs/architecture/16-runtime-backend-layer.md §16.5:
 *   POST /api/settings/runtime-backend                      — switchRuntimeBackend()
 *   GET  /api/settings/runtime-backend/{backend}/models      — fetchBackendModels()
 *   POST /api/settings/runtime-backend/{backend}/chat-model  — pinChatModel()
 *
 * Follows retentionSettings.ts's pattern: on failure, state is left
 * untouched (no optimistic update) — the UI must reflect the server's actual
 * last-known state, not a pending guess.
 */

import { writable, type Writable } from 'svelte/store';
import { modelConfig, type RuntimeBackend } from './model';
import { apiUrl } from '$lib/api';

export const runtimeBackendSwitchLoading: Writable<boolean> = writable(false);
export const runtimeBackendSwitchError: Writable<string | null> = writable(null);

export async function switchRuntimeBackend(
  backend: RuntimeBackend,
  chatModel?: string
): Promise<boolean> {
  runtimeBackendSwitchLoading.set(true);
  runtimeBackendSwitchError.set(null);
  try {
    const res = await fetch(apiUrl('/api/settings/runtime-backend'), {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ backend, chat_model: chatModel })
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    // Mirrors what a subsequent health poll would confirm anyway — updating
    // here just avoids a UI flicker while waiting for the next 15s tick.
    modelConfig.update((s) => ({ ...s, backend }));
    return true;
  } catch (err) {
    runtimeBackendSwitchError.set(err instanceof Error ? err.message : String(err));
    return false;
  } finally {
    runtimeBackendSwitchLoading.set(false);
  }
}

export async function fetchBackendModels(backend: RuntimeBackend): Promise<string[]> {
  try {
    const res = await fetch(apiUrl(`/api/settings/runtime-backend/${backend}/models`));
    if (!res.ok) return [];
    const data = await res.json();
    return data.models ?? [];
  } catch {
    return [];
  }
}

export async function pinChatModel(backend: RuntimeBackend, chatModel: string): Promise<boolean> {
  runtimeBackendSwitchLoading.set(true);
  runtimeBackendSwitchError.set(null);
  try {
    const res = await fetch(apiUrl(`/api/settings/runtime-backend/${backend}/chat-model`), {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ chat_model: chatModel })
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    return true;
  } catch (err) {
    runtimeBackendSwitchError.set(err instanceof Error ? err.message : String(err));
    return false;
  } finally {
    runtimeBackendSwitchLoading.set(false);
  }
}
