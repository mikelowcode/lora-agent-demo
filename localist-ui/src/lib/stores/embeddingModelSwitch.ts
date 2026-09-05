/**
 * embeddingModelSwitch.ts — live embedding-model switch
 *
 * Talks to POST /api/settings/embedding-model (docs/architecture/
 * 16-runtime-backend-layer.md §16.4/§16.5) — sets or clears which model the
 * active runtime backend's own embed() uses (tier 1 of
 * _configure_embedding_source()'s three-tier precedence). Distinct from
 * runtimeBackendSwitch.ts's pinChatModel()/switchRuntimeBackend(): those
 * deliberately never touch the embedding source; this is the dedicated,
 * intentional path for changing it.
 *
 * Follows runtimeBackendSwitch.ts's pattern: on failure, state is left
 * untouched (no optimistic update) — the UI must reflect the server's actual
 * last-known state, not a pending guess.
 */

import { writable, type Writable } from 'svelte/store';
import { apiUrl } from '$lib/api';

export const embeddingModelSwitchLoading: Writable<boolean> = writable(false);
export const embeddingModelSwitchError: Writable<string | null> = writable(null);

export async function setEmbeddingModel(model: string): Promise<boolean> {
  embeddingModelSwitchLoading.set(true);
  embeddingModelSwitchError.set(null);
  try {
    const res = await fetch(apiUrl('/api/settings/embedding-model'), {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ model })
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    return true;
  } catch (err) {
    embeddingModelSwitchError.set(err instanceof Error ? err.message : String(err));
    return false;
  } finally {
    embeddingModelSwitchLoading.set(false);
  }
}
