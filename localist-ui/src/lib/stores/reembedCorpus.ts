/**
 * reembedCorpus.ts — manual wiki/raw corpus re-embed
 *
 * Talks to POST /memory/reembed (docs/architecture/16-runtime-backend-layer.md
 * §16.4), the explicit counterpart to episodes' automatic startup re-embed.
 *
 * Follows runtimeBackendSwitch.ts's pattern: on failure, state is left
 * untouched (no optimistic update) — the UI must reflect the server's actual
 * last-known state, not a pending guess.
 */

import { writable, type Writable } from 'svelte/store';
import { apiUrl } from '$lib/api';

export interface ReembedCorpusResponse {
  reembedded: number;
  total:      number;
  model:      string | null;
}

export const reembedLoading: Writable<boolean> = writable(false);
export const reembedError: Writable<string | null> = writable(null);

export async function reembedCorpus(): Promise<ReembedCorpusResponse | null> {
  reembedLoading.set(true);
  reembedError.set(null);
  try {
    const res = await fetch(apiUrl('/api/memory/reembed'), { method: 'POST' });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    return (await res.json()) as ReembedCorpusResponse;
  } catch (err) {
    reembedError.set(err instanceof Error ? err.message : String(err));
    return null;
  } finally {
    reembedLoading.set(false);
  }
}
