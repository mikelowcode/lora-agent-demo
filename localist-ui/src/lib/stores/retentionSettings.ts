/**
 * retentionSettings.ts — Data Retention settings store
 *
 * Two independent presets as of docs/architecture/
 * 20-episode-browsing-ui.md §20.12 — previously one shared preset
 * governed both chat_turns (hard delete) and episodes (soft-retract via
 * status='retracted') identically, found live to be a real problem: a
 * stable fact ("the user games on an Xbox Series X") was silently
 * retracted by the same 30-day window set for chat-history cleanup, even
 * though it was still true. Now:
 *   - eviction_preset         — chat_turns only. Default: null ("forever").
 *   - episode_eviction_preset — episodes only. Default: "forever" (always
 *     a concrete value, never null — episodic memory doesn't inherit
 *     chat-history's TTL).
 *
 *   GET /api/settings/retention — read both current presets
 *   PUT /api/settings/retention — set either or both independently;
 *     omitting a field leaves its current stored value untouched.
 *
 * This is a separate concern from chatHistory.ts (the live-session turn
 * store) and from the searchable chat_turns list — this file only ever
 * touches retention_settings.
 */

import { writable, type Writable } from 'svelte/store';
import { apiUrl } from '$lib/api';

export type EvictionPreset = '7d' | '30d' | '90d' | 'forever';

export interface RetentionSettingsState {
  eviction_preset: string | null;
  episode_eviction_preset: string;
}

export const retentionSettings: Writable<RetentionSettingsState> =
  writable({ eviction_preset: null, episode_eviction_preset: 'forever' });

export const retentionSettingsLoading: Writable<boolean> = writable(false);
export const retentionSettingsError: Writable<string | null> = writable(null);

export async function loadRetentionSettings(): Promise<void> {
  retentionSettingsLoading.set(true);
  retentionSettingsError.set(null);
  try {
    const res = await fetch(apiUrl('/api/settings/retention'));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: RetentionSettingsState = await res.json();
    retentionSettings.set(data);
  } catch (err) {
    retentionSettingsError.set(err instanceof Error ? err.message : String(err));
  } finally {
    retentionSettingsLoading.set(false);
  }
}

async function putRetention(body: Record<string, EvictionPreset>): Promise<void> {
  retentionSettingsLoading.set(true);
  retentionSettingsError.set(null);
  try {
    const res = await fetch(apiUrl('/api/settings/retention'), {
      method:  'PUT',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: RetentionSettingsState = await res.json();
    retentionSettings.set(data);
  } catch (err) {
    retentionSettingsError.set(err instanceof Error ? err.message : String(err));
  } finally {
    retentionSettingsLoading.set(false);
  }
}

/**
 * On failure, retentionSettings is left untouched — the control must
 * reflect the server's actual last-known state, not the user's pending
 * selection, when the write didn't land.
 */
export async function setRetentionPreset(preset: EvictionPreset): Promise<void> {
  await putRetention({ eviction_preset: preset });
}

/** Same failure posture as setRetentionPreset() — independent field, independent request. */
export async function setEpisodeRetentionPreset(preset: EvictionPreset): Promise<void> {
  await putRetention({ episode_eviction_preset: preset });
}
