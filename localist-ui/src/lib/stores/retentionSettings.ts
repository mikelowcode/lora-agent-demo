/**
 * retentionSettings.ts — Data Retention settings store
 *
 * Owns the global retention-preset setting, shared by chat_turns (hard
 * delete) and episodes (soft-retract via status='retracted') once the
 * backend's startup sweep runs:
 *   GET /api/settings/retention — read the current preset
 *   PUT /api/settings/retention — set the preset
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
}

export const retentionSettings: Writable<RetentionSettingsState> =
  writable({ eviction_preset: null });

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

/**
 * On failure, retentionSettings is left untouched — the control must
 * reflect the server's actual last-known state, not the user's pending
 * selection, when the write didn't land.
 */
export async function setRetentionPreset(preset: EvictionPreset): Promise<void> {
  retentionSettingsLoading.set(true);
  retentionSettingsError.set(null);
  try {
    const res = await fetch(apiUrl('/api/settings/retention'), {
      method:  'PUT',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ eviction_preset: preset }),
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
