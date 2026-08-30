/**
 * assistantName.ts — Assistant Name settings store
 *
 * Owns the user-configurable assistant name (backend/memory_manager.py's
 * assistant_settings table, default "Localist"):
 *   GET /api/settings/assistant-name — read the current name
 *   PUT /api/settings/assistant-name — set it
 *
 * Follows retentionSettings.ts's pattern exactly: on a failed write, state
 * is left untouched (no optimistic update) — the UI must reflect the
 * server's actual last-known state, not a pending guess.
 *
 * Loaded once at app startup (+layout.svelte's onMount) so every chat-UI
 * component that displays the assistant's name reads the same reactive
 * store instead of each doing its own fetch.
 */

import { writable, type Writable } from 'svelte/store';
import { apiUrl } from '$lib/api';

export interface AssistantNameState {
  assistant_name: string;
}

const DEFAULT_STATE: AssistantNameState = { assistant_name: 'Localist' };

export const assistantName: Writable<AssistantNameState> = writable(DEFAULT_STATE);
export const assistantNameLoading: Writable<boolean> = writable(false);
export const assistantNameError: Writable<string | null> = writable(null);

export async function loadAssistantName(): Promise<void> {
  assistantNameLoading.set(true);
  assistantNameError.set(null);
  try {
    const res = await fetch(apiUrl('/api/settings/assistant-name'));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: AssistantNameState = await res.json();
    assistantName.set(data);
  } catch (err) {
    assistantNameError.set(err instanceof Error ? err.message : String(err));
  } finally {
    assistantNameLoading.set(false);
  }
}

/**
 * On failure, assistantName is left untouched — the control must reflect
 * the server's actual last-known state, not the user's pending edit, when
 * the write didn't land.
 */
export async function setAssistantName(name: string): Promise<boolean> {
  assistantNameLoading.set(true);
  assistantNameError.set(null);
  try {
    const res = await fetch(apiUrl('/api/settings/assistant-name'), {
      method:  'PUT',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ assistant_name: name }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    const data: AssistantNameState = await res.json();
    assistantName.set(data);
    return true;
  } catch (err) {
    assistantNameError.set(err instanceof Error ? err.message : String(err));
    return false;
  } finally {
    assistantNameLoading.set(false);
  }
}
