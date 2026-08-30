/**
 * newsPreferences.ts — Daily News Brief preferences store
 *
 * Owns news_preferences (docs/daily-news-brief-plan.md §4/§5):
 *   GET /api/news/preferences — read current home_country/local_query/topics
 *   PUT /api/news/preferences — set them
 *
 * Follows retentionSettings.ts's pattern exactly: on a failed write, state
 * is left untouched (no optimistic update) — the UI must reflect the
 * server's actual last-known state, not a pending guess.
 */

import { writable, type Writable } from 'svelte/store';
import { apiUrl } from '$lib/api';

export interface NewsPreferencesState {
  home_country: string;
  local_query:  string | null;
  topics:       string[];
  topic_pool:   Record<string, string>; // topic key -> display label
}

const DEFAULT_STATE: NewsPreferencesState = {
  home_country: 'us',
  local_query:  null,
  topics:       [],
  topic_pool:   {},
};

export const newsPreferences: Writable<NewsPreferencesState> = writable(DEFAULT_STATE);
export const newsPreferencesLoading: Writable<boolean> = writable(false);
export const newsPreferencesError: Writable<string | null> = writable(null);

export async function loadNewsPreferences(): Promise<void> {
  newsPreferencesLoading.set(true);
  newsPreferencesError.set(null);
  try {
    const res = await fetch(apiUrl('/api/news/preferences'));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: NewsPreferencesState = await res.json();
    newsPreferences.set(data);
  } catch (err) {
    newsPreferencesError.set(err instanceof Error ? err.message : String(err));
  } finally {
    newsPreferencesLoading.set(false);
  }
}

export async function setNewsPreferences(
  home_country: string,
  local_query:  string | null,
  topics:       string[]
): Promise<boolean> {
  newsPreferencesLoading.set(true);
  newsPreferencesError.set(null);
  try {
    const res = await fetch(apiUrl('/api/news/preferences'), {
      method:  'PUT',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ home_country, local_query, topics }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    const data: NewsPreferencesState = await res.json();
    newsPreferences.set(data);
    return true;
  } catch (err) {
    newsPreferencesError.set(err instanceof Error ? err.message : String(err));
    return false;
  } finally {
    newsPreferencesLoading.set(false);
  }
}
