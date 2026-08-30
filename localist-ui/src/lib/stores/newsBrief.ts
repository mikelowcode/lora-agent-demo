/**
 * newsBrief.ts — Daily News Brief Live Feed panel store
 *
 * Two deliberately separate calls (docs/daily-news-brief-plan.md §6/§12):
 *   GET  /api/news/brief/preview — read-only, never calls NewsAPI. Feeds
 *                                  the Live Feed panel's News block.
 *   POST /api/news/brief/open    — the "Daily News Brief Refresh" link's
 *                                  click handler. Always fetches a fresh
 *                                  brief and writes it to the backend's
 *                                  news_brief_cache — deliberately not
 *                                  idempotent within a day, since a
 *                                  link literally labeled "Refresh" should
 *                                  never silently reuse stale content.
 *                                  Only populates the Live Feed panel; it no
 *                                  longer opens or seeds a Chat Conversation
 *                                  (removed 2026-07-24 — redundant once the
 *                                  per-article "Ask about this" button let
 *                                  users chat about one story at a time
 *                                  instead of getting the whole brief dumped
 *                                  into a chat turn as noise).
 *
 * fetchPreview() is safe to call on expand/idle — it has no side effects.
 * openBrief() is the only function that triggers real NewsAPI calls.
 */

import { writable, type Writable } from 'svelte/store';
import { apiUrl } from '$lib/api';

export interface NewsBriefArticle {
  title:        string;
  description:  string;
  source:       string;
  published_at: string;
  url:          string;
}

export interface NewsBriefSection {
  key:      string;
  label:    string;
  articles: NewsBriefArticle[];
  error:    string | null;
}

export interface NewsBriefPreview {
  available:  boolean;
  brief_date: string | null;
  sections:   NewsBriefSection[];
}

export const newsBriefPreview: Writable<NewsBriefPreview> =
  writable({ available: false, brief_date: null, sections: [] });
export const newsBriefPreviewLoading: Writable<boolean> = writable(false);

export const newsBriefOpening: Writable<boolean> = writable(false);
export const newsBriefError: Writable<string | null> = writable(null);

export async function fetchNewsBriefPreview(): Promise<void> {
  newsBriefPreviewLoading.set(true);
  try {
    const res = await fetch(apiUrl('/api/news/brief/preview'));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: NewsBriefPreview = await res.json();
    newsBriefPreview.set(data);
  } catch (err) {
    console.warn('Failed to load news brief preview:', err);
  } finally {
    newsBriefPreviewLoading.set(false);
  }
}

/**
 * Triggers a fresh brief generation. Returns whether it succeeded — the
 * caller is responsible for re-fetching the preview to reflect the new
 * content, same separation reembedCorpus.ts/runtimeBackendSwitch.ts
 * already use.
 */
export async function openNewsBrief(): Promise<boolean> {
  newsBriefOpening.set(true);
  newsBriefError.set(null);
  try {
    const res = await fetch(apiUrl('/api/news/brief/open'), { method: 'POST' });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    return true;
  } catch (err) {
    newsBriefError.set(err instanceof Error ? err.message : String(err));
    return false;
  } finally {
    newsBriefOpening.set(false);
  }
}
