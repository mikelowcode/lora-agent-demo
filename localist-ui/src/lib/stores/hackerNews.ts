/**
 * hackerNews.ts — Hacker News Live Feed panel store
 *
 * Two deliberately separate calls, same split as githubWatch.ts:
 *   GET  /api/hacker-news/top/preview — read-only, never calls Hacker News.
 *                                       Feeds the Live Feed panel's Hacker
 *                                       News block.
 *   POST /api/hacker-news/top/refresh — the Hacker News block's refresh
 *                                       link click handler. Always fetches
 *                                       a fresh top-10 feed and writes it
 *                                       to the backend's hacker_news_cache.
 *
 * fetchHackerNewsPreview() is safe to call on expand/idle — it has no side
 * effects. openHackerNews() is the only function that triggers real
 * Hacker News API calls.
 */

import { writable, type Writable } from 'svelte/store';
import { apiUrl } from '$lib/api';

export interface HackerNewsStory {
  key:    string;
  title:  string;
  url:    string;
  hn_url: string;
  score:  number | null;
  by:     string | null;
  error:  string | null;
}

export interface HackerNewsPreview {
  available:    boolean;
  generated_at: number | null;
  stories:      HackerNewsStory[];
}

export const hackerNewsPreview: Writable<HackerNewsPreview> =
  writable({ available: false, generated_at: null, stories: [] });
export const hackerNewsPreviewLoading: Writable<boolean> = writable(false);

export const hackerNewsOpening: Writable<boolean> = writable(false);
export const hackerNewsError: Writable<string | null> = writable(null);

export async function fetchHackerNewsPreview(): Promise<void> {
  hackerNewsPreviewLoading.set(true);
  try {
    const res = await fetch(apiUrl('/api/hacker-news/top/preview'));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: HackerNewsPreview = await res.json();
    hackerNewsPreview.set(data);
  } catch (err) {
    console.warn('Failed to load Hacker News preview:', err);
  } finally {
    hackerNewsPreviewLoading.set(false);
  }
}

/**
 * Triggers a fresh top-stories fetch. Returns whether it succeeded — the
 * caller is responsible for re-fetching the preview to reflect the new
 * content, same separation githubWatch.ts's openGithubWatch() already uses.
 */
export async function openHackerNews(): Promise<boolean> {
  hackerNewsOpening.set(true);
  hackerNewsError.set(null);
  try {
    const res = await fetch(apiUrl('/api/hacker-news/top/refresh'), { method: 'POST' });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    return true;
  } catch (err) {
    hackerNewsError.set(err instanceof Error ? err.message : String(err));
    return false;
  } finally {
    hackerNewsOpening.set(false);
  }
}
