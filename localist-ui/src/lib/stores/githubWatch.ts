/**
 * githubWatch.ts — GitHub Watch Feed Live Feed panel store
 *
 * Two deliberately separate calls, same split as newsBrief.ts:
 *   GET  /api/github/watch/preview — read-only, never calls GitHub. Feeds
 *                                    the Live Feed panel's GitHub block.
 *   POST /api/github/watch/refresh — the GitHub block's refresh link click
 *                                    handler. Always fetches a fresh feed
 *                                    (every watched repo's latest release)
 *                                    and writes it to the backend's
 *                                    github_watch_cache.
 *
 * fetchGithubWatchPreview() is safe to call on expand/idle — it has no
 * side effects. openGithubWatch() is the only function that triggers real
 * GitHub API calls.
 */

import { writable, type Writable } from 'svelte/store';
import { apiUrl } from '$lib/api';

export interface GithubWatchRelease {
  tag_name:     string;
  name:         string;
  published_at: string;
  html_url:     string;
}

export interface GithubWatchRepo {
  key:            string;
  label:          string;
  repo_url:       string;
  latest_release: GithubWatchRelease | null;
  error:          string | null;
  source:         'watched' | 'pinned';
}

export interface GithubWatchPreview {
  available:    boolean;
  generated_at: number | null;
  repos:        GithubWatchRepo[];
}

export const githubWatchPreview: Writable<GithubWatchPreview> =
  writable({ available: false, generated_at: null, repos: [] });
export const githubWatchPreviewLoading: Writable<boolean> = writable(false);

export const githubWatchOpening: Writable<boolean> = writable(false);
export const githubWatchError: Writable<string | null> = writable(null);

export async function fetchGithubWatchPreview(): Promise<void> {
  githubWatchPreviewLoading.set(true);
  try {
    const res = await fetch(apiUrl('/api/github/watch/preview'));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: GithubWatchPreview = await res.json();
    githubWatchPreview.set(data);
  } catch (err) {
    console.warn('Failed to load GitHub watch feed preview:', err);
  } finally {
    githubWatchPreviewLoading.set(false);
  }
}

/**
 * Triggers a fresh watch-feed fetch. Returns whether it succeeded — the
 * caller is responsible for re-fetching the preview to reflect the new
 * content, same separation newsBrief.ts's openNewsBrief() already uses.
 */
export async function openGithubWatch(): Promise<boolean> {
  githubWatchOpening.set(true);
  githubWatchError.set(null);
  try {
    const res = await fetch(apiUrl('/api/github/watch/refresh'), { method: 'POST' });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    return true;
  } catch (err) {
    githubWatchError.set(err instanceof Error ? err.message : String(err));
    return false;
  } finally {
    githubWatchOpening.set(false);
  }
}
