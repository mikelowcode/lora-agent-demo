/**
 * githubWatchPins.ts — pinned GitHub repos store
 *
 * Owns pinned_github_repos:
 *   GET /api/github/watch/pinned-repos — read the user's pinned "owner/repo"
 *                                        slugs
 *   PUT /api/github/watch/pinned-repos — full-list replace
 *
 * Separate from githubWatch.ts on purpose, same split as
 * newsBrief.ts/newsPreferences.ts: this store owns the user-editable
 * settings list, githubWatch.ts owns the read-only Live Feed preview/refresh
 * pair. A pinned repo is release-tracked independent of GitHub's
 * Watch/subscriptions relationship — saving here does not itself refresh
 * the feed (main.py's PUT handler documents the same "takes effect on next
 * generation" posture as PUT /news/preferences); the user triggers that via
 * the existing GitHub Watch Feed refresh action.
 *
 * Follows newsPreferences.ts's pattern exactly: on a failed write, state is
 * left untouched (no optimistic update).
 */

import { writable, type Writable } from 'svelte/store';

export interface GithubWatchPinsState {
  repos: string[];
}

const DEFAULT_STATE: GithubWatchPinsState = { repos: [] };

export const githubWatchPins: Writable<GithubWatchPinsState> = writable(DEFAULT_STATE);
export const githubWatchPinsLoading: Writable<boolean> = writable(false);
export const githubWatchPinsError: Writable<string | null> = writable(null);

export async function loadGithubWatchPins(): Promise<void> {
  githubWatchPinsLoading.set(true);
  githubWatchPinsError.set(null);
  try {
    const res = await fetch('/api/github/watch/pinned-repos');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: GithubWatchPinsState = await res.json();
    githubWatchPins.set(data);
  } catch (err) {
    githubWatchPinsError.set(err instanceof Error ? err.message : String(err));
  } finally {
    githubWatchPinsLoading.set(false);
  }
}

export async function setGithubWatchPins(repos: string[]): Promise<boolean> {
  githubWatchPinsLoading.set(true);
  githubWatchPinsError.set(null);
  try {
    const res = await fetch('/api/github/watch/pinned-repos', {
      method:  'PUT',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ repos }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    const data: GithubWatchPinsState = await res.json();
    githubWatchPins.set(data);
    return true;
  } catch (err) {
    githubWatchPinsError.set(err instanceof Error ? err.message : String(err));
    return false;
  } finally {
    githubWatchPinsLoading.set(false);
  }
}
