/**
 * episodes.ts — Localist episodic memory store
 *
 * Fetches from GET /api/memory/episodes and exposes paginated,
 * filterable episode state to EpisodesPanel.
 */

import { writable, derived } from 'svelte/store';
import { apiUrl } from '$lib/api';

export interface EpisodeItem {
  id:              number;
  episode_type:    string;
  subject:         string;
  content:         string;
  confidence:      number;
  source:          string;
  task_id:         string | null;
  project_context: string | null;
  status:          string;
  created_at:      number;
  last_accessed:   number | null;
}

export interface EpisodesState {
  episodes:     EpisodeItem[];
  loading:      boolean;
  error:        string | null;
  offset:       number;
  limit:        number;
  total:        number;
  typeFilter:   string;   // "" = all types
  statusFilter: string;   // "active" | "pending" | ... — see main.py's GET /memory/episodes
}

const _initial: EpisodesState = {
  episodes:     [],
  loading:      false,
  error:        null,
  offset:       0,
  limit:        50,
  total:        0,
  typeFilter:   '',
  statusFilter: 'active',
};

export const episodesStore = writable<EpisodesState>(_initial);

export const EPISODE_TYPES = [
  'preference',
  'correction',
  'decision',
  'workflow',
  'fact',
  'relationship',
  'context',
] as const;

// Human-readable labels for episode types
export const TYPE_LABELS: Record<string, string> = {
  preference:   'Preference',
  correction:   'Correction',
  decision:     'Decision',
  workflow:     'Workflow',
  fact:         'Fact',
  relationship: 'Relationship',
  context:      'Context',
};

// Colour tokens per episode type — preference/decision/workflow/correction
// map to the four semantic accent tokens per the design handoff; fact/
// relationship/context aren't specified there, so they keep their existing
// bespoke hues (unaffected by the light/dark token swap — a pre-existing
// tradeoff, not introduced by this pass).
export const TYPE_COLORS: Record<string, { bg: string; color: string; border: string }> = {
  preference:   { bg: 'var(--accent-dim)',  color: 'var(--accent)',  border: 'var(--accent-mid)' },
  decision:     { bg: 'var(--success-dim)', color: 'var(--success)', border: 'var(--success)' },
  workflow:     { bg: 'var(--warning-dim)', color: 'var(--warning)', border: 'var(--warning)' },
  correction:   { bg: 'var(--error-dim)',   color: 'var(--error)',   border: 'var(--error)' },
  fact:         { bg: '#2a1a2e', color: '#b07ecf', border: '#4a2d5a' },
  relationship: { bg: '#1e2a2a', color: '#7ecfcf', border: '#2d4a4a' },
  context:      { bg: '#1e1e1e', color: '#9a9a9a', border: '#2a2a2a' },
};

export async function loadEpisodes(opts: {
  typeFilter?: string;
  statusFilter?: string;
  offset?: number;
  limit?: number;
} = {}): Promise<void> {
  episodesStore.update((s) => ({ ...s, loading: true, error: null }));

  const statusFilter = opts.statusFilter ?? 'active';
  const params = new URLSearchParams();
  params.set('status', statusFilter);
  params.set('limit',  String(opts.limit  ?? 50));
  params.set('offset', String(opts.offset ?? 0));
  if (opts.typeFilter) params.set('episode_type', opts.typeFilter);

  try {
    const res = await fetch(apiUrl(`/api/memory/episodes?${params}`));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: { episodes: EpisodeItem[]; total: number; offset: number; limit: number }
      = await res.json();

    episodesStore.update((s) => ({
      ...s,
      loading:      false,
      episodes:     data.episodes,
      total:        data.total,
      offset:       data.offset,
      limit:        data.limit,
      statusFilter,
    }));
  } catch (err) {
    episodesStore.update((s) => ({
      ...s,
      loading: false,
      error:   err instanceof Error ? err.message : String(err),
    }));
  }
}

export function resetEpisodes(): void {
  episodesStore.set(_initial);
}

// ---------------------------------------------------------------------------
// Pending count — independent of episodesStore's currently-applied filter.
// Feeds both the Memory tab's own "Pending (N)" chip and the Sidebar badge,
// so it must stay decoupled from whatever filter the tab currently shows.
// ---------------------------------------------------------------------------

export const pendingCount = writable<number>(0);

export async function refreshPendingCount(): Promise<void> {
  const params = new URLSearchParams();
  params.set('status', 'pending');
  params.set('limit', '1');   // only `total` is needed, not the row data

  try {
    const res = await fetch(apiUrl(`/api/memory/episodes?${params}`));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: { total: number } = await res.json();
    pendingCount.set(data.total);
  } catch {
    // Non-fatal — a badge count isn't worth its own error UI. Leave the
    // previous value in place rather than resetting to 0 on a transient
    // network blip.
  }
}

// ---------------------------------------------------------------------------
// All-statuses total — independent of episodesStore's currently-applied
// filter, same reasoning as pendingCount above. Lets the panel show e.g.
// "68 episodes total" regardless of which status filter is selected, so a
// user looking at "Active (3)" can see the rest of their history hasn't
// gone missing — it's just filtered out of the current view.
// ---------------------------------------------------------------------------

export const allEpisodesTotal = writable<number>(0);

export async function refreshAllEpisodesTotal(): Promise<void> {
  const params = new URLSearchParams();
  params.set('status', 'all');
  params.set('limit', '1');   // only `total` is needed, not the row data

  try {
    const res = await fetch(apiUrl(`/api/memory/episodes?${params}`));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: { total: number } = await res.json();
    allEpisodesTotal.set(data.total);
  } catch {
    // Non-fatal — see refreshPendingCount()'s identical rationale above.
  }
}

// ---------------------------------------------------------------------------
// Approve / reject — write-approval gate actions
// ---------------------------------------------------------------------------

export async function approveEpisode(id: number): Promise<boolean> {
  const res = await fetch(apiUrl(`/api/memory/episodes/${id}/approve`), { method: 'POST' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data: { episode_id: number; status: string; updated: boolean } = await res.json();
  return data.updated;
}

export async function rejectEpisode(id: number): Promise<boolean> {
  const res = await fetch(apiUrl(`/api/memory/episodes/${id}/reject`), { method: 'POST' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data: { episode_id: number; status: string; updated: boolean } = await res.json();
  return data.updated;
}

export async function reactivateEpisode(id: number): Promise<boolean> {
  const res = await fetch(apiUrl(`/api/memory/episodes/${id}/reactivate`), { method: 'POST' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data: { episode_id: number; status: string; updated: boolean } = await res.json();
  return data.updated;
}
