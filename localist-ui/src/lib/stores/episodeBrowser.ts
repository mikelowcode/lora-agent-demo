/**
 * episodeBrowser.ts — Episode Browsing UI data layer
 *
 * Wraps the extended GET /api/chat/history (keyword/semantic search,
 * conversation/date-range/has-tool-result filters — see
 * MemoryManager.get_chat_turns() in memory_manager.py) and
 * GET /api/chat/history/conversations for the three-pane Episode Browsing
 * UI (Filters / Episode list / Episode detail).
 *
 * The old /history route (chatHistoryList.ts) is retired — this route is
 * a strict superset (same keyword search, plus semantic mode, filters,
 * detail pane, tool-result rendering) and its retention-policy section
 * duplicated a control that already lives on /settings. See
 * docs/architecture/12-chat-history-tab.md's retirement note.
 */

import { writable, get, type Writable } from 'svelte/store';
import { apiUrl } from '$lib/api';

export interface PendingDiff {
  page_name: string;
  diff: string;
  status: 'pending' | 'applied';
}

export interface EpisodeTurn {
  id: number;
  task_id: string;
  role: string;
  content: string;
  sources: Record<string, unknown>[];
  status_message: string | null;
  metadata: {
    chart?: {
      png_path: string;
      chart_config: {
        chart_type: 'bar' | 'line' | 'pie';
        title: string;
        labels: string[];
        datasets: { label: string; data: number[] }[];
      };
    };
    pending_diffs?: PendingDiff[];
    workflow_id?: string;
    workflow_steps?: {
      tool_name: string;
      parameters: string;
      success: boolean;
      result: string;
    }[];
    [key: string]: unknown;
  };
  conversation_id: string;
  conversation_title: string | null;
  created_at: number;
  score: number | null;
}

export interface ConversationSummary {
  conversation_id: string;
  conversation_title: string | null;
  last_created_at: number;
  first_created_at: number;
}

export type SearchMode = 'keyword' | 'semantic';

export interface EpisodeFilters {
  query: string;
  mode: SearchMode;
  conversationId: string | null;
  dateFrom: number | null; // unix seconds
  dateTo: number | null; // unix seconds
  hasToolResult: boolean;
}

export const PAGE_SIZE = 25;

export const episodeFilters: Writable<EpisodeFilters> = writable({
  query: '',
  mode: 'keyword',
  conversationId: null,
  dateFrom: null,
  dateTo: null,
  hasToolResult: false
});

export const episodeOffset: Writable<number> = writable(0);
export const episodeTurns: Writable<EpisodeTurn[]> = writable([]);
export const episodeTurnsTotal: Writable<number> = writable(0);
export const episodeTurnsLoading: Writable<boolean> = writable(false);
export const episodeTurnsError: Writable<string | null> = writable(null);

export const selectedEpisodeId: Writable<number | null> = writable(null);

export const conversations: Writable<ConversationSummary[]> = writable([]);
export const conversationsLoading: Writable<boolean> = writable(false);
export const conversationsError: Writable<string | null> = writable(null);

/** Call whenever a filter changes so the next load starts back at page 1. */
export function resetEpisodeOffset(): void {
  episodeOffset.set(0);
}

export async function loadEpisodeTurns(): Promise<void> {
  episodeTurnsLoading.set(true);
  episodeTurnsError.set(null);

  const f = get(episodeFilters);
  const offset = get(episodeOffset);

  const params = new URLSearchParams();
  if (f.query) params.set('q', f.query);
  if (f.query && f.mode === 'semantic') params.set('mode', 'semantic');
  if (f.conversationId) params.set('conversation_id', f.conversationId);
  if (f.dateFrom !== null) params.set('date_from', String(f.dateFrom));
  if (f.dateTo !== null) params.set('date_to', String(f.dateTo));
  if (f.hasToolResult) params.set('has_tool_result', 'true');
  params.set('limit', String(PAGE_SIZE));
  params.set('offset', String(offset));

  try {
    const res = await fetch(apiUrl(`/api/chat/history?${params}`));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: { turns: EpisodeTurn[]; total: number } = await res.json();
    episodeTurns.set(data.turns);
    episodeTurnsTotal.set(data.total);
  } catch (err) {
    episodeTurnsError.set(err instanceof Error ? err.message : String(err));
  } finally {
    episodeTurnsLoading.set(false);
  }
}

export async function loadConversationsList(): Promise<void> {
  conversationsLoading.set(true);
  conversationsError.set(null);
  try {
    const res = await fetch(apiUrl('/api/chat/history/conversations'));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: { conversations: ConversationSummary[] } = await res.json();
    conversations.set(data.conversations);
  } catch (err) {
    conversationsError.set(err instanceof Error ? err.message : String(err));
  } finally {
    conversationsLoading.set(false);
  }
}
