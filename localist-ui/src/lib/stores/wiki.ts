/**
 * wiki.ts — review-then-apply wiki diff actions
 *
 * Owns the single client action for the diff-review UI: applyDiff() posts
 * a previously-proposed diff (surfaced via a chat turn's
 * metadata.pending_diffs, see tasks.ts's PendingDiff) to
 * POST /wiki/apply-diff. Discard has no backend counterpart by design —
 * it's a pure client-side "stop rendering this block" action handled
 * entirely in ChatPanel.svelte.
 */

import { apiUrl } from '$lib/api';

export interface ApplyDiffResult {
  success: boolean;
  error?: string;
}

/**
 * Apply a proposed diff to disk via POST /wiki/apply-diff.
 *
 * task_id identifies the chat turn the diff came from, so the backend can
 * mark that turn's persisted pending_diffs entry "applied" (surviving a
 * page reload) in addition to writing the page itself. page_name/diff
 * round-trip back exactly as originally proposed — content-based matching
 * server-side is what detects staleness (page edited since proposal) and
 * fails cleanly instead of corrupting the page.
 */
export async function applyDiff(
  task_id: string,
  page_name: string,
  diff: string
): Promise<ApplyDiffResult> {
  try {
    const res = await fetch(apiUrl('/api/wiki/apply-diff'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id, page_name, diff })
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      return {
        success: false,
        error: (body as { detail?: string }).detail ?? `HTTP ${res.status}`
      };
    }

    return { success: true };
  } catch (err) {
    return { success: false, error: err instanceof Error ? err.message : String(err) };
  }
}
