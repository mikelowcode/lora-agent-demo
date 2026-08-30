<script lang="ts">
  // Read-only "related memory" overlay for the Episode Browsing UI's
  // detail pane (episode-browsing-ui-plan.md Phase 6) — active episodes
  // semantically related to the selected chat_turns row's own content, via
  // GET /memory/episodes/related (backend: EpisodicMemoryReader.by_similarity(),
  // Mode 3 §2.6 — real cosine similarity where an embedding exists,
  // keyword/Jaccard fallback otherwise). Replaces an earlier task_id
  // exact-match query that read as "no related memory" for the common
  // case, since episodes are sparse by design (§2.1) and most turns
  // produce zero episodes of their own. Purely an annotation surface: no
  // approve/reject here (that stays EpisodesPanel.svelte's job on the
  // existing /memory route) — episodes surfaced here are always
  // status=active already.
  //
  // The caller wraps this component in a {#key selected.task_id} block
  // (see EpisodeDetailPane.svelte) so a new instance — and a fresh
  // onMount() fetch — is created on every turn selection, rather than this
  // component needing its own taskId-change-detection logic.
  import { onMount } from 'svelte';
  import { TYPE_LABELS, TYPE_COLORS } from '$lib/stores/episodes';
  import { apiUrl } from '$lib/api';

  export let taskId: string;
  export let content: string;

  interface RelatedEpisode {
    id: number;
    episode_type: string;
    subject: string;
    content: string;
    confidence: number;
  }

  let episodes: RelatedEpisode[] = [];
  let loading = false;
  let error: string | null = null;

  onMount(async () => {
    if (!content) return;
    loading = true;
    try {
      const res = await fetch(
        apiUrl(`/api/memory/episodes/related?${new URLSearchParams({ content, task_id: taskId, limit: '10' })}`)
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: { episodes: RelatedEpisode[] } = await res.json();
      episodes = data.episodes;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  });

  function chipStyle(type: string): string {
    const col = TYPE_COLORS[type] ?? TYPE_COLORS['context'];
    return `background:${col.bg};color:${col.color};border-color:${col.border};`;
  }
</script>

{#if loading}
  <p class="annotations-state">Loading related memory…</p>
{:else if error}
  <p class="annotations-state annotations-error">{error}</p>
{:else if episodes.length > 0}
  <div class="annotations-list">
    {#each episodes as ep (ep.id)}
      <div class="annotation-card">
        <span class="type-chip" style={chipStyle(ep.episode_type)}>
          {TYPE_LABELS[ep.episode_type] ?? ep.episode_type}
        </span>
        <span class="annotation-content">{ep.content}</span>
      </div>
    {/each}
  </div>
{:else}
  <p class="annotations-state">No related memory for this episode.</p>
{/if}

<style>
  .annotations-state {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    margin: 0;
  }
  .annotations-error { color: var(--error); }

  .annotations-list {
    display: flex;
    flex-direction: column;
    gap: var(--sp-2);
  }

  .annotation-card {
    display: flex;
    align-items: baseline;
    gap: var(--sp-2);
    padding: var(--sp-2) var(--sp-3);
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }

  .type-chip {
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 2px 7px;
    border-radius: 999px;
    border: 1px solid transparent;
    font-weight: 500;
    letter-spacing: 0.03em;
    flex-shrink: 0;
  }

  .annotation-content {
    font-size: var(--text-xs);
    color: var(--text-secondary);
    line-height: 1.5;
  }
</style>
