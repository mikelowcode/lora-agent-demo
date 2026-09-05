<script lang="ts">
  import { onMount } from 'svelte';
  import {
    episodesStore,
    loadEpisodes,
    pendingCount,
    refreshPendingCount,
    allEpisodesTotal,
    refreshAllEpisodesTotal,
    approveEpisode,
    rejectEpisode,
    reactivateEpisode,
    EPISODE_TYPES,
    TYPE_LABELS,
    TYPE_COLORS,
    type EpisodeItem,
  } from '$lib/stores/episodes';
  import { assistantName } from '$lib/stores/assistantName';

  $: state     = $episodesStore;
  $: episodes  = state.episodes;
  $: loading   = state.loading;
  $: error     = state.error;

  let typeFilter = '';
  let statusFilter = 'active';

  // Type chips (All | Preference | ...) always operate within the "active"
  // status — their pre-existing meaning is unchanged. Pending is a separate
  // filter dimension (see applyPendingFilter) with its own toggle.
  function applyFilter(type: string) {
    typeFilter = type;
    statusFilter = 'active';
    loadEpisodes({ typeFilter: type, statusFilter: 'active', offset: 0 });
  }

  function applyPendingFilter() {
    typeFilter = '';
    statusFilter = 'pending';
    loadEpisodes({ typeFilter: '', statusFilter: 'pending', offset: 0 });
  }

  function applyRetractedFilter() {
    typeFilter = '';
    statusFilter = 'retracted';
    loadEpisodes({ typeFilter: '', statusFilter: 'retracted', offset: 0 });
  }

  function applySupersededFilter() {
    typeFilter = '';
    statusFilter = 'superseded';
    loadEpisodes({ typeFilter: '', statusFilter: 'superseded', offset: 0 });
  }

  // Distinct from the type-chip row's "All" button above (which means "all
  // *types*, active status only") — this is the one view that actually
  // shows every episode regardless of status, matching the allEpisodesTotal
  // count shown next to the panel title. Without this there was no way to
  // browse the full history in one place, only per-status subsets.
  function applyEverythingFilter() {
    typeFilter = '';
    statusFilter = 'all';
    loadEpisodes({ typeFilter: '', statusFilter: 'all', offset: 0, limit: 200 });
  }

  function formatDate(ts: number): string {
    return new Date(ts * 1000).toLocaleDateString([], {
      month: 'short', day: 'numeric', year: 'numeric',
    });
  }

  function formatConfidence(c: number): string {
    return Math.round(c * 100) + '%';
  }

  function chipStyle(type: string): string {
    const col = TYPE_COLORS[type] ?? TYPE_COLORS['context'];
    return `background:${col.bg};color:${col.color};border-color:${col.border};`;
  }

  // Per-card approve/reject/reactivate in-flight + error state, keyed by
  // episode id. Kept local rather than on episodesStore — a failed action
  // on one card must not disturb the rest of the list. `action` records
  // which button was clicked so only that button's label changes to the
  // "…ing" form while busy (all buttons are disabled either way).
  interface ActionState { busy: boolean; action: 'approve' | 'reject' | 'reactivate' | null; error: string | null; }
  let actionState: Record<number, ActionState> = {};

  async function runAction(
    ep: EpisodeItem,
    action: 'approve' | 'reject' | 'reactivate',
    call: (id: number) => Promise<boolean>,
  ) {
    actionState = { ...actionState, [ep.id]: { busy: true, action, error: null } };
    try {
      const updated = await call(ep.id);
      if (!updated) {
        actionState = {
          ...actionState,
          [ep.id]: { busy: false, action: null, error: `Could not ${action} — try refreshing.` },
        };
        return;
      }
      // Optimistic local removal — no longer pending, so if the Pending
      // filter is active it should disappear immediately rather than
      // waiting on a full loadEpisodes() round-trip.
      episodesStore.update((s) => ({
        ...s,
        episodes: s.episodes.filter((e) => e.id !== ep.id),
      }));
      refreshPendingCount();
    } catch (err) {
      actionState = {
        ...actionState,
        [ep.id]: {
          busy: false, action: null,
          error: err instanceof Error ? err.message : String(err),
        },
      };
    }
  }

  const handleApprove    = (ep: EpisodeItem) => runAction(ep, 'approve', approveEpisode);
  const handleReject     = (ep: EpisodeItem) => runAction(ep, 'reject', rejectEpisode);
  const handleReactivate = (ep: EpisodeItem) => runAction(ep, 'reactivate', reactivateEpisode);

  onMount(() => {
    loadEpisodes({ typeFilter: '' });
    refreshPendingCount();
    refreshAllEpisodesTotal();
  });
</script>

<div class="episodes-panel">
  <!-- Header -->
  <div class="panel-header">
    <h2 class="panel-title">
      Episodic Memory
      <span class="panel-total" title="Total episodes across all statuses">
        {$allEpisodesTotal} total
      </span>
    </h2>
    <button
      class="refresh-btn"
      on:click={() => loadEpisodes({ typeFilter, statusFilter, offset: 0 })}
      disabled={loading}
      aria-label="Refresh episodes"
      title="Refresh"
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round"
           class:spinning={loading}>
        <polyline points="23 4 23 10 17 10"/>
        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
      </svg>
    </button>
  </div>

  <!-- Type filters -->
  <div class="filter-row">
    <button
      class="filter-chip"
      class:active={statusFilter === 'active' && typeFilter === ''}
      on:click={() => applyFilter('')}
    >All</button>
    {#each EPISODE_TYPES as type}
      <button
        class="filter-chip"
        class:active={statusFilter === 'active' && typeFilter === type}
        style={statusFilter === 'active' && typeFilter === type ? chipStyle(type) : ''}
        on:click={() => applyFilter(type)}
      >{TYPE_LABELS[type]}</button>
    {/each}
    <span class="filter-separator" aria-hidden="true" />
    <button
      class="filter-chip filter-chip-pending"
      class:active={statusFilter === 'pending'}
      on:click={applyPendingFilter}
    >Pending ({$pendingCount})</button>
    <button
      class="filter-chip filter-chip-retracted"
      class:active={statusFilter === 'retracted'}
      on:click={applyRetractedFilter}
    >Retracted</button>
    <button
      class="filter-chip filter-chip-superseded"
      class:active={statusFilter === 'superseded'}
      on:click={applySupersededFilter}
    >Superseded</button>
    <button
      class="filter-chip filter-chip-everything"
      class:active={statusFilter === 'all'}
      on:click={applyEverythingFilter}
      title="Every episode regardless of status"
    >Everything ({$allEpisodesTotal})</button>
  </div>

  <!-- Content -->
  <div class="episode-list">
    {#if loading && episodes.length === 0}
      <div class="state-msg">
        <span class="dot dot-muted dot-pulse" />
        Loading…
      </div>

    {:else if error}
      <div class="state-msg state-error">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2" aria-hidden="true">
          <circle cx="12" cy="12" r="10"/>
          <line x1="15" y1="9" x2="9" y2="15"/>
          <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
        {error}
      </div>

    {:else if episodes.length === 0}
      <div class="state-msg">
        No episodes stored yet. {$assistantName.assistant_name} will learn from your conversations.
      </div>

    {:else}
      {#each episodes as ep (ep.id)}
        <div class="episode-card fade-in">
          <div class="episode-header">
            <span
              class="type-chip"
              style={chipStyle(ep.episode_type)}
            >{TYPE_LABELS[ep.episode_type] ?? ep.episode_type}</span>
            <span class="episode-subject">{ep.subject}</span>
            <span class="episode-date">{formatDate(ep.created_at)}</span>
          </div>
          <p class="episode-content">{ep.content}</p>
          <div class="episode-footer">
            <span class="meta-item" title="Confidence">
              ◈ {formatConfidence(ep.confidence)}
            </span>
            {#if ep.project_context}
              <span class="meta-item">
                ⬡ {ep.project_context}
              </span>
            {/if}
            <span class="meta-item meta-source" title={ep.source}>
              via {ep.source}
            </span>
          </div>
          {#if ep.status === 'pending'}
            <div class="episode-actions">
              <button
                class="episode-action-btn episode-action-approve"
                on:click={() => handleApprove(ep)}
                disabled={actionState[ep.id]?.busy}
              >{actionState[ep.id]?.action === 'approve' ? 'Approving…' : 'Approve'}</button>
              <button
                class="episode-action-btn episode-action-reject"
                on:click={() => handleReject(ep)}
                disabled={actionState[ep.id]?.busy}
              >{actionState[ep.id]?.action === 'reject' ? 'Rejecting…' : 'Reject'}</button>
            </div>
            {#if actionState[ep.id]?.error}
              <p class="episode-action-error">{actionState[ep.id]?.error}</p>
            {/if}
          {:else if ep.status === 'retracted'}
            <div class="episode-actions">
              <button
                class="episode-action-btn episode-action-approve"
                on:click={() => handleReactivate(ep)}
                disabled={actionState[ep.id]?.busy}
              >{actionState[ep.id]?.action === 'reactivate' ? 'Reactivating…' : 'Reactivate'}</button>
            </div>
            {#if actionState[ep.id]?.error}
              <p class="episode-action-error">{actionState[ep.id]?.error}</p>
            {/if}
          {/if}
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  .episodes-panel {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  /* Header */
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--sp-5) var(--sp-6) var(--sp-4);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .panel-title {
    font-size: var(--text-md);
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    display: flex;
    align-items: baseline;
    gap: var(--sp-2);
  }

  .panel-total {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    font-weight: 400;
    color: var(--text-muted);
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: var(--radius);
    background: transparent;
    color: var(--text-tertiary);
    transition: color var(--dur-fast) var(--ease),
                background var(--dur-fast) var(--ease);
  }
  .refresh-btn:hover:not(:disabled) {
    background: var(--bg-hover);
    color: var(--text-secondary);
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  .spinning { animation: spin 0.8s linear infinite; }

  /* Filter row */
  .filter-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-1);
    padding: var(--sp-3) var(--sp-6);
    border-bottom: 1px solid var(--border-soft);
    flex-shrink: 0;
  }

  .filter-chip {
    font-size: 11px;
    font-family: var(--font-mono);
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg-active);
    color: var(--text-tertiary);
    cursor: pointer;
    transition: all var(--dur-fast) var(--ease);
  }
  .filter-chip:hover { color: var(--text-secondary); }
  .filter-chip.active {
    color: #fff;
    border-color: var(--accent);
    background: var(--accent);
  }

  /* Vertical divider between the type-filter chips and the Pending toggle —
     they're different filter dimensions (type vs. status), not more items
     in the same row of choices. */
  .filter-separator {
    align-self: stretch;
    width: 1px;
    margin: 0 var(--sp-1);
    background: var(--border-soft);
  }

  /* Pending uses warning (amber) rather than accent — it's the one chip
     that means "needs your review", distinct from the neutral type chips. */
  .filter-chip-pending.active {
    color: var(--warning);
    border-color: var(--warning);
    background: var(--warning-dim);
  }

  /* Superseded uses the same neutral treatment as Retracted (default
     .filter-chip.active accent styling) — it's a historical/inactive
     status, not one that needs its own warning-style color. */

  /* Episode list */
  .episode-list {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: var(--sp-4) var(--sp-6);
    display: flex;
    flex-direction: column;
    gap: var(--sp-3);
  }

  .state-msg {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    font-size: var(--text-sm);
    color: var(--text-tertiary);
    padding: var(--sp-8) 0;
    justify-content: center;
  }
  .state-error { color: var(--error); }

  /* Episode card */
  .episode-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: var(--sp-3) var(--sp-4);
    display: flex;
    flex-direction: column;
    gap: var(--sp-2);
    transition: border-color var(--dur-fast) var(--ease);
  }
  .episode-card:hover { border-color: #3a3a3a; }

  .episode-header {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    flex-wrap: wrap;
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

  .episode-subject {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .episode-date {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--text-muted);
    flex-shrink: 0;
  }

  .episode-content {
    font-size: 12.5px;
    color: var(--text-secondary);
    line-height: 1.6;
    margin: 0;
  }

  .episode-footer {
    display: flex;
    align-items: center;
    gap: var(--sp-3);
    flex-wrap: wrap;
  }

  .meta-item {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--text-muted);
  }

  .meta-source {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 180px;
  }

  /* Pending-card approve/reject actions */
  .episode-actions {
    display: flex;
    gap: var(--sp-2);
  }

  .episode-action-btn {
    font-size: var(--text-xs);
    font-weight: 500;
    font-family: inherit;
    padding: var(--sp-1) var(--sp-3);
    border-radius: var(--radius);
    border: 1px solid transparent;
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease), opacity var(--dur-fast) var(--ease);
  }
  .episode-action-btn:disabled { opacity: 0.5; cursor: default; }

  .episode-action-approve {
    background: var(--success-dim);
    color: var(--success);
    border-color: var(--success-dim);
  }
  .episode-action-approve:hover:not(:disabled) { background: #3ecf8e33; }

  .episode-action-reject {
    background: var(--error-dim);
    color: var(--error);
    border-color: var(--error-dim);
  }
  .episode-action-reject:hover:not(:disabled) { background: #e0525233; }

  .episode-action-error {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--error);
    margin: 0;
  }
</style>
