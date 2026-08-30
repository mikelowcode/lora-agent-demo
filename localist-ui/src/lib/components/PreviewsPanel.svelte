<script lang="ts">
  import { onMount } from 'svelte';
  import { get } from 'svelte/store';
  import { goto } from '$app/navigation';
  import { previewsPanelCollapsed, togglePreviewsPanel } from '$lib/stores/previewsPanel';
  import {
    newsBriefPreview,
    fetchNewsBriefPreview,
    newsBriefOpening,
    newsBriefError,
    openNewsBrief,
    type NewsBriefArticle
  } from '$lib/stores/newsBrief';
  import {
    githubWatchPreview,
    fetchGithubWatchPreview,
    githubWatchOpening,
    githubWatchError,
    openGithubWatch,
    type GithubWatchRepo
  } from '$lib/stores/githubWatch';
  import {
    hackerNewsPreview,
    fetchHackerNewsPreview,
    hackerNewsOpening,
    hackerNewsError,
    openHackerNews,
    type HackerNewsStory
  } from '$lib/stores/hackerNews';
  import { previewBlocksCollapsed, togglePreviewBlock } from '$lib/stores/previewBlocks';
  import { tasksStore, submitTask } from '$lib/stores/tasks';
  import { chatHistoryStore } from '$lib/stores/chatHistory';
  import { currentConversationId, isFirstTurnOfConversation } from '$lib/stores/conversation';

  // Fetched once on mount regardless of collapsed state, so the preview is
  // already populated the moment the user expands the tab — this replaces
  // StatusBar's old hover-triggered fetch (docs/daily-news-brief-plan.md §8),
  // which only had room to show one truncated line per section.
  onMount(() => {
    void fetchNewsBriefPreview();
    void fetchGithubWatchPreview();
    void fetchHackerNewsPreview();
  });

  async function handleOpenBrief(): Promise<void> {
    const ok = await openNewsBrief();
    if (ok) {
      // /open already refreshed the backend's news_brief_cache by the time it
      // responds, but this panel's own store is only ever populated once on
      // mount — without this, the Live Feed list stays stale (showing
      // whatever was cached at page-load) until a full browser reload.
      // Refresh stays on the Live Feed panel — it no longer opens a Chat
      // Conversation (see newsBrief.ts).
      await fetchNewsBriefPreview();
    }
  }

  // Same pattern as handleOpenBrief — /refresh already wrote a fresh
  // github_watch_cache by the time it responds, but this panel's own store
  // is only ever populated once on mount, so it needs an explicit re-fetch.
  async function handleOpenGithubWatch(): Promise<void> {
    const ok = await openGithubWatch();
    if (ok) {
      await fetchGithubWatchPreview();
    }
  }

  // Same pattern as handleOpenGithubWatch — /refresh already wrote a fresh
  // hacker_news_cache by the time it responds, but this panel's own store
  // is only ever populated once on mount, so it needs an explicit re-fetch.
  async function handleOpenHackerNews(): Promise<void> {
    const ok = await openHackerNews();
    if (ok) {
      await fetchHackerNewsPreview();
    }
  }

  // Sends just one clicked article into the currently open conversation,
  // mirroring ChatPanel.svelte's handleSubmit (optimistic turns + submitTask)
  // rather than files.ts's ingestFile pattern, since the user is expected to
  // watch this stream in live rather than land on it after the fact.
  async function handleAskAboutArticle(article: NewsBriefArticle): Promise<void> {
    if ($tasksStore.finalizing) return;

    const instruction = `Tell me more about this news story: "${article.title}"`;
    const task_id = crypto.randomUUID();
    const now = Date.now();
    const conversationId = get(currentConversationId);

    let conversationTitle: string | undefined;
    if (get(isFirstTurnOfConversation)) {
      conversationTitle =
        article.title.length > 60 ? article.title.slice(0, 60) + '…' : article.title;
      isFirstTurnOfConversation.set(false);
    }

    chatHistoryStore.update((turns) => [
      ...turns,
      { role: 'user', content: instruction, task_id, timestamp: now },
      {
        role: 'assistant',
        content: '',
        task_id,
        timestamp: now + 1,
        status: 'planning',
        status_message: 'Planning…',
        sources: []
      }
    ]);

    await submitTask(
      instruction,
      { web_search_queries: [article.title], news_article_url: article.url },
      task_id,
      conversationId,
      conversationTitle
    );
    await goto(`/conversation/${conversationId}`);
  }

  // Same pattern as handleAskAboutArticle — the instruction names "Hacker
  // News" explicitly so Planner's P3-hacker-news gate fires (it runs ahead
  // of P3-news precisely because "hacker news" contains the bare word
  // "news", see planner.py), and hn_story_url pins hacker_news_search to
  // this one already-known story rather than trusting the title text alone
  // to find it again among similarly-titled submissions.
  async function handleAskAboutHackerNewsStory(story: HackerNewsStory): Promise<void> {
    if ($tasksStore.finalizing) return;

    const instruction = `Tell me more about this Hacker News story: "${story.title}"`;
    const task_id = crypto.randomUUID();
    const now = Date.now();
    const conversationId = get(currentConversationId);

    let conversationTitle: string | undefined;
    if (get(isFirstTurnOfConversation)) {
      conversationTitle = story.title.length > 60 ? story.title.slice(0, 60) + '…' : story.title;
      isFirstTurnOfConversation.set(false);
    }

    chatHistoryStore.update((turns) => [
      ...turns,
      { role: 'user', content: instruction, task_id, timestamp: now },
      {
        role: 'assistant',
        content: '',
        task_id,
        timestamp: now + 1,
        status: 'planning',
        status_message: 'Planning…',
        sources: []
      }
    ]);

    await submitTask(
      instruction,
      { hn_story_url: story.url },
      task_id,
      conversationId,
      conversationTitle
    );
    await goto(`/conversation/${conversationId}`);
  }

  // Same pattern as handleAskAboutArticle/handleAskAboutHackerNewsStory.
  // The instruction names both "release notes" and "latest release" so
  // Planner's _priority3_github_release gate fires regardless of which
  // phrase survives future wording tweaks (planner.py). github_repo pins
  // MCPToolDispatcher._run_github_release() to this exact repo, skipping
  // its own github_search resolution step; github_tag additionally pins
  // the exact release shown in the feed right now, so the summary can't
  // drift onto a newer release that lands between this click and the
  // model's tool call.
  async function handleAskAboutRepo(repo: GithubWatchRepo): Promise<void> {
    if ($tasksStore.finalizing || !repo.latest_release) return;

    const instruction = `Summarize the release notes for the latest release of ${repo.label}`;
    const task_id = crypto.randomUUID();
    const now = Date.now();
    const conversationId = get(currentConversationId);

    let conversationTitle: string | undefined;
    if (get(isFirstTurnOfConversation)) {
      conversationTitle = repo.label.length > 60 ? repo.label.slice(0, 60) + '…' : repo.label;
      isFirstTurnOfConversation.set(false);
    }

    chatHistoryStore.update((turns) => [
      ...turns,
      { role: 'user', content: instruction, task_id, timestamp: now },
      {
        role: 'assistant',
        content: '',
        task_id,
        timestamp: now + 1,
        status: 'planning',
        status_message: 'Planning…',
        sources: []
      }
    ]);

    await submitTask(
      instruction,
      { github_repo: repo.key, github_tag: repo.latest_release.tag_name },
      task_id,
      conversationId,
      conversationTitle
    );
    await goto(`/conversation/${conversationId}`);
  }
</script>

{#if $previewsPanelCollapsed}
  <button
    type="button"
    class="previews-tab-collapsed"
    on:click={togglePreviewsPanel}
    aria-label="Expand Live Feed panel"
    title="Live Feed"
  >
    <span class="previews-tab-label">Live Feed</span>
  </button>
{:else}
  <div class="previews-panel">
    <div class="previews-panel-header">
      <span class="previews-panel-title">Live Feed</span>
      <button
        type="button"
        class="previews-collapse-btn"
        on:click={togglePreviewsPanel}
        aria-label="Collapse Live Feed panel"
        title="Collapse"
      >›</button>
    </div>

    <div class="previews-panel-body">
      <!-- News block — live, moved out of StatusBar's cramped hover popover -->
      <section class="preview-block">
        <div class="preview-block-header">
          <span class="preview-block-title">Daily News Brief</span>
          <button
            type="button"
            class="preview-block-collapse-btn"
            on:click={() => togglePreviewBlock('news')}
            aria-label={$previewBlocksCollapsed.news ? 'Expand Daily News Brief' : 'Collapse Daily News Brief'}
            title={$previewBlocksCollapsed.news ? 'Expand' : 'Collapse'}
          >{$previewBlocksCollapsed.news ? '▸' : '▾'}</button>
        </div>
        {#if !$previewBlocksCollapsed.news}
          <button
            type="button"
            class="preview-block-refresh-link"
            on:click={handleOpenBrief}
            disabled={$newsBriefOpening}
          >{$newsBriefOpening ? 'Refreshing…' : 'Daily News Brief Refresh'}</button>
          <div class="preview-block-body">
            {#if $newsBriefPreview.sections.length === 0}
              <p class="preview-empty">No brief generated yet today — click the link above to generate.</p>
            {:else}
              {#each $newsBriefPreview.sections as section (section.key)}
                <div class="preview-news-section">
                  <div class="preview-news-section-label">{section.label}</div>
                  {#if section.error}
                    <p class="preview-news-unavailable">unavailable</p>
                  {:else if section.articles.length === 0}
                    <p class="preview-news-unavailable">no articles found</p>
                  {:else}
                    {#each section.articles.slice(0, 3) as article}
                      <div class="preview-news-article-row">
                        <a
                          class="preview-news-article"
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <span class="preview-news-article-title">{article.title}</span>
                          <span class="preview-news-article-source">{article.source}</span>
                        </a>
                        <button
                          type="button"
                          class="preview-news-article-ask"
                          on:click={() => handleAskAboutArticle(article)}
                          disabled={$tasksStore.finalizing}
                          title="Ask about this story in the current chat"
                        >Ask about this</button>
                      </div>
                    {/each}
                  {/if}
                </div>
              {/each}
            {/if}
            {#if $newsBriefError}
              <p class="preview-error">{$newsBriefError}</p>
            {/if}
          </div>
        {/if}
      </section>

      <!-- Hacker News block — live. Same shape as the News block above
           (single refresh link + preview.stories rendered generically,
           with a per-story "Ask about this" button — unlike GitHub Watch,
           this feed does touch chat via the hacker_news_search MCP tool,
           see mcp_server/hacker_news.py). Each story links straight to the
           original article; self-posts (Ask HN/Show HN) fall back to the
           HN discussion page since they have no external article. -->
      <section class="preview-block">
        <div class="preview-block-header">
          <span class="preview-block-title">Hacker News</span>
          <button
            type="button"
            class="preview-block-collapse-btn"
            on:click={() => togglePreviewBlock('hackerNews')}
            aria-label={$previewBlocksCollapsed.hackerNews ? 'Expand Hacker News' : 'Collapse Hacker News'}
            title={$previewBlocksCollapsed.hackerNews ? 'Expand' : 'Collapse'}
          >{$previewBlocksCollapsed.hackerNews ? '▸' : '▾'}</button>
        </div>
        {#if !$previewBlocksCollapsed.hackerNews}
          <button
            type="button"
            class="preview-block-refresh-link"
            on:click={handleOpenHackerNews}
            disabled={$hackerNewsOpening}
          >{$hackerNewsOpening ? 'Refreshing…' : 'Hacker News Refresh'}</button>
          <div class="preview-block-body">
            {#if $hackerNewsPreview.stories.length === 0}
              <p class="preview-empty">No top stories fetched yet — click the link above to generate.</p>
            {:else if $hackerNewsPreview.stories[0].key === '_error'}
              <p class="preview-news-unavailable">{$hackerNewsPreview.stories[0].error}</p>
            {:else}
              {#each $hackerNewsPreview.stories as story (story.key)}
                <div class="preview-news-article-row">
                  {#if story.error}
                    <p class="preview-news-unavailable">story unavailable</p>
                  {:else}
                    <a
                      class="preview-news-article"
                      href={story.url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <span class="preview-news-article-title">{story.title}</span>
                      <span class="preview-news-article-source">
                        {#if story.score !== null}{story.score} points{/if}
                        {#if story.by} · {story.by}{/if}
                      </span>
                    </a>
                    <button
                      type="button"
                      class="preview-news-article-ask"
                      on:click={() => handleAskAboutHackerNewsStory(story)}
                      disabled={$tasksStore.finalizing}
                      title="Ask about this story in the current chat"
                    >Ask about this</button>
                  {/if}
                </div>
              {/each}
            {/if}
            {#if $hackerNewsError}
              <p class="preview-error">{$hackerNewsError}</p>
            {/if}
          </div>
        {/if}
      </section>

      <!-- GitHub Watch Feed block — live. Same shape as the News block
           above (single refresh link + preview.repos rendered generically),
           plus a per-repo "Ask about this" button (parity with News/Hacker
           News) pinning github_release via context.github_repo/github_tag
           — see handleAskAboutRepo. -->
      <section class="preview-block">
        <div class="preview-block-header">
          <span class="preview-block-title">GitHub Watch Feed</span>
          <button
            type="button"
            class="preview-block-collapse-btn"
            on:click={() => togglePreviewBlock('github')}
            aria-label={$previewBlocksCollapsed.github ? 'Expand GitHub Watch Feed' : 'Collapse GitHub Watch Feed'}
            title={$previewBlocksCollapsed.github ? 'Expand' : 'Collapse'}
          >{$previewBlocksCollapsed.github ? '▸' : '▾'}</button>
        </div>
        {#if !$previewBlocksCollapsed.github}
          <button
            type="button"
            class="preview-block-refresh-link"
            on:click={handleOpenGithubWatch}
            disabled={$githubWatchOpening}
          >{$githubWatchOpening ? 'Refreshing…' : 'GitHub Watch Feed Refresh'}</button>
          <div class="preview-block-body">
            {#if $githubWatchPreview.repos.length === 0}
              <p class="preview-empty">No watch feed generated yet — click the link above to generate.</p>
            {:else if $githubWatchPreview.repos[0].key === '_error'}
              <p class="preview-news-unavailable">{$githubWatchPreview.repos[0].error}</p>
            {:else}
              {#each $githubWatchPreview.repos as repo (repo.key)}
                <div class="preview-news-article-row">
                  <a
                    class="preview-news-article"
                    href={repo.repo_url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <span class="preview-news-article-title">{repo.label}</span>
                    {#if repo.error}
                      <span class="preview-news-article-source">unavailable</span>
                    {:else if repo.latest_release}
                      <span class="preview-news-article-source">
                        {repo.latest_release.name || repo.latest_release.tag_name}
                        {#if repo.latest_release.published_at}
                          · {repo.latest_release.published_at.slice(0, 10)}
                        {/if}
                      </span>
                    {:else}
                      <span class="preview-news-article-source">no releases yet</span>
                    {/if}
                  </a>
                  {#if repo.latest_release && !repo.error}
                    <button
                      type="button"
                      class="preview-news-article-ask"
                      on:click={() => handleAskAboutRepo(repo)}
                      disabled={$tasksStore.finalizing}
                      title="Ask about this release in the current chat"
                    >Ask about this</button>
                  {/if}
                </div>
              {/each}
            {/if}
            {#if $githubWatchError}
              <p class="preview-error">{$githubWatchError}</p>
            {/if}
          </div>
        {/if}
      </section>
    </div>
  </div>
{/if}

<style>
  .previews-tab-collapsed,
  .previews-panel {
    grid-column: 3;
    grid-row: 1 / -1;
    height: 100%;
    border-left: 1px solid var(--border);
    background: var(--bg-panel);
  }

  /* ── Collapsed: a slim always-visible tab strip ─────────────── */
  .previews-tab-collapsed {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--sp-3) 0;
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease);
  }
  .previews-tab-collapsed:hover { background: var(--bg-hover); }

  .previews-tab-label {
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.02em;
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .previews-tab-collapsed:hover .previews-tab-label { color: var(--text-secondary); }

  /* ── Expanded panel ──────────────────────────────────────────── */
  .previews-panel {
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
  }

  .previews-panel-header {
    flex-shrink: 0;
    height: var(--topbar-h);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 var(--sp-4);
    border-bottom: 1px solid var(--border);
  }

  .previews-panel-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .previews-collapse-btn {
    width: 22px;
    height: 22px;
    border-radius: var(--radius-sm);
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-size: 15px;
    line-height: 1;
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease);
  }
  .previews-collapse-btn:hover { background: var(--bg-hover); }

  .previews-panel-body {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: var(--sp-4);
    display: flex;
    flex-direction: column;
    gap: var(--sp-4);
  }

  /* ── Blob blocks ─────────────────────────────────────────────── */
  .preview-block {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--sp-3);
  }

  .preview-block-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-2);
    margin-bottom: var(--sp-2);
  }

  .preview-block-title {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .preview-block-collapse-btn {
    width: 18px;
    height: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    background: transparent;
    border: none;
    color: var(--text-tertiary);
    font-size: 10px;
    line-height: 1;
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
  }
  .preview-block-collapse-btn:hover {
    background: var(--bg-hover);
    color: var(--text-secondary);
  }

  .preview-block-refresh-link {
    display: inline-block;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-accent);
    background: none;
    border: none;
    cursor: pointer;
    padding: 0;
    margin-bottom: var(--sp-2);
    text-decoration: underline;
  }
  .preview-block-refresh-link:hover:not(:disabled) { color: var(--accent-2); }
  .preview-block-refresh-link:disabled {
    color: var(--text-tertiary);
    cursor: default;
    text-decoration: none;
  }

  .preview-empty {
    font-size: 12px;
    color: var(--text-tertiary);
    line-height: 1.5;
  }

  .preview-news-section { margin-bottom: var(--sp-3); }
  .preview-news-section:last-child { margin-bottom: 0; }

  .preview-news-section-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: var(--sp-1);
  }

  .preview-news-unavailable {
    font-size: 12px;
    color: var(--text-tertiary);
    font-style: italic;
  }

  .preview-news-article-row {
    position: relative;
    padding: var(--sp-1) 0;
  }
  .preview-news-article-row:hover .preview-news-article-ask { opacity: 1; }

  .preview-news-article {
    display: flex;
    flex-direction: column;
    gap: 2px;
    text-decoration: none;
  }
  .preview-news-article:hover .preview-news-article-title { color: var(--text-accent); }

  .preview-news-article-title {
    font-size: 12.5px;
    line-height: 1.4;
    color: var(--text-primary);
  }

  .preview-news-article-source {
    font-size: 11px;
    color: var(--text-tertiary);
  }

  .preview-news-article-ask {
    position: absolute;
    top: var(--sp-1);
    right: 0;
    font-size: 10.5px;
    font-weight: 600;
    color: var(--text-accent);
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 2px var(--sp-2);
    cursor: pointer;
    opacity: 0;
    transition: opacity var(--dur-fast) var(--ease);
  }
  .preview-news-article-ask:hover:not(:disabled) {
    color: var(--accent-2);
    border-color: var(--accent-2);
  }
  .preview-news-article-ask:disabled {
    color: var(--text-tertiary);
    cursor: default;
  }

  .preview-error {
    font-size: 12px;
    color: var(--error);
    margin-top: var(--sp-2);
  }
</style>
