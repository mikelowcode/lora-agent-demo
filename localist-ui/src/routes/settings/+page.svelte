<script lang="ts">
  import { onMount } from 'svelte';
  import { health, checkHealth } from '$lib/stores/server';
  import { apiUrl } from '$lib/api';
  import { modelConfig, RUNTIME_BACKENDS, RUNTIME_BACKEND_LABELS, type RuntimeBackend } from '$lib/stores/model';
  import {
    runtimeBackendSwitchLoading,
    runtimeBackendSwitchError,
    switchRuntimeBackend,
    fetchBackendModels,
    pinChatModel
  } from '$lib/stores/runtimeBackendSwitch';
  import { theme } from '$lib/stores/theme';
  import { fontSize, FONT_SIZES } from '$lib/stores/fontSize';
  import { browser } from '$app/environment';
  import {
    retentionSettings,
    retentionSettingsLoading,
    retentionSettingsError,
    loadRetentionSettings,
    setRetentionPreset,
    type EvictionPreset
  } from '$lib/stores/retentionSettings';
  import {
    reembedLoading,
    reembedError,
    reembedCorpus
  } from '$lib/stores/reembedCorpus';
  import {
    newsPreferences,
    newsPreferencesLoading,
    newsPreferencesError,
    loadNewsPreferences,
    setNewsPreferences
  } from '$lib/stores/newsPreferences';
  import {
    assistantName,
    assistantNameLoading,
    assistantNameError,
    loadAssistantName,
    setAssistantName
  } from '$lib/stores/assistantName';
  import {
    githubWatchPins,
    githubWatchPinsLoading,
    githubWatchPinsError,
    loadGithubWatchPins,
    setGithubWatchPins
  } from '$lib/stores/githubWatchPins';

  const EVICTION_PRESETS: { value: EvictionPreset; label: string }[] = [
    { value: '7d',      label: '7 days' },
    { value: '30d',     label: '30 days' },
    { value: '90d',     label: '90 days' },
    { value: 'forever', label: 'Forever' }
  ];

  onMount(() => {
    loadRetentionSettings();
    loadMemoryStats();
    void loadNewsPreferences().then(() => {
      homeCountryInput = $newsPreferences.home_country;
      localQueryInput  = $newsPreferences.local_query ?? '';
      selectedTopics   = $newsPreferences.topics;
    });
    void loadAssistantName().then(() => {
      assistantNameInput = $assistantName.assistant_name;
    });
    void loadGithubWatchPins().then(() => {
      pinnedRepos = $githubWatchPins.repos;
    });
  });

  // Corpus staleness — fetched once on mount and again after a re-embed
  // completes, so the badge reflects the server's current MemoryManager
  // state without adding a dedicated poll.
  let corpusStale = false;
  let reembedResult: string | null = null;

  async function loadMemoryStats() {
    try {
      const res = await fetch(apiUrl('/api/memory/stats'));
      if (!res.ok) return;
      const data = await res.json();
      corpusStale = Boolean(data.corpus_stale);
    } catch {
      // Leave last-known state — a transient fetch failure here isn't
      // worth surfacing on its own card.
    }
  }

  async function handleReembedCorpus() {
    const proceed = confirm(
      'Re-embed the wiki/raw corpus now with the active embedding model? ' +
      'This blocks until every document has been re-embedded, which can take ' +
      'a while for a large corpus.'
    );
    if (!proceed) return;

    reembedResult = null;
    const result = await reembedCorpus();
    if (result) {
      reembedResult = `Re-embedded ${result.reembedded} of ${result.total} documents.`;
      await loadMemoryStats();
    }
  }

  // Local form state — mirrors store, saved on blur/change
  let backendUrl = '';
  let chatModel  = '';

  // Initialize from store
  $: if (browser) {
    backendUrl = localStorage.getItem('localist-backend-url') ?? 'http://127.0.0.1:8000';
    chatModel  = $modelConfig.chat_model;
  }

  function saveBackendUrl() {
    if (browser) localStorage.setItem('localist-backend-url', backendUrl);
  }

  // The backend whose models the Chat Model dropdown below is scoped to.
  // Deliberately independent of $modelConfig.backend (the real active
  // backend, synced from health polling) — clicking a segmented-control
  // button previews that backend's models immediately, even before (or
  // without) confirming an actual switch to it.
  let selectedUiBackend: RuntimeBackend = $modelConfig.backend as RuntimeBackend;
  let availableModels: string[] = [];
  let modelsFetchToken = 0;

  async function refreshModelsForSelected() {
    const token = ++modelsFetchToken;
    const models = await fetchBackendModels(selectedUiBackend);
    if (token === modelsFetchToken) availableModels = models;
  }

  $: if (selectedUiBackend) refreshModelsForSelected();

  function selectRuntimeBackend(backend: RuntimeBackend) {
    selectedUiBackend = backend;
    if (backend === $modelConfig.backend) return; // already active — nothing to switch

    const proceed = confirm(
      `Switch the active runtime backend to ${RUNTIME_BACKEND_LABELS[backend]}? ` +
      `In-flight requests will keep running on the current backend unaffected, ` +
      `but this switch itself can take several seconds.`
    );
    if (!proceed) return;

    void switchRuntimeBackend(backend);
  }

  async function handleChatModelPick(value: string) {
    if (!value) return;
    chatModel = value;
    modelConfig.update((s) => ({ ...s, chat_model: value }));
    await pinChatModel(selectedUiBackend, value);
  }

  function onChatModelSelectChange(e: Event) {
    const value = (e.target as HTMLSelectElement).value;
    void handleChatModelPick(value);
  }

  let checking = false;
  async function runHealthCheck() {
    checking = true;
    await checkHealth();
    checking = false;
  }

  // Streaming / episodic write-approval — UI-only preferences for now; no
  // backend endpoint exposes either as a live-switchable setting yet (see
  // CLAUDE_CODE_PROMPT.md's request to flag missing backend data). Persisted
  // so the toggle state survives a reload, same pattern as backendUrl above.
  let streaming = true;
  let episodicApproval = false;

  $: if (browser) {
    streaming = localStorage.getItem('localist-streaming') !== '0';
    episodicApproval = localStorage.getItem('localist-episodic-approval') === '1';
  }

  function toggleStreaming() {
    streaming = !streaming;
    if (browser) localStorage.setItem('localist-streaming', streaming ? '1' : '0');
  }

  function toggleEpisodicApproval() {
    episodicApproval = !episodicApproval;
    if (browser) localStorage.setItem('localist-episodic-approval', episodicApproval ? '1' : '0');
  }

  // Daily News Brief preferences (docs/daily-news-brief-plan.md §5) — local
  // form state mirrors the store, synced whenever it reloads, saved
  // explicitly via the Save button below (not on every keystroke/toggle,
  // since exactly-3-topics is only checkable once the user is done picking).
  let homeCountryInput = 'us';
  let localQueryInput  = '';
  let selectedTopics: string[] = [];
  let newsSaveResult: string | null = null;

  function toggleTopic(key: string) {
    newsSaveResult = null;
    if (selectedTopics.includes(key)) {
      selectedTopics = selectedTopics.filter((t) => t !== key);
    } else if (selectedTopics.length < 3) {
      selectedTopics = [...selectedTopics, key];
    }
  }

  // The backend requires exactly 3 topics on every PUT (main.py enforces
  // this — it's not a frontend-only convention), and Local area/Home
  // country are saved in that same request. So an incomplete topic
  // selection silently blocks a city-only edit too — the Save button below
  // is disabled on this condition so that's no longer discoverable only
  // after a failed click.
  $: topicsIncomplete = selectedTopics.length !== 3;

  async function handleSaveNewsPreferences() {
    newsSaveResult = null;
    if (topicsIncomplete) {
      newsPreferencesError.set('Select exactly 3 special-interest topics.');
      return;
    }
    const ok = await setNewsPreferences(
      homeCountryInput.trim().toLowerCase(),
      localQueryInput.trim() || null,
      selectedTopics
    );
    if (ok) {
      const city = $newsPreferences.local_query;
      newsSaveResult = city
        ? `Saved — Local area set to "${city}".`
        : 'Saved — Local area cleared.';
    }
  }

  // Pinned GitHub repos — release-only tracking independent of GitHub's
  // Watch/subscriptions relationship (no PR/issue email noise). Local list
  // mirrors the store, synced whenever it reloads, saved explicitly via the
  // Save button (whole-list replace, same as News Preferences' topics).
  const PINNED_REPO_RE = /^[\w.-]+\/[\w.-]+$/;
  let pinnedRepos: string[] = [];
  let pinnedRepoInput = '';
  let pinnedRepoAddError: string | null = null;
  let pinnedReposSaveResult: string | null = null;

  function addPinnedRepo() {
    pinnedReposSaveResult = null;
    const trimmed = pinnedRepoInput.trim();
    if (!trimmed) return;
    if (!PINNED_REPO_RE.test(trimmed)) {
      pinnedRepoAddError = 'Expected the form "owner/repo", e.g. ollama/ollama.';
      return;
    }
    if (pinnedRepos.some((r) => r.toLowerCase() === trimmed.toLowerCase())) {
      pinnedRepoAddError = 'Already pinned.';
      return;
    }
    if (pinnedRepos.length >= 20) {
      pinnedRepoAddError = 'Up to 20 pinned repos.';
      return;
    }
    pinnedRepos = [...pinnedRepos, trimmed];
    pinnedRepoInput = '';
    pinnedRepoAddError = null;
  }

  function removePinnedRepo(repo: string) {
    pinnedRepos = pinnedRepos.filter((r) => r !== repo);
    pinnedReposSaveResult = null;
  }

  $: pinnedReposUnchanged =
    JSON.stringify(pinnedRepos) === JSON.stringify($githubWatchPins.repos);

  async function handleSavePinnedRepos() {
    pinnedReposSaveResult = null;
    const ok = await setGithubWatchPins(pinnedRepos);
    if (ok) {
      pinnedReposSaveResult =
        pinnedRepos.length > 0
          ? `Saved — ${pinnedRepos.length} repo${pinnedRepos.length === 1 ? '' : 's'} pinned.`
          : 'Saved — no repos pinned.';
    }
  }

  // Assistant name — form state mirrors the store, saved explicitly via the
  // Save button (not on every keystroke), same convention as News Preferences.
  let assistantNameInput = 'Localist';
  let assistantNameSaveResult: string | null = null;

  $: assistantNameUnchanged =
    assistantNameInput.trim() === '' ||
    assistantNameInput.trim() === $assistantName.assistant_name;

  async function handleSaveAssistantName() {
    assistantNameSaveResult = null;
    const trimmed = assistantNameInput.trim();
    if (!trimmed) {
      assistantNameError.set('Assistant name must not be empty.');
      return;
    }
    const ok = await setAssistantName(trimmed);
    if (ok) {
      assistantNameSaveResult = `Saved — now called "${$assistantName.assistant_name}".`;
    }
  }
</script>

<svelte:head>
  <title>Settings — Localist</title>
</svelte:head>

<div class="settings-page">
  <div class="settings-inner">

    <!-- Runtime backend -->
    <section class="settings-card">
      <div class="card-title">Runtime Backend</div>
      <div class="segmented">
        {#each RUNTIME_BACKENDS as b}
          <button
            type="button"
            class="seg-btn"
            class:active={$modelConfig.backend === b}
            class:previewing={selectedUiBackend === b && selectedUiBackend !== $modelConfig.backend}
            disabled={$runtimeBackendSwitchLoading}
            on:click={() => selectRuntimeBackend(b)}
          >{RUNTIME_BACKEND_LABELS[b]}</button>
        {/each}
        {#if $runtimeBackendSwitchLoading}
          <span class="spinner" aria-hidden="true" />
        {/if}
      </div>
      <p class="card-hint">
        Live-switches the active backend via <code>POST /settings/runtime-backend</code>
        (docs/architecture/16-runtime-backend-layer.md §16.5) — health-checked before anything
        changes, and persisted to <code>.env</code> so it survives a restart. Switching takes a
        few seconds (mostly model-load time on the new backend); in-flight requests keep running
        on the backend that was active when they started.
      </p>
      {#if $runtimeBackendSwitchError}
        <p class="card-hint" style="color:var(--error)">{$runtimeBackendSwitchError}</p>
      {/if}
    </section>

    <!-- Corpus re-embed -->
    <section class="settings-card">
      <div class="card-title">Corpus Embeddings</div>
      <p class="card-hint">
        Re-embeds the wiki/raw corpus with the active embedding model via
        <code>POST /memory/reembed</code> (docs/architecture/16-runtime-backend-layer.md §16.4).
        Needed after switching embedding models or embed sources — until then, retrieval
        falls back to keyword-only ranking.
      </p>
      {#if corpusStale}
        <span class="badge badge-warning">
          Corpus embeddings out of date — re-ranking is running keyword-only.
        </span>
      {/if}
      <div class="field-row">
        <button
          type="button"
          class="btn-secondary"
          disabled={$reembedLoading}
          on:click={handleReembedCorpus}
        >
          {#if $reembedLoading}
            <span class="spinner" aria-hidden="true" />
            Re-embedding…
          {:else}
            Re-embed Corpus Now
          {/if}
        </button>
      </div>
      {#if reembedResult}
        <p class="card-hint">{reembedResult}</p>
      {/if}
      {#if $reembedError}
        <p class="card-hint" style="color:var(--error)">{$reembedError}</p>
      {/if}
    </section>

    <!-- Chat model -->
    <section class="settings-card">
      <div class="card-title">Chat Model — {RUNTIME_BACKEND_LABELS[selectedUiBackend]}</div>
      <div class="field-group">
        <select
          id="chat-model"
          aria-label={`Chat model for ${RUNTIME_BACKEND_LABELS[selectedUiBackend]}`}
          class="settings-input"
          disabled={$runtimeBackendSwitchLoading || availableModels.length === 0}
          on:change={onChatModelSelectChange}
        >
          <option value="" disabled selected={!chatModel || !availableModels.includes(chatModel)}>
            {availableModels.length ? 'Select a model…' : 'No models reported by this backend'}
          </option>
          {#each availableModels as m}
            <option value={m} selected={m === chatModel}>{m}</option>
          {/each}
        </select>
        {#if selectedUiBackend === $modelConfig.backend}
          <p class="field-hint">Applies immediately — {RUNTIME_BACKEND_LABELS[selectedUiBackend]} is the active backend.</p>
        {:else}
          <p class="field-hint">Saved for {RUNTIME_BACKEND_LABELS[selectedUiBackend]} — takes effect the next time you switch to it.</p>
        {/if}
      </div>
    </section>

    <!-- Streaming / episodic approval toggles -->
    <section class="settings-card row">
      <div class="card-title">Streaming responses</div>
      <button
        type="button"
        class="switch"
        class:on={streaming}
        aria-label="Toggle streaming responses"
        on:click={toggleStreaming}
      ><span class="switch-knob" /></button>
    </section>

    <section class="settings-card row">
      <div class="card-title">Episodic write-approval</div>
      <button
        type="button"
        class="switch"
        class:on={episodicApproval}
        aria-label="Toggle episodic write-approval"
        on:click={toggleEpisodicApproval}
      ><span class="switch-knob" /></button>
    </section>

    <!-- Appearance -->
    <section class="settings-card row">
      <div class="card-title">Theme — currently <strong>{$theme}</strong></div>
      <button
        type="button"
        class="switch"
        class:on={$theme === 'light'}
        aria-label="Toggle light/dark theme"
        on:click={theme.toggle}
      ><span class="switch-knob" /></button>
    </section>

    <!-- Font size -->
    <section class="settings-card">
      <div class="card-title">Font Size</div>
      <p class="card-desc">Scales text throughout the app. Some UI chrome and labels are a fixed size and won't change.</p>
      <div class="segmented">
        {#each FONT_SIZES as f}
          <button
            type="button"
            class="seg-btn"
            class:active={$fontSize === f.value}
            on:click={() => fontSize.set(f.value)}
          >{f.label}</button>
        {/each}
      </div>
    </section>

    <!-- Backend connection -->
    <section class="settings-card">
      <div class="card-title">Backend Connection</div>
      <p class="card-desc">Configure the FastAPI server that the UI connects to.</p>

      <div class="field-group">
        <label class="field-label" for="backend-url">Server URL</label>
        <div class="field-row">
          <input
            id="backend-url"
            type="text"
            class="settings-input"
            bind:value={backendUrl}
            on:blur={saveBackendUrl}
            placeholder="http://127.0.0.1:8000"
            spellcheck="false"
          />
          <button
            class="btn-secondary"
            on:click={runHealthCheck}
            disabled={checking}
            aria-label="Test connection"
          >
            {#if checking}
              <span class="spinner" aria-hidden="true" />
              Checking…
            {:else}
              Test connection
            {/if}
          </button>
        </div>

        <div class="health-readout">
          <div class="health-row">
            <span class="health-label">Status</span>
            <span
              class="badge"
              class:badge-success={$health.healthy}
              class:badge-error={!$health.reachable && !$health.checking}
              class:badge-warning={$health.reachable && !$health.healthy}
              class:badge-muted={$health.checking}
            >
              {#if $health.checking}checking…
              {:else if $health.healthy}online
              {:else if $health.reachable}degraded
              {:else}offline{/if}
            </span>
          </div>
          <div class="health-row">
            <span class="health-label">Embeddings</span>
            <span
              class="badge"
              class:badge-success={$health.embed_model_found}
              class:badge-warning={!$health.embed_model_found}
              title={$health.embed_model_found
                ? 'Cosine-similarity retrieval active (mlx-community/embeddinggemma-300m-4bit).'
                : 'Running in keyword-only fallback — check backend logs for the load error.'}
            >
              {$health.embed_model_found ? 'ready' : 'keyword-only fallback'}
            </span>
          </div>
          {#if $health.base_url}
            <div class="health-row">
              <span class="health-label">Resolved URL</span>
              <span class="health-value text-mono">{$health.base_url}</span>
            </div>
          {/if}
          {#if $health.models.length > 0}
            <div class="health-row">
              <span class="health-label">Available models</span>
              <div class="model-chips">
                {#each $health.models as m}
                  <span class="badge badge-muted">{m}</span>
                {/each}
              </div>
            </div>
          {/if}
          {#if $health.error}
            <div class="health-row error-row">
              <span class="health-label">Error</span>
              <span class="health-value text-mono" style="color:var(--error)">{$health.error}</span>
            </div>
          {/if}
        </div>
      </div>
    </section>

    <!-- Assistant name -->
    <section class="settings-card">
      <div class="card-title">Assistant Name</div>
      <p class="card-desc">
        What the assistant calls itself in chat — updates the system prompt
        and persona on save; takes effect on your very next message.
      </p>

      <label class="news-field-label" for="assistant-name-input">Name</label>
      <input
        id="assistant-name-input"
        class="news-text-input"
        type="text"
        maxlength="60"
        bind:value={assistantNameInput}
        placeholder="Localist"
      />

      <button
        type="button"
        class="seg-btn news-save-btn"
        disabled={$assistantNameLoading || assistantNameUnchanged}
        on:click={handleSaveAssistantName}
      >Save</button>

      {#if assistantNameSaveResult}
        <p class="card-hint">{assistantNameSaveResult}</p>
      {/if}
      {#if $assistantNameError}
        <p class="card-hint" style="color:var(--error)">{$assistantNameError}</p>
      {/if}
    </section>

    <!-- Data retention (chat history + episodes) -->
    <section class="settings-card">
      <div class="card-title">Data Retention</div>
      <p class="card-desc">Choose how long chat history and saved episodes are kept before they're automatically cleaned up.</p>
      <div class="segmented">
        {#each EVICTION_PRESETS as p}
          <button
            type="button"
            class="seg-btn"
            class:active={$retentionSettings.eviction_preset === p.value}
            disabled={$retentionSettingsLoading}
            on:click={() => setRetentionPreset(p.value)}
          >{p.label}</button>
        {/each}
      </div>
      {#if $retentionSettingsError}
        <p class="card-hint" style="color:var(--error)">{$retentionSettingsError}</p>
      {/if}
    </section>

    <!-- Daily News Brief -->
    <section class="settings-card">
      <div class="card-title">Daily News Brief</div>
      <p class="card-desc">
        Top Stories (anchored to your home country) is fixed; Local is
        keyword-matched against a place you set (NewsAPI has no true
        local-news coverage). Pick exactly 3 special-interest topics.
      </p>

      <label class="news-field-label" for="news-home-country">Home country (2-letter code)</label>
      <input
        id="news-home-country"
        class="news-text-input"
        type="text"
        maxlength="2"
        bind:value={homeCountryInput}
        placeholder="us"
      />

      <label class="news-field-label" for="news-local-query">Local area (e.g. a city name)</label>
      <input
        id="news-local-query"
        class="news-text-input"
        type="text"
        autocomplete="off"
        bind:value={localQueryInput}
        placeholder="e.g. Seattle"
      />

      <div class="news-field-label">
        Special-interest topics ({selectedTopics.length}/3 selected)
      </div>
      <div class="news-topic-grid">
        {#each Object.entries($newsPreferences.topic_pool) as [key, label]}
          <button
            type="button"
            class="news-topic-chip"
            class:active={selectedTopics.includes(key)}
            disabled={!selectedTopics.includes(key) && selectedTopics.length >= 3}
            on:click={() => toggleTopic(key)}
          >{label}</button>
        {/each}
      </div>
      {#if topicsIncomplete}
        <p class="card-hint" style="color:var(--text-tertiary)">
          Pick exactly 3 topics to enable Save — this also saves your Local
          area and Home country changes above; they're sent together in one
          request.
        </p>
      {/if}

      <button
        type="button"
        class="seg-btn news-save-btn"
        disabled={$newsPreferencesLoading || topicsIncomplete}
        title={topicsIncomplete ? 'Select exactly 3 topics to save' : undefined}
        on:click={handleSaveNewsPreferences}
      >Save</button>

      {#if newsSaveResult}
        <p class="card-hint">{newsSaveResult}</p>
      {/if}
      {#if $newsPreferencesError}
        <p class="card-hint" style="color:var(--error)">{$newsPreferencesError}</p>
      {/if}
    </section>

    <!-- Pinned GitHub Repos -->
    <section class="settings-card">
      <div class="card-title">Pinned GitHub Repos</div>
      <p class="card-desc">
        Track release notes for repos without clicking "Watch" on GitHub —
        pinning here doesn't subscribe you to that repo's PR/issue emails.
        Shows up alongside any repos you do watch in the Live Feed panel's
        GitHub block.
      </p>

      <label class="news-field-label" for="pinned-repo-input">Add a repo (owner/repo)</label>
      <div class="pinned-repo-add-row">
        <input
          id="pinned-repo-input"
          class="news-text-input"
          type="text"
          autocomplete="off"
          bind:value={pinnedRepoInput}
          placeholder="e.g. ollama/ollama"
          on:input={() => (pinnedRepoAddError = null)}
          on:keydown={(e) => e.key === 'Enter' && addPinnedRepo()}
        />
        <button type="button" class="seg-btn" on:click={addPinnedRepo}>Add</button>
      </div>
      {#if pinnedRepoAddError}
        <p class="card-hint" style="color:var(--error)">{pinnedRepoAddError}</p>
      {/if}

      {#if pinnedRepos.length > 0}
        <div class="news-topic-grid">
          {#each pinnedRepos as repo (repo)}
            <span class="pinned-repo-chip">
              {repo}
              <button
                type="button"
                class="pinned-repo-remove"
                aria-label={`Remove ${repo}`}
                on:click={() => removePinnedRepo(repo)}
              >&times;</button>
            </span>
          {/each}
        </div>
      {/if}

      <button
        type="button"
        class="seg-btn news-save-btn"
        disabled={$githubWatchPinsLoading || pinnedReposUnchanged}
        on:click={handleSavePinnedRepos}
      >Save</button>

      {#if pinnedReposSaveResult}
        <p class="card-hint">{pinnedReposSaveResult}</p>
      {/if}
      {#if $githubWatchPinsError}
        <p class="card-hint" style="color:var(--error)">{$githubWatchPinsError}</p>
      {/if}
    </section>

    <!-- Version -->
    <section class="settings-card row">
      <div class="card-title">Version</div>
      <div class="card-value text-mono">0.2.0 · web</div>
    </section>

  </div>
</div>

<style>
  .settings-page {
    flex: 1;
    overflow-y: auto;
    padding: var(--sp-8);
  }

  .settings-inner {
    max-width: 560px;
    display: flex;
    flex-direction: column;
    gap: var(--sp-4);
  }

  /* Card */
  .settings-card {
    display: flex;
    flex-direction: column;
    gap: var(--sp-3);
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: var(--sp-4) var(--sp-5);
  }

  .settings-card.row {
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-4);
  }

  .card-title {
    font-size: 12.5px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .card-desc {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    margin-top: calc(var(--sp-3) * -1);
  }

  .card-value {
    font-size: 12.5px;
    color: var(--text-secondary);
  }

  .card-hint {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    line-height: 1.6;
  }
  .card-hint code {
    font-size: 10.5px;
  }

  /* Fields */
  .field-group {
    display: flex;
    flex-direction: column;
    gap: var(--sp-2);
  }

  .field-label {
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--text-secondary);
  }

  .field-hint {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    line-height: 1.5;
  }

  .field-row {
    display: flex;
    gap: var(--sp-2);
  }

  .settings-input {
    flex: 1;
    padding: var(--sp-2) var(--sp-3);
    font-size: var(--text-sm);
    font-family: var(--font-mono);
    min-width: 0;
  }

  /* Buttons */
  .btn-secondary {
    display: inline-flex;
    align-items: center;
    gap: var(--sp-2);
    padding: var(--sp-2) var(--sp-4);
    background: var(--bg-raised);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: var(--text-sm);
    border-radius: var(--radius);
    white-space: nowrap;
    transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  /* Spinner */
  .spinner {
    display: inline-block;
    width: 11px; height: 11px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Health readout */
  .health-readout {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--sp-4);
    display: flex;
    flex-direction: column;
    gap: var(--sp-3);
  }

  .health-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--sp-4);
    font-size: var(--text-xs);
  }

  .health-label {
    color: var(--text-tertiary);
    font-family: var(--font-mono);
    flex-shrink: 0;
  }

  .health-value {
    color: var(--text-secondary);
    text-align: right;
    word-break: break-all;
  }

  .error-row .health-value { color: var(--error); }

  .model-chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-1);
    justify-content: flex-end;
  }

  .news-field-label {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: var(--sp-3);
  }

  .news-text-input {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: var(--text-sm);
    padding: var(--sp-2) var(--sp-3);
    margin-top: var(--sp-1);
  }

  .news-topic-grid {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-2);
    margin-top: var(--sp-2);
  }

  .news-topic-chip {
    padding: var(--sp-1) var(--sp-3);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    background: var(--bg-raised);
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
  }
  .news-topic-chip:hover:not(:disabled) { background: var(--bg-hover); }
  .news-topic-chip.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .news-topic-chip:disabled:not(.active) { cursor: not-allowed; opacity: 0.5; }

  .news-save-btn {
    margin-top: var(--sp-3);
    align-self: flex-start;
  }

  .pinned-repo-add-row {
    display: flex;
    gap: var(--sp-2);
    align-items: center;
  }
  .pinned-repo-add-row .news-text-input { flex: 1; }

  .pinned-repo-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--sp-1);
    padding: var(--sp-1) var(--sp-2) var(--sp-1) var(--sp-3);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    background: var(--bg-raised);
    color: var(--text-secondary);
    font-size: 12px;
    font-family: var(--font-mono);
  }

  .pinned-repo-remove {
    background: none;
    border: none;
    color: var(--text-tertiary);
    cursor: pointer;
    font-size: 14px;
    line-height: 1;
    padding: 0 var(--sp-1);
  }
  .pinned-repo-remove:hover { color: var(--error); }
</style>
