<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';
  import { apiUrl } from '$lib/api';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { startNewConversation } from '$lib/stores/conversation';
  import { pendingCount, refreshPendingCount } from '$lib/stores/episodes';
  import { theme } from '$lib/stores/theme';
  import {
    sidebarWidth, sidebarCollapsed,
    MIN_WIDTH, MAX_WIDTH, COLLAPSE_THRESHOLD
  } from '$lib/stores/sidebar';
  import {
    rawFiles, wikiFiles, generatedFiles,
    rawLoading, wikiLoading, generatedLoading,
    rawError, wikiError, generatedError,
    ingest, isIngesting,
    loadRawFiles, loadWikiFiles, loadGeneratedFiles,
    uploadFile, ingestFile, resetIngest,
    formatBytes,
    type FileEntry,
  } from '$lib/stores/files';
  import { selectedFile, selectFile, closeFile } from '$lib/stores/fileSelection';
  import { OCR_EXTENSIONS, extOf } from '$lib/utils/ocr';

  $: active = $page.url.pathname;
  $: isChatActive  = active.startsWith('/conversation');
  $: isFilesActive = active.startsWith('/files');

  // ── Chat sub-nav (conversation list) ─────────────────────────────
  let chatHistoryExpanded = false;

  interface ConversationSummary {
    conversation_id:    string;
    conversation_title: string | null;
    last_created_at:    number;
    first_created_at:   number;
  }

  let conversations: ConversationSummary[] = [];
  let conversationsLoading = false;
  let conversationsError: string | null = null;

  async function loadConversations(): Promise<void> {
    conversationsLoading = true;
    conversationsError = null;
    try {
      const res = await fetch(apiUrl('/api/chat/history/conversations'));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: { conversations: ConversationSummary[] } = await res.json();
      conversations = data.conversations;
    } catch (err) {
      conversationsError = err instanceof Error ? err.message : String(err);
    } finally {
      conversationsLoading = false;
    }
  }

  // Re-fetch whenever the sub-list is opened while on a /conversation* route
  // (also covers the moment right after startNewConversation() navigates to
  // the freshly minted conversation).
  $: if (chatHistoryExpanded && isChatActive) {
    loadConversations();
  }

  function conversationLabel(c: ConversationSummary): string {
    if (c.conversation_title) return c.conversation_title;
    const ts = new Date(c.last_created_at * 1000).toLocaleString([], {
      month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'
    });
    return `New conversation — ${ts}`;
  }

  // Delete (same two-step inline-confirm pattern as the Files delete below).
  let confirmDeleteConversationId: string | null = null;
  let deletingConversationId: string | null = null;
  let deleteConversationError: string | null = null;

  function requestDeleteConversation(id: string, e: Event): void {
    e.stopPropagation();
    deleteConversationError = null;
    confirmDeleteConversationId = id;
  }

  function cancelDeleteConversation(e: Event): void {
    e.stopPropagation();
    confirmDeleteConversationId = null;
  }

  async function confirmDeleteConversation(c: ConversationSummary, e: Event): Promise<void> {
    e.stopPropagation();
    deletingConversationId = c.conversation_id;
    deleteConversationError = null;
    try {
      const res = await fetch(apiUrl(`/api/chat/history/conversations/${c.conversation_id}`), { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      await loadConversations();

      // Deleted the conversation currently open in the main pane — same
      // "spin up a fresh one" behavior as the New chat button, since the
      // old id no longer resolves to anything.
      if (c.conversation_id === $page.params.id) {
        const id = startNewConversation();
        await goto(`/conversation/${id}`);
      }
    } catch (err) {
      deleteConversationError = err instanceof Error ? err.message : String(err);
    } finally {
      confirmDeleteConversationId = null;
      deletingConversationId = null;
    }
  }

  function handleChatNavClick(): void {
    if (isChatActive) {
      chatHistoryExpanded = !chatHistoryExpanded;
    } else {
      chatHistoryExpanded = true;
      goto('/conversation');
    }
  }

  async function handleNewConversation(): Promise<void> {
    const id = startNewConversation();
    chatHistoryExpanded = false;
    await goto(`/conversation/${id}`);
  }

  // ── Files sub-nav (Wiki / Raw / Generated groups) ────────────────
  let filesNavExpanded = false;
  let filesExpanded: Record<string, boolean> = { Wiki: true, Raw: true, Generated: true };

  function toggleFileGroup(name: string): void {
    filesExpanded = { ...filesExpanded, [name]: !(filesExpanded[name] !== false) };
  }

  function handleFilesNavClick(): void {
    if (isFilesActive) {
      filesNavExpanded = !filesNavExpanded;
    } else {
      filesNavExpanded = true;
      goto('/files');
    }
  }

  $: fileGroups = [
    { key: 'Wiki',      files: $wikiFiles,      loading: $wikiLoading,      error: $wikiError,      refresh: loadWikiFiles },
    { key: 'Raw',       files: $rawFiles,       loading: $rawLoading,       error: $rawError,       refresh: loadRawFiles },
    { key: 'Generated', files: $generatedFiles, loading: $generatedLoading, error: $generatedError, refresh: loadGeneratedFiles },
  ];

  function formatSize(bytes?: number): string {
    return bytes === undefined ? '' : formatBytes(bytes);
  }

  async function handleIngest(entry: FileEntry, e: Event): Promise<void> {
    e.stopPropagation();
    if ($isIngesting) return;
    resetIngest();
    await ingestFile(entry);
  }

  // ── Delete (two-step: click primes an inline "are you sure", a second
  // click on the same row confirms; clicking any other row's delete button,
  // or Cancel, drops the pending confirmation) ──────────────────────
  let confirmDeletePath: string | null = null;
  let deletingPath: string | null = null;
  let deleteError: string | null = null;

  function requestDelete(path: string, e: Event): void {
    e.stopPropagation();
    deleteError = null;
    confirmDeletePath = path;
  }

  function cancelDelete(e: Event): void {
    e.stopPropagation();
    confirmDeletePath = null;
  }

  async function confirmDelete(entry: FileEntry, e: Event): Promise<void> {
    e.stopPropagation();
    deletingPath = entry.path;
    deleteError = null;
    try {
      const res = await fetch(apiUrl(`/api/files?path=${encodeURIComponent(entry.path)}`), { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      if (entry.type === 'wiki') await loadWikiFiles();
      else if (entry.type === 'raw') await loadRawFiles();
      else await loadGeneratedFiles();

      if ($selectedFile?.path === entry.path) closeFile();
    } catch (err) {
      deleteError = err instanceof Error ? err.message : String(err);
    } finally {
      confirmDeletePath = null;
      deletingPath = null;
    }
  }

  // ── Upload (raw files) ───────────────────────────────────────────
  // .md/.txt plus the same OCR-eligible (image/PDF) extensions
  // POST /files/upload accepts — see mcp_server/ocr.py's
  // OCR_MIME_BY_EXTENSION. OCR-eligible uploads are saved as-is; WikiAgent
  // extracts text from them lazily at ingest time, not here.
  const RAW_UPLOAD_EXTENSIONS = new Set(['.md', '.txt', ...OCR_EXTENSIONS]);

  let uploading = false;
  let uploadError: string | null = null;
  let fileInputEl: HTMLInputElement;

  async function onFileInput(e: Event): Promise<void> {
    const input = e.target as HTMLInputElement;
    // Snapshot into a plain array of File objects (immutable blobs) BEFORE
    // resetting input.value — resetting .value empties the same live
    // FileList object in place rather than replacing it with a fresh one,
    // so a reference to the FileList itself (as opposed to the individual
    // File objects) captured beforehand still reads back as length 0 once
    // .value is cleared. Matches ChatPanel.svelte's handleFileSelect(),
    // which extracts its File before resetting for the same reason.
    const list = Array.from(input.files ?? []);
    input.value = '';
    if (!list.length) return;

    uploadError = null;
    for (const f of list) {
      const ext = extOf(f.name);
      if (!RAW_UPLOAD_EXTENSIONS.has(ext)) {
        uploadError = `Unsupported type: ${ext}.`;
        return;
      }
    }

    uploading = true;
    try {
      for (const f of list) await uploadFile(f);
    } catch (err) {
      uploadError = err instanceof Error ? err.message : String(err);
    } finally {
      uploading = false;
    }
  }

  // ── Sidebar resize / collapse ─────────────────────────────────────
  let dragging = false;
  let dragStartX = 0;
  let dragStartW = 0;

  function startResize(e: MouseEvent): void {
    dragging = true;
    dragStartX = e.clientX;
    dragStartW = $sidebarWidth;
    e.preventDefault();
  }

  function onMove(e: MouseEvent): void {
    if (!dragging) return;
    const dx = e.clientX - dragStartX;
    const w = dragStartW + dx;
    if (w < COLLAPSE_THRESHOLD) {
      sidebarCollapsed.set(true);
    } else {
      sidebarCollapsed.set(false);
      sidebarWidth.set(Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, w)));
    }
  }

  function onUp(): void {
    dragging = false;
  }

  function onDividerKeydown(e: KeyboardEvent): void {
    if (e.key === 'ArrowLeft') {
      sidebarWidth.set(Math.max(MIN_WIDTH, $sidebarWidth - 8));
    } else if (e.key === 'ArrowRight') {
      sidebarWidth.set(Math.min(MAX_WIDTH, $sidebarWidth + 8));
    } else {
      return;
    }
    e.preventDefault();
  }

  onMount(() => {
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    refreshPendingCount();
    loadWikiFiles();
    loadRawFiles();
    loadGeneratedFiles();
  });

  onDestroy(() => {
    if (browser) {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    }
  });
</script>

<aside
  class="sidebar"
  aria-label="Main navigation"
  aria-hidden={$sidebarCollapsed}
  inert={$sidebarCollapsed}
>
  <!-- Wordmark -->
  <div class="wordmark">
    <svg class="brand-mark" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path fill-rule="evenodd" clip-rule="evenodd" fill="currentColor" d="
        M 8.111,14.389
        A 5.5,5.5 0 0 1 8.111,6.611
        A 5.5,5.5 0 0 1 15.889,6.611
        A 5.5,5.5 0 0 1 15.889,14.389
        L 12,18.278
        Z
        M 12,11.7 A 1.7,1.7 0 1 0 12,8.3 A 1.7,1.7 0 1 0 12,11.7 Z
      "></path>
    </svg>
    <span class="wordmark-text">LOCALIST</span>
  </div>

  <!-- Nav links -->
  <nav>
    <ul>
      <!-- Chat -->
      <li>
        <button
          type="button"
          class="nav-link"
          class:active={isChatActive}
          aria-current={isChatActive ? 'page' : undefined}
          aria-expanded={chatHistoryExpanded}
          on:click={handleChatNavClick}
        >
          <span class="nav-label">Chat</span>
          <span class="nav-chevron" aria-hidden="true">{chatHistoryExpanded ? '⌃' : '⌄'}</span>
        </button>
        {#if chatHistoryExpanded}
          <ul class="sub-nav">
            <li>
              <button type="button" class="new-item-btn" on:click={handleNewConversation}>
                <span class="new-item-plus" aria-hidden="true">+</span>
                New chat
              </button>
            </li>
            {#if conversationsLoading}
              <li class="sub-nav-state">Loading…</li>
            {:else if conversationsError}
              <li class="sub-nav-state" style="color:var(--error)">{conversationsError}</li>
            {:else if conversations.length === 0}
              <li class="sub-nav-state">No conversations yet.</li>
            {:else}
              {#each conversations as c (c.conversation_id)}
                {@const isActive = c.conversation_id === $page.params.id}
                <li>
                  {#if confirmDeleteConversationId === c.conversation_id}
                    <div class="delete-confirm">
                      <span class="delete-confirm-text truncate" title={conversationLabel(c)}>
                        Delete "{conversationLabel(c)}"?
                      </span>
                      <button
                        type="button"
                        class="delete-confirm-btn delete-confirm-yes"
                        disabled={deletingConversationId === c.conversation_id}
                        on:click={(e) => confirmDeleteConversation(c, e)}
                      >{deletingConversationId === c.conversation_id ? 'Deleting…' : 'Delete'}</button>
                      <button
                        type="button"
                        class="delete-confirm-btn delete-confirm-no"
                        disabled={deletingConversationId === c.conversation_id}
                        on:click={cancelDeleteConversation}
                      >Cancel</button>
                    </div>
                  {:else}
                    <div class="conversation-row">
                      <a
                        href={`/conversation/${c.conversation_id}`}
                        class="sub-nav-link"
                        class:active={isActive}
                        aria-current={isActive ? 'page' : undefined}
                        title={conversationLabel(c)}
                      >
                        {conversationLabel(c)}
                      </a>
                      <button
                        type="button"
                        class="conversation-row-delete"
                        title="Delete conversation"
                        on:click={(e) => requestDeleteConversation(c.conversation_id, e)}
                      >
                        <svg viewBox="0 0 24 24" width="11" height="11" fill="none"
                             stroke="currentColor" stroke-width="2"
                             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                          <polyline points="3 6 5 6 21 6"/>
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                        </svg>
                      </button>
                    </div>
                  {/if}
                </li>
              {/each}
              {#if deleteConversationError}
                <li class="sub-nav-state" style="color:var(--error)">{deleteConversationError}</li>
              {/if}
            {/if}
          </ul>
        {/if}
      </li>

      <!-- Memory -->
      <li>
        <a
          href="/memory"
          class="nav-link"
          class:active={active.startsWith('/memory')}
          aria-current={active.startsWith('/memory') ? 'page' : undefined}
        >
          <span class="nav-label">Memory</span>
          {#if $pendingCount > 0}
            <span class="badge badge-warning nav-badge">{$pendingCount}</span>
          {/if}
        </a>
      </li>

      <!-- Episodes -->
      <li>
        <a
          href="/episodes"
          class="nav-link"
          class:active={active.startsWith('/episodes')}
          aria-current={active.startsWith('/episodes') ? 'page' : undefined}
        >
          <span class="nav-label">Episodes</span>
        </a>
      </li>

      <!-- Files -->
      <li>
        <button
          type="button"
          class="nav-link"
          class:active={isFilesActive}
          aria-current={isFilesActive ? 'page' : undefined}
          aria-expanded={filesNavExpanded}
          on:click={handleFilesNavClick}
        >
          <span class="nav-label">Files</span>
          <span class="nav-chevron" aria-hidden="true">{filesNavExpanded ? '⌃' : '⌄'}</span>
        </button>
        {#if filesNavExpanded}
          <div class="sub-nav files-sub-nav">
            <!-- Upload -->
            <label class="upload-row" class:uploading>
              <input
                bind:this={fileInputEl}
                type="file"
                multiple
                accept={[...RAW_UPLOAD_EXTENSIONS].join(',')}
                on:change={onFileInput}
                class="sr-only"
              />
              {#if uploading}
                <span class="spinner-sm" aria-hidden="true" />
                <span>Uploading…</span>
              {:else}
                <span class="new-item-plus" aria-hidden="true">+</span>
                <span>Upload file</span>
              {/if}
            </label>
            {#if uploadError}
              <p class="sub-nav-state" style="color:var(--error)">{uploadError}</p>
            {/if}

            {#if $ingest.phase === 'planning' || $ingest.phase === 'streaming'}
              <p class="sub-nav-state ingest-state">
                <span class="spinner-sm accent" aria-hidden="true" />
                {$ingest.phase === 'planning' ? ($ingest.statusMsg || 'Planning…') : `Ingesting — ${$ingest.tokens.length} chunks…`}
              </p>
            {:else if $ingest.phase === 'error'}
              <p class="sub-nav-state" style="color:var(--error)">Ingest failed: {$ingest.error}</p>
            {/if}

            {#each fileGroups as grp (grp.key)}
              {@const expanded = filesExpanded[grp.key] !== false}
              <button type="button" class="file-group-head" on:click={() => toggleFileGroup(grp.key)}>
                <span class="fg-chevron" aria-hidden="true">{expanded ? '⌄' : '⌃'}</span>
                <span class="fg-label">{grp.key}</span>
                <span class="fg-count">{grp.files.length}</span>
              </button>
              {#if expanded}
                {#if grp.loading}
                  <p class="sub-nav-state">Loading…</p>
                {:else if grp.error}
                  <p class="sub-nav-state" style="color:var(--error)">{grp.error}</p>
                {:else if grp.files.length === 0}
                  <p class="sub-nav-state">No files.</p>
                {:else}
                  {#each grp.files as file (file.path)}
                    {@const isSelected = $selectedFile?.path === file.path}
                    {@const isActiveIngest = $ingest.sourceFile?.path === file.path && $isIngesting}
                    <div class="file-row-nested" class:selected={isSelected}>
                      {#if confirmDeletePath === file.path}
                        <div class="delete-confirm">
                          <span class="delete-confirm-text truncate" title={file.name}>
                            Delete "{file.name}"?
                          </span>
                          <button
                            type="button"
                            class="delete-confirm-btn delete-confirm-yes"
                            disabled={deletingPath === file.path}
                            on:click={(e) => confirmDelete(file, e)}
                          >{deletingPath === file.path ? 'Deleting…' : 'Delete'}</button>
                          <button
                            type="button"
                            class="delete-confirm-btn delete-confirm-no"
                            disabled={deletingPath === file.path}
                            on:click={cancelDelete}
                          >Cancel</button>
                        </div>
                      {:else}
                        <button type="button" class="file-row-btn" on:click={() => selectFile(file)}>
                          <span class="file-row-name truncate">{file.name}</span>
                          <span class="file-row-meta">{grp.key} · {formatSize(file.size)}</span>
                        </button>
                        {#if grp.key === 'Raw'}
                          <button
                            type="button"
                            class="file-row-ingest"
                            class:loading={isActiveIngest}
                            disabled={$isIngesting}
                            title="Ingest into the wiki"
                            on:click={(e) => handleIngest(file, e)}
                          >
                            {#if isActiveIngest}
                              <span class="spinner-sm accent" aria-hidden="true" />
                            {:else}
                              ⇧
                            {/if}
                          </button>
                        {/if}
                        <button
                          type="button"
                          class="file-row-delete"
                          title="Delete file"
                          on:click={(e) => requestDelete(file.path, e)}
                        >
                          <svg viewBox="0 0 24 24" width="11" height="11" fill="none"
                               stroke="currentColor" stroke-width="2"
                               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <polyline points="3 6 5 6 21 6"/>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                          </svg>
                        </button>
                      {/if}
                    </div>
                  {/each}
                {/if}
              {/if}
            {/each}
            {#if deleteError}
              <p class="sub-nav-state" style="color:var(--error)">{deleteError}</p>
            {/if}
          </div>
        {/if}
      </li>

      <!-- Settings -->
      <li>
        <a
          href="/settings"
          class="nav-link"
          class:active={active.startsWith('/settings')}
          aria-current={active.startsWith('/settings') ? 'page' : undefined}
        >
          <span class="nav-label">Settings</span>
        </a>
      </li>
    </ul>
  </nav>

  <!-- Footer: version + theme toggle -->
  <div class="sidebar-footer">
    <span class="text-tertiary sf-version">v0.2.0</span>
    <button
      type="button"
      class="switch"
      class:on={$theme === 'light'}
      title="Toggle light/dark"
      aria-label="Toggle light/dark theme"
      on:click={theme.toggle}
    >
      <span class="switch-knob" />
    </button>
  </div>

  {#if !$sidebarCollapsed}
    <!-- svelte-ignore a11y-no-noninteractive-tabindex -->
    <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
    <div
      class="divider"
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize sidebar"
      aria-valuenow={$sidebarWidth}
      aria-valuemin={MIN_WIDTH}
      aria-valuemax={MAX_WIDTH}
      tabindex="0"
      on:mousedown={startResize}
      on:keydown={onDividerKeydown}
    />
  {/if}
</aside>

<style>
  .sidebar {
    grid-column: 1;
    grid-row: 1 / -1;
    display: flex;
    flex-direction: column;
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
    width: 100%;
    min-width: 0;
    height: 100vh;
    overflow: hidden;
    position: relative;
    z-index: 10;
  }

  /* Wordmark */
  .wordmark {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    padding: 0 var(--sp-4);
    height: var(--topbar-h);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    user-select: none;
    white-space: nowrap;
  }

  .brand-mark {
    width: 20px;
    height: 20px;
    flex-shrink: 0;
    color: var(--text-primary);
  }

  .wordmark-text {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--text-primary);
  }

  /* Nav */
  nav {
    flex: 1;
    min-height: 0;
    padding: var(--sp-1) 0;
    overflow-y: auto;
  }

  ul {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 0 var(--sp-2);
  }

  .nav-link {
    position: relative;
    width: 100%;
    display: flex;
    align-items: center;
    gap: var(--sp-3);
    padding: 8px 10px;
    border-radius: var(--radius);
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 400;
    font-family: var(--font-body);
    text-align: left;
    text-decoration: none;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    transition: color var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
  }

  .nav-link:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  .nav-link.active {
    color: var(--accent);
    background: var(--accent-glow);
    font-weight: 500;
  }

  .nav-label { flex: 1; }

  .nav-chevron {
    flex-shrink: 0;
    font-size: 10px;
    color: var(--text-tertiary);
  }

  .nav-badge {
    flex-shrink: 0;
    padding: 1px 6px;
    font-size: 9.5px;
  }

  /* Sub-nav (conversation list / files groups) */
  .sub-nav {
    display: flex;
    flex-direction: column;
    gap: 1px;
    margin: 2px 4px 8px 14px;
    padding-left: 8px;
    border-left: 1px solid var(--border);
  }

  .new-item-btn {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    width: 100%;
    padding: 6px 8px;
    border-radius: 6px;
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 12px;
    font-family: var(--font-body);
    cursor: pointer;
    text-align: left;
    transition: color var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
  }
  .new-item-btn:hover { background: var(--bg-hover); color: var(--text-primary); }

  .new-item-plus {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 4px;
    background: var(--accent-glow);
    color: var(--accent);
    font-size: 12px;
    font-weight: 700;
    flex-shrink: 0;
  }

  .sub-nav-link {
    display: block;
    padding: 6px 8px;
    border-radius: 6px;
    color: var(--text-tertiary);
    font-size: 11.5px;
    text-decoration: none;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: color var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
  }
  .sub-nav-link:hover { color: var(--text-secondary); background: var(--bg-hover); }
  .sub-nav-link.active { color: var(--accent); font-weight: 500; }

  .sub-nav-state {
    padding: 6px 8px;
    font-size: 11px;
    color: var(--text-tertiary);
  }

  /* Chat sub-nav: conversation row + hover-revealed delete button */
  .conversation-row {
    display: flex;
    align-items: center;
    gap: 2px;
  }
  .conversation-row .sub-nav-link { flex: 1; min-width: 0; }

  .conversation-row-delete {
    flex-shrink: 0;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    border-radius: 5px;
    color: var(--text-tertiary);
    cursor: pointer;
    opacity: 0;
    transition: opacity var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
  }
  .conversation-row:hover .conversation-row-delete,
  .conversation-row-delete:focus-visible {
    opacity: 1;
  }
  .conversation-row-delete:hover { color: var(--error); background: var(--error-dim); }

  /* Files sub-nav specifics */
  .files-sub-nav { gap: 1px; }

  .upload-row {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    padding: 6px 8px;
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: color var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
  }
  .upload-row:hover { background: var(--bg-hover); color: var(--text-primary); }
  .upload-row.uploading { opacity: 0.7; cursor: wait; }

  .ingest-state { color: var(--accent); }

  .file-group-head {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    width: 100%;
    background: none;
    border: none;
    padding: 6px 8px;
    border-radius: 6px;
    cursor: pointer;
    font-family: var(--font-body);
    transition: background var(--dur-fast) var(--ease);
  }
  .file-group-head:hover { background: var(--bg-hover); }

  .fg-chevron { font-size: 9px; color: var(--text-tertiary); width: 9px; flex-shrink: 0; }
  .fg-label {
    flex: 1;
    text-align: left;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
  }
  .fg-count { font-size: 9.5px; color: var(--text-tertiary); font-family: var(--font-mono); }

  .file-row-nested {
    display: flex;
    align-items: center;
    gap: 2px;
    padding-left: 18px;
    border-radius: 6px;
    transition: background var(--dur-fast) var(--ease);
  }
  .file-row-nested:hover { background: var(--bg-hover); }
  .file-row-nested.selected { background: var(--accent-glow); }

  .file-row-btn {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 1px;
    background: none;
    border: none;
    text-align: left;
    padding: 5px 6px;
    cursor: pointer;
    color: var(--text-secondary);
  }
  .file-row-nested.selected .file-row-btn { color: var(--accent); }

  .file-row-name { font-size: 12px; color: inherit; }
  .file-row-meta { font-size: 10px; font-family: var(--font-mono); color: var(--text-tertiary); }

  .file-row-ingest {
    flex-shrink: 0;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    border-radius: 5px;
    color: var(--text-tertiary);
    font-size: 11px;
    cursor: pointer;
    margin-right: 4px;
  }
  .file-row-ingest:hover:not(:disabled) { color: var(--accent); background: var(--accent-dim); }
  .file-row-ingest:disabled { opacity: 0.5; cursor: not-allowed; }

  .file-row-delete {
    flex-shrink: 0;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    border-radius: 5px;
    color: var(--text-tertiary);
    cursor: pointer;
    margin-right: 4px;
  }
  .file-row-delete:hover { color: var(--error); background: var(--error-dim); }

  /* Two-step delete confirmation — replaces the row in place */
  .delete-confirm {
    display: flex;
    align-items: center;
    gap: 6px;
    flex: 1;
    min-width: 0;
    padding: 5px 6px;
  }

  .delete-confirm-text {
    flex: 1;
    min-width: 0;
    font-size: 11px;
    color: var(--error);
  }

  .delete-confirm-btn {
    flex-shrink: 0;
    font-size: 10.5px;
    font-weight: 500;
    font-family: var(--font-body);
    padding: 3px 8px;
    border-radius: 5px;
    border: 1px solid transparent;
    cursor: pointer;
    white-space: nowrap;
  }
  .delete-confirm-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .delete-confirm-yes {
    background: var(--error-dim);
    color: var(--error);
    border-color: var(--error-dim);
  }
  .delete-confirm-yes:hover:not(:disabled) { background: color-mix(in srgb, var(--error) 30%, transparent); }

  .delete-confirm-no {
    background: var(--bg-active);
    color: var(--text-secondary);
  }
  .delete-confirm-no:hover:not(:disabled) { background: var(--bg-hover); }

  .spinner-sm {
    display: inline-block;
    width: 10px; height: 10px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    flex-shrink: 0;
  }
  .spinner-sm.accent { border-color: currentColor; border-top-color: transparent; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Footer */
  .sidebar-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-2);
    padding: var(--sp-3) var(--sp-4);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }

  .sf-version {
    font-size: 10px;
    font-family: var(--font-mono);
    white-space: nowrap;
    overflow: hidden;
  }

  /* Resize divider */
  .divider {
    position: absolute;
    top: 0;
    right: -3px;
    width: 6px;
    height: 100%;
    cursor: col-resize;
    z-index: 6;
  }
</style>
