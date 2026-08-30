<script lang="ts">
  import { page } from '$app/stores';
  import ChatPanel from '$lib/components/ChatPanel.svelte';
  import { apiUrl } from '$lib/api';
  import { currentConversationId } from '$lib/stores/conversation';
  import { chatHistoryStore, type Turn } from '$lib/stores/chatHistory';
  import type { Task } from '$lib/stores/tasks';

  interface BackendChatTurn {
    id:                 number;
    task_id:            string;
    role:               string;
    content:            string;
    sources:            Task['sources'];
    status_message:     string | null;
    metadata:           Task['metadata'];
    conversation_id:    string;
    conversation_title: string | null;
    created_at:         number;
  }

  // Fetches this conversation's real history and replaces chatHistoryStore
  // with it. Failure degrades to an empty store (ChatPanel's existing
  // empty-state UI), not a thrown error — a missed history load shouldn't
  // break the page.
  //
  // Known limitation: if the user navigates away mid-stream and back before
  // the task completes, the in-progress streaming state is not reconstructed
  // here — the completed turn simply appears once this fetch runs again.
  async function loadConversationHistory(conversationId: string): Promise<void> {
    try {
      const res = await fetch(
        apiUrl(`/api/chat/history?conversation_id=${encodeURIComponent(conversationId)}&limit=200`)
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: { turns: BackendChatTurn[] } = await res.json();

      // Backend orders newest-first; reverse so the feed renders oldest-first.
      const mapped: Turn[] = data.turns
        .slice()
        .reverse()
        .map((t) => ({
          role:           t.role as Turn['role'],
          content:        t.content,
          task_id:        t.task_id,
          timestamp:      t.created_at * 1000, // seconds → ms
          status:         'complete',           // historical turns are always done streaming
          sources:        t.sources,
          status_message: t.status_message ?? undefined,
          metadata:       t.metadata,
        }));

      chatHistoryStore.set(mapped);
    } catch (err) {
      console.warn('Failed to load conversation history:', err);
    }
  }

  // Keep currentConversationId in sync with the [id] route param, and load
  // that conversation's history — this reactive statement runs on initial
  // mount and again whenever the param changes (Svelte reuses this component
  // instance across /conversation/[id] navigations rather than re-mounting
  // it). chatHistoryStore is cleared immediately, before the fetch, so
  // switching conversations doesn't flash the previous conversation's turns
  // while the new ones are loading.
  $: {
    const id = $page.params.id;
    currentConversationId.set(id);
    chatHistoryStore.set([]);
    loadConversationHistory(id);
  }
</script>

<svelte:head>
  <title>Conversation — Localist</title>
</svelte:head>

<div class="page-inner">
  <ChatPanel />
</div>

<style>
  .page-inner {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
</style>
