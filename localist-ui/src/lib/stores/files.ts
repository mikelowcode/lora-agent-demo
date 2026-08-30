/**
 * files.ts — Localist file management store
 *
 * Owns all state and async operations for the /files page:
 *   rawFiles / wikiFiles   — listing state
 *   uploadFile()           — POST /files/upload
 *   ingestFile()           — POST /task/stream (streams progress, then navigates)
 *
 * injectCompletedTask() is exported separately so FileBrowser can push a
 * completed ingest result into tasksStore before navigating to /conversation.
 * It works against the real tasksStore shape from tasks.ts.
 */

import { writable, derived, get } from 'svelte/store';
import { goto } from '$app/navigation';
import { tasksStore } from './tasks';
import type { Task } from './tasks';
import { currentConversationId } from './conversation';
import { apiUrl } from '$lib/api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface FileEntry {
  name:     string;   // stem without extension
  filename: string;   // filename with extension, e.g. "my-doc.md"
  path:     string;   // absolute path — used directly as context.raw_path
  size:     number;   // bytes
  modified: string;   // ISO-8601 UTC
  type:     'raw' | 'wiki' | 'generated';
}

export type IngestPhase = 'idle' | 'planning' | 'streaming' | 'done' | 'error';

export interface IngestState {
  phase:      IngestPhase;
  taskId:     string | null;
  statusMsg:  string;
  tokens:     string[];       // raw chunks, mirrors Task.tokens shape
  error:      string | null;
  sourceFile: FileEntry | null;
}

// ---------------------------------------------------------------------------
// Stores
// ---------------------------------------------------------------------------

export const rawFiles       = writable<FileEntry[]>([]);
export const wikiFiles      = writable<FileEntry[]>([]);
export const generatedFiles = writable<FileEntry[]>([]);

export const rawLoading       = writable(false);
export const wikiLoading      = writable(false);
export const generatedLoading = writable(false);

export const rawError       = writable<string | null>(null);
export const wikiError      = writable<string | null>(null);
export const generatedError = writable<string | null>(null);

export const ingest = writable<IngestState>({
  phase:      'idle',
  taskId:     null,
  statusMsg:  '',
  tokens:     [],
  error:      null,
  sourceFile: null,
});

export const isIngesting = derived(
  ingest,
  ($i) => $i.phase === 'planning' || $i.phase === 'streaming',
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function formatBytes(bytes: number): string {
  if (bytes < 1024)         return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Push a completed ingest task directly into tasksStore so the conversation
 * page renders it immediately on navigation.
 *
 * Constructs a full Task object matching the shape in tasks.ts, marks it
 * complete, and sets it as the active task.
 */
export function injectCompletedTask(
  task_id:     string,
  instruction: string,
  tokens:      string[],
  sources:     Task['sources'],
): void {
  const now = Date.now();
  const task: Task = {
    task_id,
    instruction,
    status:         'complete',
    status_message: 'Ingest complete.',
    tokens,
    answer:         tokens.join(''),
    sources,
    started_at:     now,
    completed_at:   now,
    source:         'ingest',
  };

  tasksStore.update((s) => ({
    ...s,
    active_task_id: task_id,
    streaming:      false,
    tasks: { ...s.tasks, [task_id]: task },
  }));
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

export async function loadRawFiles(): Promise<void> {
  rawLoading.set(true);
  rawError.set(null);
  try {
    const res = await fetch(apiUrl('/api/files/raw'));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: { files: FileEntry[] } = await res.json();
    rawFiles.set(data.files);
  } catch (err) {
    rawError.set(err instanceof Error ? err.message : String(err));
  } finally {
    rawLoading.set(false);
  }
}

export async function loadWikiFiles(): Promise<void> {
  wikiLoading.set(true);
  wikiError.set(null);
  try {
    const res = await fetch(apiUrl('/api/files/wiki'));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: { files: FileEntry[] } = await res.json();
    wikiFiles.set(data.files);
  } catch (err) {
    wikiError.set(err instanceof Error ? err.message : String(err));
  } finally {
    wikiLoading.set(false);
  }
}

export async function loadGeneratedFiles(): Promise<void> {
  generatedLoading.set(true);
  generatedError.set(null);
  try {
    const res = await fetch(apiUrl('/api/files/generated'));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: { files: FileEntry[] } = await res.json();
    generatedFiles.set(data.files);
  } catch (err) {
    generatedError.set(err instanceof Error ? err.message : String(err));
  } finally {
    generatedLoading.set(false);
  }
}

/**
 * Upload a File to POST /api/files/upload.
 * Refreshes the raw listing on success. Throws on failure.
 */
export async function uploadFile(file: File): Promise<FileEntry> {
  const form = new FormData();
  // FastAPI expects the field name "file" (singular) — UploadFile = File(...)
  form.append('file', file);

  const res = await fetch(apiUrl('/api/files/upload'), { method: 'POST', body: form });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { detail?: string }).detail ?? `Upload failed: HTTP ${res.status}`);
  }

  const entry: FileEntry = await res.json();
  await loadRawFiles();
  return entry;
}

/**
 * Save arbitrary content (e.g. an edited chat turn) to POST /api/files/generated.
 *
 * Never overwrites an existing file — the backend auto-versions on a name
 * collision (name_2.ext, ...), so the returned FileEntry.filename may differ
 * from the requested filename+extension. Refreshes the generated listing on
 * success. Throws on failure.
 */
export async function saveGeneratedFile(
  filename:  string,
  extension: 'txt' | 'md',
  content:   string,
): Promise<FileEntry> {
  const res = await fetch(apiUrl('/api/files/generated'), {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, extension, content }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { detail?: string }).detail ?? `Save failed: HTTP ${res.status}`);
  }

  const entry: FileEntry = await res.json();
  await loadGeneratedFiles();
  return entry;
}

/**
 * Ingest a raw file via POST /api/task/stream.
 *
 * Streams SSE progress into the `ingest` store.
 * On success: injects the result into tasksStore, then navigates to
 * /conversation so the user lands on a populated conversation view.
 */
export async function ingestFile(entry: FileEntry): Promise<void> {
  const task_id = crypto.randomUUID();
  const instruction = `Ingest the file "${entry.filename}" into the wiki and produce a research note.`;

  ingest.set({
    phase:      'planning',
    taskId:     task_id,
    statusMsg:  'Sending to agent…',
    tokens:     [],
    error:      null,
    sourceFile: entry,
  });

  let response: Response;
  try {
    response = await fetch(apiUrl('/api/task/stream'), {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        task_id,
        instruction,
        context: {
          raw_path:   entry.path,
          auto_apply: true,
        },
        conversation_id: get(currentConversationId),
      }),
    });
  } catch (err) {
    ingest.update((s) => ({
      ...s,
      phase: 'error',
      error: err instanceof Error ? err.message : String(err),
    }));
    return;
  }

  if (!response.ok || !response.body) {
    ingest.update((s) => ({
      ...s,
      phase: 'error',
      error: `Stream request failed: HTTP ${response.status}`,
    }));
    return;
  }

  const reader  = response.body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = '';
  // Accumulate sources separately — they arrive before "done".
  let   pendingSources: Task['sources'] = [];

  outer: while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed === 'data: [DONE]') break outer;
      if (!trimmed.startsWith('data:')) continue;

      let event: Record<string, unknown>;
      try {
        event = JSON.parse(trimmed.slice(5).trim());
      } catch {
        continue;
      }

      switch (event.type as string) {
        case 'status':
          ingest.update((s) => ({
            ...s,
            phase:     'planning',
            statusMsg: String(event.message ?? ''),
          }));
          break;

        case 'token':
          ingest.update((s) => ({
            ...s,
            phase:  'streaming',
            tokens: [...s.tokens, String(event.token ?? '')],
          }));
          break;

        case 'sources':
          pendingSources = (event.sources as Task['sources']) ?? [];
          break;

        case 'done': {
          const currentTokens = get(ingest).tokens;

          // Pre-populate the conversation store before navigation.
          injectCompletedTask(task_id, instruction, currentTokens, pendingSources);

          ingest.update((s) => ({ ...s, phase: 'done' }));

          // Refresh wiki listing — a new page was likely created.
          loadWikiFiles();

          // Do not call goto() here — break out of the loop and let the
          // post-loop block handle navigation. This avoids a race where
          // [DONE] and the done event arrive in the same read chunk and
          // the break outer on [DONE] fires before goto() is reached.
          break outer;
        }

        case 'error':
          ingest.update((s) => ({
            ...s,
            phase: 'error',
            error: String(event.message ?? 'Unknown error'),
          }));
          break outer;
      }
    }
  }

  // Navigate or surface error based on final phase.
  const current = get(ingest);
  if (current.phase === 'done') {
    goto('/conversation');
  } else if (current.phase !== 'error') {
    ingest.update((s) => ({
      ...s,
      phase: 'error',
      error: 'Stream ended without a completion event.',
    }));
  }
}

export function resetIngest(): void {
  ingest.set({
    phase:      'idle',
    taskId:     null,
    statusMsg:  '',
    tokens:     [],
    error:      null,
    sourceFile: null,
  });
}
