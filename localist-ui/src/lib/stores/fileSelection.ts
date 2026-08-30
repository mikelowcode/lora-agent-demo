/**
 * fileSelection.ts — currently-previewed file, shared between Sidebar
 * (which now owns the Wiki/Raw/Generated listing + selection UI) and
 * FileBrowser.svelte (which renders only the full-width preview pane).
 */

import { writable, get } from 'svelte/store';
import type { FileEntry } from './files';
import { isOcrExtension } from '$lib/utils/ocr';
import { apiUrl } from '$lib/api';

export const selectedFile = writable<FileEntry | null>(null);
export const fileContent = writable<string | null>(null);
export const fileContentLoading = writable(false);
export const fileContentError = writable<string | null>(null);
// Set instead of fetching /files/content for OCR-eligible (image/PDF) raw
// files — that endpoint reads as UTF-8 text and 500s on binary content.
export const filePreviewUnavailable = writable(false);

export async function selectFile(file: FileEntry): Promise<void> {
  // Toggle off if re-selecting the already-open file.
  if (
    get(selectedFile)?.path === file.path &&
    (get(fileContent) !== null || get(filePreviewUnavailable))
  ) {
    closeFile();
    return;
  }

  selectedFile.set(file);
  fileContent.set(null);
  fileContentError.set(null);
  filePreviewUnavailable.set(false);

  if (isOcrExtension(file.filename)) {
    filePreviewUnavailable.set(true);
    return;
  }

  fileContentLoading.set(true);

  try {
    const res = await fetch(apiUrl(`/api/files/content?path=${encodeURIComponent(file.path)}`));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: { path: string; content: string } = await res.json();
    fileContent.set(data.content);
  } catch (err) {
    fileContentError.set(err instanceof Error ? err.message : String(err));
  } finally {
    fileContentLoading.set(false);
  }
}

export function closeFile(): void {
  selectedFile.set(null);
  fileContent.set(null);
  fileContentError.set(null);
  filePreviewUnavailable.set(false);
}
