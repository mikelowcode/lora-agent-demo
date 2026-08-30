# Localist Wiki Schema

This file defines the structure and conventions for wiki pages in this knowledge base.
Front matter follows Google's OKF (Open Knowledge Framework) convention, as adopted by
LangChain's OpenWiki 0.2.

## Page Types

`type` in front matter must be one of the three values the wiki agent's action schema
enforces: `SYSTEM`, `CONCEPT`, `RESEARCH_NOTE`.

### CONCEPT
A concise explanation of a single idea, term, or technique.

### RESEARCH_NOTE
A synthesis of findings from one or more source documents.

### SYSTEM
A page describing the system itself (architecture, persona, process) rather than a
researched external topic.

## Front Matter (OKF-aligned)

Every page begins with a YAML front-matter block:

```yaml
---
type: RESEARCH_NOTE        # required — one of CONCEPT / RESEARCH_NOTE / SYSTEM
title: Human-readable page title
description: One-line summary of the page's content
resource: original-source-file-or-url
tags: [tag-one, tag-two]
timestamp: 2026-07-21
status: draft               # this project's own field, not part of OKF
created: 2026-07-21          # this project's own field, not part of OKF
updated: 2026-07-21          # this project's own field, not part of OKF
query: original ingestion query text   # this project's own field, not part of OKF
---
```

`type` is the only OKF-required field; `title`/`description`/`resource`/`tags`/`timestamp`
are OKF-optional but always populated by the wiki agent going forward.
`status`/`created`/`updated`/`query` are producer-defined extra fields OKF explicitly
permits alongside its own — kept for this project's own revision-tracking needs.

## Structural Files (auto-generated, not model-authored)

- `index.md` — regenerated after every wiki write; summarizes every page grouped by
  `type`, showing each page's `title` and `description`.
- `logs.md` — appended after every wiki write; a dated changelog of Creation/Update
  entries linking to the affected page(s).

Neither file is ever produced by the wiki agent's own inference — both are pure
functions of the pages already on disk, computed deterministically in
`WikiAgent._finalize()`. Never edit them by hand; they are overwritten/appended to on
the next write, the same way `MEMORY.md` already works for episodic memory.

## Conventions

- All content pages are Markdown files stored flat in `wiki/`.
- Page filenames use kebab-case, e.g. `attention-mechanism.md`.
- Sources are cited inline as `(source: <page-or-file-name>)`.
- `index.md`, `logs.md`, and `MEMORY.md` are structural/generated files, not content
  pages — never a graph node, never RAG-indexed, never a valid diff target.
