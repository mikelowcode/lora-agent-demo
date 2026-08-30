# Third-Party Licenses

Localist's own code is MIT-licensed (see `LICENSE`). This file audits the
licenses of everything it depends on — backend (Python), `localist-mcp`
(Python), `localist-ui` (npm) — plus the model weights the code downloads
at runtime, which are **not** covered by the MIT license above and carry
their own terms.

Audited 2026-08-30 by inspecting each installed package's own metadata
(`pip show`, `pip-licenses --from=mixed`) and, for the frontend, a full
`npm install` + `license-checker` run against `localist-ui`'s actual
resolved dependency tree — not just the top-level `package.json` entries.

## Backend / `localist-mcp` (Python)

All direct and transitive dependencies are permissively licensed
(MIT / BSD / Apache-2.0 / ISC / MPL-2.0 / PSF), **except two**:

| Package | Version | License | Used for |
|---|---|---|---|
| `mlx-embeddings` | 0.1.0 | **GPLv3** | Local embedding inference (the `[mlx]` extra — EmbeddingGemma) |
| `pymupdf` | 1.28.0 | **AGPL-3.0-or-later** (or Artifex commercial license) | PDF text-layer extraction + page rasterization in the local OCR service (`mcp_server/ocr.py`) |

**Status:**
- `mlx-embeddings` (GPLv3) — **kept**, scoped as an optional, separately
  pip-installed extra (`localist[mlx]`), never statically bundled into the
  same distributed artifact as the MIT-licensed core. This is the standard
  pattern for an MIT project with an optional GPL dependency the user opts
  into installing themselves. Revisit before any PyInstaller `.app` build
  (build-order step 7a) — freezing everything into one binary is a
  different distribution shape and may raise different obligations than a
  pip extra does.
- `pymupdf` (AGPL-3.0) — **flagged for replacement, not yet done.** Unlike
  the GPLv3 embeddings dependency, AGPL's network-use clause is a real
  concern for software that could be run as a hosted/multi-tenant service
  (Localist is local-first today, but that's a deployment choice, not a
  license guarantee against a future fork). Tracked as follow-up work:
  replace `pymupdf`'s two roles in `mcp_server/ocr.py`'s `_extract_pdf` —
  text-layer extraction and per-page rasterization for the scanned-PDF OCR
  fallback — with permissively-licensed alternatives (candidates:
  `pypdf`/`pdfminer.six` for text-layer extraction; rasterization needs
  separate research since most drop-in options shell out to Poppler, which
  carries its own GPL terms as a system binary, not a linked dependency —
  a different legal shape than PyMuPDF's, but worth confirming rather than
  assuming). Not resolved as of this audit.

`sentencepiece` (0.2.1) reports no license classifier in package metadata
(`UNKNOWN`) — its actual license is Apache-2.0, publicly documented at
https://github.com/google/sentencepiece; metadata gap only, not a real
concern.

Full audited list (via `pip-licenses --from=mixed` against a clean
`backend/.venv`):

| Name | Version | License |
|---|---|---|
| Jinja2 | 3.1.6 | BSD License |
| MarkupSafe | 3.0.3 | BSD-3-Clause |
| PyJWT | 2.13.0 | MIT |
| PyYAML | 6.0.3 | MIT License |
| Pygments | 2.20.0 | BSD-2-Clause |
| aiohappyeyeballs | 2.6.2 | Python Software Foundation License |
| aiohttp | 3.14.1 | Apache-2.0 AND MIT |
| aiosignal | 1.4.0 | Apache Software License |
| annotated-doc | 0.0.4 | MIT |
| annotated-types | 0.7.0 | MIT License |
| anyio | 4.13.0 | MIT |
| attrs | 26.1.0 | MIT |
| certifi | 2026.5.20 | Mozilla Public License 2.0 (MPL 2.0) |
| cffi | 2.0.0 | MIT |
| chardet | 7.4.3 | 0BSD |
| charset-normalizer | 3.4.7 | MIT |
| click | 8.4.1 | BSD-3-Clause |
| contourpy | 1.3.3 | BSD License |
| cryptography | 49.0.0 | Apache-2.0 OR BSD-3-Clause |
| cssselect | 1.4.0 | BSD-3-Clause |
| cycler | 0.12.1 | BSD License |
| datasets | 5.0.0 | Apache Software License |
| dill | 0.4.1 | BSD License |
| fastapi | 0.136.3 | MIT |
| filelock | 3.29.3 | MIT |
| fonttools | 4.63.0 | MIT |
| frozenlist | 1.8.0 | Apache-2.0 |
| fsspec | 2026.4.0 | BSD-3-Clause |
| h11 | 0.16.0 | MIT License |
| hf-xet | 1.5.1 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httptools | 0.8.0 | MIT |
| httpx | 0.28.1 | BSD License |
| httpx-sse | 0.4.3 | MIT |
| huggingface_hub | 1.18.0 | Apache Software License |
| idna | 3.17 | BSD-3-Clause |
| iniconfig | 2.3.0 | MIT |
| jsonschema | 4.26.0 | MIT |
| jsonschema-specifications | 2025.9.1 | MIT |
| kiwisolver | 1.5.0 | BSD License |
| llguidance | 1.7.6 | MIT |
| lxml | 6.1.1 | BSD-3-Clause |
| lxml_html_clean | 0.4.5 | BSD-3-Clause |
| markdown-it-py | 4.2.0 | MIT License |
| matplotlib | 3.11.1 | Python Software Foundation License |
| mcp | 1.28.1 | MIT License |
| mdurl | 0.1.2 | MIT License |
| miniaudio | 1.71 | MIT License |
| mlx | 0.31.2 | MIT |
| mlx-audio | 0.4.4 | MIT |
| **mlx-embeddings** | 0.1.0 | **GPLv3 — see note above** |
| mlx-lm | 0.31.3 | MIT |
| mlx-metal | 0.31.2 | MIT |
| mlx-vlm | 0.6.3 | MIT License |
| multidict | 6.7.1 | Apache License 2.0 |
| multiprocess | 0.70.19 | BSD License |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| opencv-python | 4.13.0.92 | Apache Software License |
| packaging | 26.2 | Apache-2.0 OR BSD-2-Clause |
| pandas | 3.0.3 | BSD License |
| pillow | 12.2.0 | MIT-CMU |
| pluggy | 1.6.0 | MIT License |
| propcache | 0.5.2 | Apache Software License |
| protobuf | 7.35.0 | 3-Clause BSD License |
| psutil | 7.2.2 | BSD-3-Clause |
| pyarrow | 24.0.0 | Apache-2.0 |
| pycparser | 3.0 | BSD-3-Clause |
| pydantic | 2.13.4 | MIT |
| pydantic-settings | 2.14.1 | MIT |
| pydantic_core | 2.46.4 | MIT |
| **pymupdf** | 1.28.0 | **Dual: AGPL-3.0 or Artifex Commercial — see note above** |
| pyobjc-core | 12.2.1 | MIT |
| pyobjc-framework-Cocoa | 12.2.1 | MIT |
| pyobjc-framework-CoreML | 12.2.1 | MIT |
| pyobjc-framework-Quartz | 12.2.1 | MIT |
| pyobjc-framework-Vision | 12.2.1 | MIT |
| pyparsing | 3.3.2 | MIT |
| pytest | 9.1.1 | MIT |
| python-dateutil | 2.9.0.post0 | Apache Software License; BSD License |
| python-dotenv | 1.2.2 | BSD-3-Clause |
| python-multipart | 0.0.32 | Apache-2.0 |
| readability-lxml | 0.8.4.1 | Apache License 2.0 |
| referencing | 0.37.0 | MIT |
| regex | 2026.5.9 | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | Apache Software License |
| rich | 15.0.0 | MIT License |
| rpds-py | 2026.6.3 | MIT |
| safetensors | 0.8.0 | Apache Software License |
| scipy | 1.17.1 | BSD License |
| sentencepiece | 0.2.1 | Apache-2.0 (see note above re: metadata gap) |
| shellingham | 1.5.4 | ISC License (ISCL) |
| six | 1.17.0 | MIT License |
| sounddevice | 0.5.5 | MIT |
| sse-starlette | 3.4.5 | BSD-3-Clause |
| starlette | 1.2.0 | BSD-3-Clause |
| tokenizers | 0.22.2 | Apache Software License |
| tqdm | 4.68.2 | MPL-2.0 AND MIT |
| transformers | 5.11.0 | Apache 2.0 License |
| typer | 0.25.1 | MIT |
| typing-inspection | 0.4.2 | MIT |
| typing_extensions | 4.15.0 | PSF-2.0 |
| urllib3 | 2.7.0 | MIT |
| uvicorn | 0.48.0 | BSD-3-Clause |
| uvloop | 0.22.1 | Apache Software License; MIT License |
| watchfiles | 1.2.0 | MIT License |
| websockets | 16.0 | BSD-3-Clause |
| xxhash | 3.7.0 | BSD License |
| yarl | 1.24.2 | Apache-2.0 |

Some rows above (`datasets`, `pyarrow`, `opencv-python`, `scipy`, etc.) are
transitive dependencies pulled in by `mlx-embeddings`/`transformers`, not
direct requirements — included here for completeness since they ship into
the same environment.

## Frontend (`localist-ui`, npm)

98 production dependencies audited via `license-checker` against a real,
freshly-installed `node_modules` (not just declared `package.json` entries).
All permissive:

| License | Count |
|---|---|
| MIT | 77 |
| ISC | 13 |
| Apache-2.0 | 4 |
| CC0-1.0 | 1 |
| BSD-3-Clause | 1 |
| 0BSD | 1 |

No copyleft dependencies on the frontend.

## Model weights — not covered by the MIT license

Localist's code is MIT-licensed; the model weights it downloads and runs
are separate works under their own terms, chosen at install/runtime by
whichever backend and models the user configures:

- **`mlx-community/embeddinggemma-300m-4bit`** (~400MB) — downloaded
  automatically by `EmbeddingEngine` on first startup when the `[mlx]`
  extra is active (Apple Silicon only). Built on Google's Gemma, under the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — **not MIT**,
  and not OSI-approved open source (it's a permissive-but-custom license
  with acceptable-use terms). This needs its own explicit line in the
  README so a user installing `localist[mlx]` knows the code they're
  running is MIT but the weights it fetches are under Google's terms, not
  Localist's.
- Chat models (oMLX/Ollama/Foundry) and any Ollama-served embedding model
  (e.g. `nomic-embed-text`) are entirely user-chosen and pulled by the
  user's own runtime — never bundled or shipped by this repo — so they
  carry whatever license the user's chosen model/provider sets, and don't
  need a line here beyond a general disclaimer that Localist ships no
  model weights of its own except the one EmbeddingGemma download above.

## Open items

- Replace `pymupdf` (AGPL-3.0) with a permissively-licensed alternative for
  PDF text-layer extraction and rasterization — see note above. Not done.
- Add the EmbeddingGemma/Gemma-license callout to the README (currently
  documents the download at line ~58 but doesn't name the license terms).
- Trademark/fork-naming note for "Localist" — open decision, unresolved,
  low priority (see project scoping doc §12).
