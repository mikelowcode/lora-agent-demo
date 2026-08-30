"""
Localist — OllamaRuntimeClient
===========================
Concrete implementation of BaseRuntimeClient for a local Ollama server.

Layer placement
---------------
  ControllerAgent / Sub-agents  →  OllamaRuntimeClient  →  Ollama HTTP API
                                                            (local only)

Architectural contract
----------------------
- Implements BaseRuntimeClient (base_runtime_client.py).
- No FastAPI imports.  No agent logic.  Pure transport + normalisation.
- Normalises Ollama's response shapes to the same types that
  FoundryRuntimeClient / OMLXRuntimeClient return, so the Controller and
  agents are completely unaware of which backend is active.
- All network/decode errors are caught and re-raised as RuntimeError —
  the same contract as FoundryRuntimeClient / OMLXRuntimeClient.

Ollama integration notes
--------------------------
Ollama exposes its own native HTTP API (not OpenAI-compatible), so the
transport code here is structurally different from Foundry/oMLX in two
places:

  - Model listing: GET /api/tags returns {"models": [{"model": "...", ...}]}
    — a "models" list keyed by "model" (plus a duplicate "name" field) —
    not the OpenAI-style {"data": [{"id": "..."}]} shape.
  - Streaming: POST /api/chat with "stream": true returns NDJSON (one raw
    JSON object per line), not an SSE "data: {...}" envelope.  Each line
    carries the next content delta at message.content, and the final line
    is marked "done": true.  This is NOT the same wire format as Foundry's
    SSE stream, so _iter_sse_chunks is not reusable here — this module
    implements its own line-delimited JSON parser (_iter_ndjson_chunks).

embed() calls Ollama's native POST /api/embed endpoint, which is also
structurally different from Foundry/oMLX:

  - Response shape: {"embeddings": [[...]], ...} — a plural "embeddings"
    key holding a list of vectors (batch-shaped), not the OpenAI-style
    {"data": [{"embedding": [...]}]} shape Foundry/oMLX use. This client
    only ever sends a single string as "input", so it extracts index 0
    of that list.
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Iterator

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Confirmed base URL — GET /api/tags returns HTTP 200 at this address.
# Override via the base_url constructor argument.
_DEFAULT_BASE_URL = "http://localhost:11434"

_CHAT_PATH  = "/api/chat"
_TAGS_PATH  = "/api/tags"
_EMBED_PATH = "/api/embed"

# No default chat model — Ollama serves many models of wildly different size
# (multi-GB local pulls vs. lightweight cloud models), so silently falling
# back to one is never safe. Empty string means "not configured"; the
# constructor raises ValueError rather than assuming a model.
DEFAULT_CHAT_MODEL = ""

# Ollama Cloud models are proxied through the *same* local daemon
# (base_url stays http://localhost:11434 either way — confirmed live,
# see docs/architecture/16-runtime-backend-layer.md §16.4's
# "gemma4:31b-cloud, proxied through ollama.com — never resident locally"
# verification) — so base_url can never distinguish local from cloud here.
# The model tag itself is the only signal: cloud-hosted models carry a
# "-cloud" suffix on their tag (e.g. "gemma4:31b-cloud"), local pulls don't.
_CLOUD_TAG_SUFFIX = "-cloud"


def _is_cloud_model(model: str) -> bool:
    """True if `model`'s tag (the part after ':') ends in "-cloud"."""
    tag = model.split(":", 1)[1] if ":" in model else model
    return tag.endswith(_CLOUD_TAG_SUFFIX)


# ---------------------------------------------------------------------------
# NDJSON streaming helper
# ---------------------------------------------------------------------------

def _iter_ndjson_chunks(
    response: requests.Response,
    meta:     dict | None = None,
) -> Iterator[str]:
    """
    Yield text delta strings from an Ollama NDJSON chat stream.

    Each line from the stream is a standalone JSON object (not an SSE
    envelope), e.g.:
        {"model": "...", "message": {"role": "assistant", "content": "hi"}, "done": false}

    The stream ends on the line where "done" is true (which may also carry
    a trailing empty/absent content) — the loop stops there rather than
    waiting for connection close.

    Malformed (non-JSON) lines are silently skipped, as are well-formed
    lines that carry neither "message" content nor "error" nor "done"
    (e.g. a pure metadata line) — those are legitimate NDJSON shapes, not
    errors.

    meta:
        Optional out-param (2026-08-04). When given, the terminal "done"
        line's diagnostic fields — done_reason, eval_count,
        prompt_eval_count, total_duration — are written into it in place,
        keyed by those exact names, each defaulting to None if Ollama
        omitted it. Previously discarded entirely: only `content` and
        `done` were ever read off that line, so a zero-content completion
        left no trace of *why* (was a token actually generated and empty,
        or was generation never attempted?). See §4.6.2's 2026-07-30
        recurrence note in docs/architecture/04-planner-routing-model.md —
        this exists to make the next occurrence diagnosable from the log
        instead of reconstructed after the fact. None (default) preserves
        the old behavior exactly; callers that don't pass it see no
        change.

    2026-07-17: two failure modes were previously silent. Ollama sends
    {"error": "..."} instead of a normal content chunk mid-stream on a
    rate limit, context-length overflow, moderation block, or a
    mid-generation crash — that line has no "message" key, so content
    resolved to "" and was skipped via `if content:` same as any other
    empty delta, and if the connection then closed without ever sending
    "done": true, the generator just finished with zero chunks yielded and
    no exception, indistinguishable from a genuine empty completion.
    Confirmed live (repeated output_chars=0 completions, task still marked
    COMPLETE, during 2026-07-16 research-loop testing) before this fix.
    Both are now surfaced as RuntimeError instead of resolving silently.

    Raises
    ------
    RuntimeError
        If a line carries a truthy "error" field, or if the stream
        (response.iter_lines()) is exhausted without ever seeing a line
        where "done" is true.
    """
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue

        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError:
            logger.debug("NDJSON: skipping non-JSON line: %s", raw_line[:120])
            continue

        error = data.get("error")
        if error:
            raise RuntimeError(f"Ollama returned an error mid-stream: {error}")

        content = data.get("message", {}).get("content", "")
        if content:
            yield content

        if data.get("done"):
            if meta is not None:
                meta.update({
                    "done_reason":        data.get("done_reason"),
                    "eval_count":         data.get("eval_count"),
                    "prompt_eval_count":  data.get("prompt_eval_count"),
                    "total_duration":     data.get("total_duration"),
                })
            return

    raise RuntimeError(
        "Ollama NDJSON stream ended without a \"done\": true line — "
        "incomplete or truncated response."
    )


# ---------------------------------------------------------------------------
# OllamaRuntimeClient
# ---------------------------------------------------------------------------

class OllamaRuntimeClient:
    """
    Concrete BaseRuntimeClient for a local Ollama inference backend.

    Satisfies the BaseRuntimeClient Protocol:
        def infer(self, prompt, system, max_tokens, temperature) -> str
        def embed(self, text) -> list[float]
        def infer_stream(self, prompt, system, max_tokens, temperature)
               -> Generator[str, None, None]

    Parameters
    ----------
    chat_model:
        Ollama model name for chat completions. Required — there is no
        default. Construction raises ValueError if left empty, since
        silently falling back to some specific local model would risk
        pulling in a multi-gigabyte model the caller never asked for.
    embedding_model:
        Ollama model name for embedding generation. Empty string (default)
        means no embedding model is configured — embed() raises
        NotImplementedError until one is set.
    base_url:
        Base URL of the Ollama HTTP server.  Defaults to _DEFAULT_BASE_URL.
    request_timeout:
        Seconds before a non-streaming request is considered hung.
    stream_timeout:
        Seconds before the first byte of a streaming response must arrive.

    Raises
    ------
    ValueError
        If chat_model is empty — no default chat model is assumed.
    """

    def __init__(
        self,
        chat_model:      str   = DEFAULT_CHAT_MODEL,
        embedding_model: str   = "",
        base_url:        str   = _DEFAULT_BASE_URL,
        request_timeout: float = 30.0,
        stream_timeout:  float = 60.0,
    ) -> None:
        if not chat_model:
            raise ValueError(
                "OllamaRuntimeClient requires an explicit chat_model — no default "
                "is assumed. Set LOCALIST_CHAT_MODEL (or pass chat_model explicitly) "
                "to a model name confirmed present via GET /api/tags."
            )

        self._chat_model      = chat_model
        self._embedding_model = embedding_model
        self._base_url        = base_url.rstrip("/")
        self._request_timeout = request_timeout
        self._stream_timeout  = stream_timeout

        self.is_local = not _is_cloud_model(chat_model)

        self._chat_endpoint  = self._base_url + _CHAT_PATH
        self._tags_endpoint  = self._base_url + _TAGS_PATH
        self._embed_endpoint = self._base_url + _EMBED_PATH

        logger.info(
            "OllamaRuntimeClient initialised — chat: %s  embed: %s  base: %s  "
            "is_local: %s",
            self._chat_model,
            self._embedding_model,
            self._base_url,
            self.is_local,
        )

    # -----------------------------------------------------------------------
    # BaseRuntimeClient interface
    # -----------------------------------------------------------------------

    def infer(
        self,
        prompt:      str,
        system:      str   = "",
        max_tokens:  int   = 1024,
        temperature: float = 0.2,
        label:       str   = "",
        timeout:     float | None = None,
    ) -> str:
        """
        Request a blocking chat completion from Ollama via NDJSON accumulation.

        Internally calls infer_stream() and joins all chunks — this ensures
        the streaming and non-streaming paths use exactly the same transport
        code and model parameters, preventing subtle behavioural divergence.

        Parameters
        ----------
        label:
            Optional caller identifier forwarded to infer_stream() for
            diagnostic correlation. Purely diagnostic — has no effect on
            the request itself.
        timeout:
            Optional per-call override, in seconds, forwarded to
            infer_stream(). None (default) uses self._stream_timeout.

        Returns
        -------
        str
            The fully accumulated model response.

        Raises
        ------
        RuntimeError
            On any network error, non-200 status, NDJSON decode failure, a
            mid-stream {"error": ...} line from Ollama, or a stream that
            closes without ever sending "done": true (see
            _iter_ndjson_chunks).
        """
        chunks = list(self.infer_stream(
            prompt      = prompt,
            system      = system,
            max_tokens  = max_tokens,
            temperature = temperature,
            label       = label,
            timeout     = timeout,
        ))
        result = "".join(chunks)
        logger.debug("infer() ← %d chars received.", len(result))
        return result

    def embed(self, text: str) -> list[float]:
        """
        Request a dense embedding vector from Ollama via POST /api/embed.

        Raises NotImplementedError when no embedding model is configured —
        this Ollama client is currently chat-only. Set embedding_model at
        construction time when an embedding model becomes available.

        Parameters
        ----------
        text:
            The input string to embed.

        Returns
        -------
        list[float]
            The embedding vector for the input.

        Raises
        ------
        NotImplementedError
            When embedding_model is empty (no model loaded).
        RuntimeError
            On any network error, non-200 status, or unexpected response shape.
        """
        if not self._embedding_model:
            raise NotImplementedError(
                "No embedding model configured for OllamaRuntimeClient. "
                "Set embedding_model to a valid model ID when one is available. "
                "Ensure ResearchAgent is called with use_embeddings=False (the default) "
                "until then."
            )

        payload = {
            "model": self._embedding_model,
            "input": text,
        }

        logger.debug("embed() → %s  input_chars=%d", self._embed_endpoint, len(text))

        try:
            response = requests.post(
                self._embed_endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=self._request_timeout,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._embed_endpoint}. "
                f"Is the service running?  Detail: {exc}"
            ) from exc
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama embed request timed out after {self._request_timeout}s."
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama embed returned HTTP {response.status_code}: "
                f"{response.text[:400]}"
            )

        try:
            data: dict = response.json()
            # Ollama's native embed response shape:
            # {"embeddings": [[...]], ...} — plural key, list of vectors.
            vector: list[float] = data["embeddings"][0]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unexpected Ollama embed response shape: {exc}\n"
                f"Raw: {response.text[:400]}"
            ) from exc

        logger.debug("embed() ← vector dim=%d", len(vector))
        return vector

    def infer_stream(
        self,
        prompt:      str,
        system:      str   = "",
        max_tokens:  int   = 1024,
        temperature: float = 0.2,
        label:       str   = "",
        timeout:     float | None = None,
    ) -> Generator[str, None, None]:
        """
        Request a streaming chat completion from Ollama.

        Yields individual text chunks from the NDJSON stream as they arrive.
        The FastAPI streaming endpoint consumes this generator and relays
        chunks to the Svelte UI via Server-Sent Events.

        Parameters
        ----------
        label:
            Optional caller identifier for diagnostic correlation. Has no
            effect on the request itself.
        timeout:
            Optional per-call override for the request timeout, in
            seconds. None (default) uses self._stream_timeout, exactly as
            before this parameter was added (2026-07-17). Pass a smaller
            value for cheap, small-max_tokens calls (classifiers, gate
            checks) that should fail fast rather than share the full
            main-dispatch budget — see mcp_tool_dispatcher.py's
            _RESEARCH_CLASSIFIER_TIMEOUT for the motivating incident.

        Yields
        ------
        str
            One text chunk per iteration.

        Raises
        ------
        RuntimeError
            On any network error, non-200 status, NDJSON decode failure, a
            mid-stream {"error": ...} line from Ollama, or a stream that
            closes without ever sending "done": true (see
            _iter_ndjson_chunks).
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":    self._chat_model,
            "messages": messages,
            "stream":   True,
            # Reasoning-capable models (confirmed live: gemma4:e4b-mlx) emit
            # a separate "thinking" field this client has never read, and
            # that field draws from the same num_predict budget as
            # "content" — a verbose reasoning trace can consume the entire
            # budget before any content token is ever generated, producing
            # a well-formed "done": true stream with zero content
            # (done_reason="length", eval_count==num_predict). Confirmed
            # 100% reproducible (24/24) in diagnostics/
            # diag_ollama_zero_content_repro.py's Finding A — see §4.6.2 of
            # docs/architecture/04-planner-routing-model.md. think=False
            # confirmed live to eliminate the thinking field entirely and
            # return real content immediately; confirmed harmless (HTTP 200
            # no-op) against gemma4:31b-cloud, which never used thinking.
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                # No "num_ctx" here (deliberately). diagnostics/reports/
                # ollama_cloud_num_ctx_findings.md confirmed it has zero
                # observed effect on Ollama, local or cloud, across a 25x
                # sweep of values — the real enforced ceiling is always
                # each model's hardcoded native context length. Sending a
                # dead parameter here would misleadingly suggest this
                # client controls the context window; it doesn't.
            },
        }

        effective_timeout = timeout if timeout is not None else self._stream_timeout

        logger.debug(
            "infer_stream() → %s  max_tokens=%d  temp=%.2f  prompt_chars=%d  "
            "label=%s  timeout=%.1fs",
            self._chat_endpoint, max_tokens, temperature, len(prompt), label,
            effective_timeout,
        )

        try:
            response = requests.post(
                self._chat_endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=True,
                timeout=effective_timeout,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._chat_endpoint}. "
                f"Is the service running?  Detail: {exc}"
            ) from exc
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama did not respond within {effective_timeout}s "
                f"(endpoint: {self._chat_endpoint})."
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code} "
                f"from {self._chat_endpoint}: {response.text[:400]}"
            )

        stream_meta:  dict = {}
        total_chars = 0
        try:
            for chunk in _iter_ndjson_chunks(response, meta=stream_meta):
                total_chars += len(chunk)
                yield chunk
        except Exception as exc:
            raise RuntimeError(
                f"Error reading Ollama NDJSON stream: {exc}"
            ) from exc

        if total_chars == 0:
            # Passive instrumentation added 2026-08-04 — see §4.6.2's
            # 2026-07-30 recurrence note. Escalated from the old bare
            # debug-level "infer() ← 0 chars received." to a warning
            # carrying everything needed to diagnose the next occurrence
            # without reconstructing it after the fact: whether Ollama
            # actually generated tokens (eval_count) and why it stopped
            # (done_reason), alongside the call's own params and the full
            # prompt. Does not change behavior — ControllerAgent's
            # retry-then-fallback guard still owns recovery.
            logger.warning(
                "OllamaRuntimeClient: zero-content completion — model=%s "
                "done_reason=%s eval_count=%s prompt_eval_count=%s "
                "total_duration_ns=%s max_tokens=%d temperature=%.2f "
                "prompt_chars=%d label=%s\nfull prompt:\n%s",
                self._chat_model,
                stream_meta.get("done_reason"),
                stream_meta.get("eval_count"),
                stream_meta.get("prompt_eval_count"),
                stream_meta.get("total_duration"),
                max_tokens, temperature, len(prompt), label, prompt,
            )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def health_check(self) -> dict:
        """
        Verify that the Ollama service is reachable and the configured
        chat model is listed by GET /api/tags.

        Returns a dict with keys:
            reachable (bool), models (list[str]), chat_model_found (bool),
            embed_model_found (bool | None), base_url (str)

        embed_model_found is None when no embedding model is configured
        (not applicable), False when configured but not found, True when
        found.

        Does not raise — all failures are captured in the returned dict.
        """
        result: dict = {
            "reachable":         False,
            "models":            [],
            "chat_model_found":  False,
            # None = not applicable (no embedding model configured)
            "embed_model_found": None if not self._embedding_model else False,
            "base_url":          self._base_url,
        }

        try:
            resp = requests.get(
                self._tags_endpoint,
                timeout=self._request_timeout,
            )
            resp.raise_for_status()
            data   = resp.json()
            models = [m["model"] for m in data.get("models", [])]
            result.update({
                "reachable":        True,
                "models":           models,
                "chat_model_found": self._chat_model in models,
                "embed_model_found": (
                    self._embedding_model in models
                    if self._embedding_model else None
                ),
            })
        except Exception as exc:
            result["error"] = str(exc)
            logger.warning("OllamaRuntimeClient health_check() failed: %s", exc)

        return result

    # -----------------------------------------------------------------------
    # Dunder helpers
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"OllamaRuntimeClient("
            f"chat_model={self._chat_model!r}, "
            f"embedding_model={self._embedding_model!r}, "
            f"base_url={self._base_url!r})"
        )


# ---------------------------------------------------------------------------
# Protocol conformance check (import-time, debug aid)
# ---------------------------------------------------------------------------

def _assert_protocol_conformance() -> None:
    """Verify OllamaRuntimeClient satisfies BaseRuntimeClient at import time."""
    from .base_runtime_client import BaseRuntimeClient
    # chat_model has no default (see class docstring) — pass a placeholder
    # purely to satisfy the constructor for this conformance check.
    assert isinstance(OllamaRuntimeClient(chat_model="placeholder"), BaseRuntimeClient), (
        "OllamaRuntimeClient does not satisfy the BaseRuntimeClient Protocol."
    )
    logger.debug("OllamaRuntimeClient protocol conformance check passed.")
