"""
Localist — FoundryRuntimeClient
============================
Concrete implementation of BaseRuntimeClient for Azure AI Foundry
running locally.

Layer placement
---------------
  ControllerAgent / Sub-agents  →  FoundryRuntimeClient  →  Foundry HTTP API
                                                            (local only, never cloud)

Architectural contract
----------------------
- This module lives in the Local Runtime Layer.
- Agents and the Controller import and call this — they never call the
  Foundry HTTP API directly.
- No FastAPI imports.  No agent logic.  Pure transport + error handling.
- Satisfies BaseRuntimeClient (base_runtime_client.py) — the shared
  interface for all Localist runtime backends.
- Also satisfies the legacy RuntimeClient Protocol defined in
  controller_agent.py (infer + embed subset); existing call sites remain
  unchanged.

Design notes
------------
- Foundry binds to a random ephemeral port on every restart.  The client
  resolves the live port at construction time by parsing `foundry service
  status` output, with a configurable fallback URL.
- Both infer() and embed() are synchronous.  Wrap in asyncio.to_thread()
  at the FastAPI layer if async streaming is needed later.
- Streaming is used for infer() and infer_stream() (SSE accumulation) to
  match the tested behaviour of the standalone WikiAgent.  infer() is
  implemented as list(infer_stream(...)) so both paths share the same
  transport code.  embed() uses a standard non-streaming POST.
- All network errors are caught and re-raised as RuntimeError so callers
  get a consistent exception type regardless of whether the failure was
  a connection error, a bad status code, or a JSON decode failure.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from typing import Generator, Iterator

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FALLBACK_URL          = "http://127.0.0.1:50763"
_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
_EMBEDDINGS_PATH       = "/v1/embeddings"

# Default model IDs — override via constructor if your Foundry instance
# uses different identifiers.
DEFAULT_CHAT_MODEL      = "Phi-4-mini-instruct-generic-gpu:5"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"   # swap for your local embedding model


# ---------------------------------------------------------------------------
# Port resolution
# ---------------------------------------------------------------------------

def resolve_foundry_base_url(fallback: str = _FALLBACK_URL) -> str:
    """
    Resolve the live base URL by parsing `foundry service status` output.

    Example CLI output:
        🟢 Model management service is running on http://127.0.0.1:50763/openai/status

    Returns the base URL without any path component, e.g. http://127.0.0.1:50763.
    Falls back to `fallback` if the CLI is unavailable or produces no match.
    """
    try:
        result = subprocess.run(
            ["foundry", "service", "status"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            match = re.search(r"(https?://[\d.]+:\d+)", line)
            if match:
                url = match.group(1)
                logger.debug("Resolved Foundry base URL from CLI: %s", url)
                return url
    except FileNotFoundError:
        logger.warning("'foundry' CLI not found — falling back to %s", fallback)
    except subprocess.TimeoutExpired:
        logger.warning("'foundry service status' timed out — falling back to %s", fallback)
    except Exception as exc:
        logger.warning("Unexpected error resolving Foundry URL (%s) — falling back to %s", exc, fallback)

    logger.warning("Using fallback Foundry URL: %s", fallback)
    return fallback


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------

def _iter_sse_chunks(response: requests.Response) -> Iterator[str]:
    """
    Yield text delta strings from an OpenAI-compatible SSE stream.

    Each line from the stream looks like:
        data: {"choices": [{"delta": {"content": "hello"}, ...}], ...}
    or:
        data: [DONE]

    Malformed lines and empty deltas are silently skipped.
    """
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith("data:"):
            continue

        data_str = raw_line[len("data:"):].strip()
        if data_str == "[DONE]":
            return

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            logger.debug("SSE: skipping non-JSON line: %s", raw_line[:120])
            continue

        choices = data.get("choices", [])
        if not choices:
            continue

        delta_content = choices[0].get("delta", {}).get("content", "")
        if delta_content:
            yield delta_content


# ---------------------------------------------------------------------------
# FoundryRuntimeClient
# ---------------------------------------------------------------------------

class FoundryRuntimeClient:
    """
    Concrete RuntimeClient for Azure AI Foundry (local execution).

    Satisfies the RuntimeClient Protocol:
        def infer(self, prompt, system, max_tokens, temperature) -> str
        def embed(self, text) -> list[float]

    Parameters
    ----------
    chat_model:
        Model ID for chat completions.  Must match the id returned by
        GET /v1/models on your Foundry instance.
    embedding_model:
        Model ID for embedding generation.
    base_url:
        Override the auto-resolved base URL.  Useful in tests or when
        the Foundry CLI is not on PATH.
    fallback_url:
        URL used when `foundry service status` cannot be parsed.
    request_timeout:
        Seconds before a non-streaming request is considered hung.
    stream_timeout:
        Seconds before the first byte of a streaming response must arrive.
    """

    def __init__(
        self,
        chat_model:      str   = DEFAULT_CHAT_MODEL,
        embedding_model: str   = DEFAULT_EMBEDDING_MODEL,
        base_url:        str | None = None,
        fallback_url:    str   = _FALLBACK_URL,
        request_timeout: float = 30.0,
        stream_timeout:  float = 60.0,
    ) -> None:
        self._chat_model      = chat_model
        self._embedding_model = embedding_model
        self._base_url        = base_url or resolve_foundry_base_url(fallback_url)
        self._request_timeout = request_timeout
        self._stream_timeout  = stream_timeout

        # Azure AI Foundry is local-only in this deployment (see module
        # docstring) — never a candidate for the cloud tier's relaxed
        # context-window ceilings (context_profile.py).
        self.is_local = True

        self._chat_endpoint  = self._base_url + _CHAT_COMPLETIONS_PATH
        self._embed_endpoint = self._base_url + _EMBEDDINGS_PATH

        logger.info(
            "FoundryRuntimeClient initialised — chat: %s  embed: %s  base: %s",
            self._chat_model,
            self._embedding_model,
            self._base_url,
        )

    # -----------------------------------------------------------------------
    # RuntimeClient interface
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
        Request a chat completion from Foundry (blocking).

        `label` is accepted for signature parity with OMLXRuntimeClient
        (whose call sites pass it for overlap/throughput diagnostics) but is
        unused here — Foundry is not the single-shared-instance backend that
        diagnostic exists for.

        Delegates to infer_stream() and accumulates all chunks so that the
        streaming and non-streaming paths share the same transport code.
        The public signature and return type are unchanged from the original
        implementation — all existing call sites continue to work.

        Parameters
        ----------
        prompt:
            The user-turn content.
        system:
            Optional system prompt.
        max_tokens:
            Hard cap on generated tokens.
        temperature:
            Sampling temperature.
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
            On any network error, non-200 status, or stream decode failure.
        """
        chunks = list(self.infer_stream(
            prompt      = prompt,
            system      = system,
            max_tokens  = max_tokens,
            temperature = temperature,
            timeout     = timeout,
        ))
        result = "".join(chunks)
        logger.debug("infer() ← %d chars received.", len(result))
        return result

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
        Request a streaming chat completion from Foundry via SSE.

        `label` is accepted for signature parity with OMLXRuntimeClient but
        unused here (see infer()).

        This is the canonical transport path.  infer() calls this method
        and accumulates its output; the FastAPI streaming endpoint can
        consume this generator directly to relay tokens to the Svelte UI.

        Parameters
        ----------
        timeout:
            Optional per-call override for the request timeout, in
            seconds. None (default) uses self._stream_timeout, unchanged
            from before this parameter was added (2026-07-17).

        Yields
        ------
        str
            One text chunk per iteration (token or small word group).

        Raises
        ------
        RuntimeError
            On any network error, non-200 status, or SSE decode failure.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":       self._chat_model,
            "messages":    messages,
            "stream":      True,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }

        effective_timeout = timeout if timeout is not None else self._stream_timeout

        logger.debug(
            "infer_stream() → %s  max_tokens=%d  temp=%.2f  prompt_chars=%d  timeout=%.1fs",
            self._chat_endpoint,
            max_tokens,
            temperature,
            len(prompt),
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
                f"Cannot reach Foundry at {self._chat_endpoint}. "
                f"Is the service running?  Detail: {exc}"
            ) from exc
        except requests.Timeout:
            raise RuntimeError(
                f"Foundry did not respond within {effective_timeout}s "
                f"(endpoint: {self._chat_endpoint})."
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Foundry returned HTTP {response.status_code} "
                f"from {self._chat_endpoint}: {response.text[:400]}"
            )

        try:
            yield from _iter_sse_chunks(response)
        except Exception as exc:
            raise RuntimeError(
                f"Error reading SSE stream from Foundry: {exc}"
            ) from exc

    def embed(self, text: str) -> list[float]:
        """
        Request a dense embedding vector from Foundry.

        Parameters
        ----------
        text:
            The input string to embed.

        Returns
        -------
        list[float]
            The embedding vector for the first (and only) input.

        Raises
        ------
        RuntimeError
            On any network error, non-200 status, or missing vector in response.
        """
        payload = {
            "model": self._embedding_model,
            "input": text,
        }

        logger.debug(
            "embed() → %s  input_chars=%d",
            self._embed_endpoint,
            len(text),
        )

        try:
            response = requests.post(
                self._embed_endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=self._request_timeout,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Foundry at {self._embed_endpoint}. "
                f"Is the service running?  Detail: {exc}"
            ) from exc
        except requests.Timeout:
            raise RuntimeError(
                f"Foundry embed request timed out after {self._request_timeout}s."
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Foundry embed returned HTTP {response.status_code}: {response.text[:400]}"
            )

        try:
            data = response.json()
            vector: list[float] = data["data"][0]["embedding"]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unexpected embed response shape from Foundry: {exc}\n"
                f"Raw response: {response.text[:400]}"
            ) from exc

        logger.debug("embed() ← vector dim=%d", len(vector))
        return vector

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def health_check(self) -> dict[str, object]:
        """
        Verify that the Foundry service is reachable and the configured
        models are listed by GET /v1/models.

        Returns a dict with keys:
            reachable (bool), models (list[str]), chat_model_found (bool),
            embed_model_found (bool), base_url (str)

        Does not raise — all failures are captured in the returned dict
        so FastAPI can surface them as a health endpoint response.
        """
        result: dict[str, object] = {
            "reachable":         False,
            "models":            [],
            "chat_model_found":  False,
            "embed_model_found": False,
            "base_url":          self._base_url,
        }

        try:
            resp = requests.get(
                self._base_url + "/v1/models",
                timeout=self._request_timeout,
            )
            resp.raise_for_status()
            data   = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            result["reachable"]         = True
            result["models"]            = models
            result["chat_model_found"]  = self._chat_model in models
            result["embed_model_found"] = self._embedding_model in models
        except Exception as exc:
            result["error"] = str(exc)
            logger.warning("health_check() failed: %s", exc)

        return result

    def refresh_base_url(self, fallback: str = _FALLBACK_URL) -> None:
        """
        Re-resolve the Foundry base URL from the CLI.

        Call this if Foundry has restarted and bound to a new port since
        the client was constructed.
        """
        new_url = resolve_foundry_base_url(fallback)
        if new_url != self._base_url:
            logger.info(
                "refresh_base_url(): updating %s → %s",
                self._base_url,
                new_url,
            )
            self._base_url       = new_url
            self._chat_endpoint  = new_url + _CHAT_COMPLETIONS_PATH
            self._embed_endpoint = new_url + _EMBEDDINGS_PATH

    # -----------------------------------------------------------------------
    # Dunder helpers
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FoundryRuntimeClient("
            f"chat_model={self._chat_model!r}, "
            f"embedding_model={self._embedding_model!r}, "
            f"base_url={self._base_url!r})"
        )


# ---------------------------------------------------------------------------
# Protocol conformance check (runs at import time in debug mode)
# ---------------------------------------------------------------------------

def _assert_protocol_conformance() -> None:
    """
    Verify FoundryRuntimeClient satisfies both BaseRuntimeClient and the
    legacy RuntimeClient at import time.
    """
    try:
        from .base_runtime_client import BaseRuntimeClient
        assert isinstance(FoundryRuntimeClient(), BaseRuntimeClient), (
            "FoundryRuntimeClient does not satisfy the BaseRuntimeClient Protocol."
        )
        logger.debug("BaseRuntimeClient conformance check passed.")
    except ImportError:
        pass  # base_runtime_client not on path in isolated test environments

    try:
        from .controller_agent import RuntimeClient
        assert isinstance(FoundryRuntimeClient(), RuntimeClient), (
            "FoundryRuntimeClient does not satisfy the legacy RuntimeClient Protocol."
        )
        logger.debug("Legacy RuntimeClient conformance check passed.")
    except ImportError:
        pass


if __name__ == "__main__":
    # Quick smoke test — run directly to verify the Foundry service is live.
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    client = FoundryRuntimeClient()

    print("\n── Health check ──")
    health = client.health_check()
    for k, v in health.items():
        print(f"  {k}: {v}")

    if not health["reachable"]:
        print("\nFoundry is not reachable — cannot run inference test.")
        sys.exit(1)

    print("\n── Inference test ──")
    response = client.infer(
        system="You are a helpful assistant. Reply in one sentence.",
        prompt="What is the capital of France?",
        max_tokens=64,
    )
    print(f"  Response: {response}")

    print("\n── Embed test ──")
    vector = client.embed("local-first multi-agent research system")
    print(f"  Vector dim: {len(vector)}  first_3: {vector[:3]}")

    print("\nSmoke test complete.")