"""
LORA — BaseRuntimeClient Protocol
==================================
The unified interface that every runtime backend must satisfy.

Layer placement
---------------
  ControllerAgent / Sub-agents  →  BaseRuntimeClient (this contract)
                                         ↓
                               FoundryRuntimeClient  |  OMLXRuntimeClient  |  …

Architectural contract
----------------------
- This module defines the Protocol only.  Zero backend-specific logic.
- All concrete runtime clients (foundry_runtime_client.py,
  omlx_runtime_client.py, …) implement this interface.
- The Controller, Planner, Synthesizer, and all sub-agents are typed
  against BaseRuntimeClient — never against a concrete class.
- Adding a new backend requires only: (a) implementing this Protocol,
  (b) registering it in runtime_factory.py.  Nothing else changes.

Why a Protocol rather than an ABC?
-----------------------------------
Using typing.Protocol keeps the design structurally typed: any class that
implements the required methods satisfies the interface without inheriting
from it.  This preserves the existing FoundryRuntimeClient without forcing
a refactor of its class hierarchy, and makes mock/test runtimes trivial to
write.  @runtime_checkable enables isinstance() checks where needed.

Streaming design note
----------------------
infer_stream() is a synchronous generator rather than an async generator
for the same reason infer() is synchronous: the FastAPI layer wraps all
runtime calls in asyncio.to_thread().  Keeping the runtime layer sync
makes it usable both from synchronous agent code and from async FastAPI
handlers without any special handling inside the runtime itself.
If a backend natively supports async streaming, wrap it in a thread-safe
queue inside the concrete client — do not change this contract.
"""

from __future__ import annotations

from typing import Generator, Protocol, runtime_checkable


@runtime_checkable
class BaseRuntimeClient(Protocol):
    """
    Structural interface for all LORA runtime backends.

    Every concrete runtime client must implement all three methods below.
    Type-checkers (mypy, pyright) and isinstance() checks use this Protocol
    to verify conformance without requiring inheritance.

    Methods
    -------
    infer(prompt, system, max_tokens, temperature, timeout) → str
        Blocking single-turn completion.  Returns the full response string.
        Used by the Planner, Synthesizer, and all sub-agents.

    embed(text) → list[float]
        Dense embedding for the given text string.  Returns a float vector
        whose dimensionality depends on the backend model.
        Used by ResearchAgent for embedding-based re-ranking.

    infer_stream(prompt, system, max_tokens, temperature, timeout) → Generator[str, None, None]
        Synchronous generator that yields text chunks (tokens or small
        word-level pieces) as they are produced by the model.
        Used by the FastAPI streaming endpoint (POST /task/stream) to relay
        tokens to the Svelte UI without buffering the full response.
        Backends that do not natively stream may yield the full string as a
        single chunk — the streaming endpoint degrades gracefully to this.

    Attributes
    ----------
    is_local: bool
        True when this client's inference runs on this machine (oMLX,
        Foundry, or Ollama serving a locally-resident model); False when it
        runs on someone else's hardware (Ollama Cloud). This is the single
        signal every backend-tier-aware ceiling (context_profile.py) keys
        off — set once at construction, never recomputed per request.
        Concrete clients set this as a plain instance attribute; it is
        declared here only so callers typed against BaseRuntimeClient can
        read it.
    """

    is_local: bool

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
        Request a blocking chat completion.

        Parameters
        ----------
        prompt:
            The user-turn content.
        system:
            Optional system prompt.
        max_tokens:
            Hard cap on generated tokens.
        temperature:
            Sampling temperature (0.0 = deterministic, 1.0 = creative).
        label:
            Optional caller identifier for diagnostic correlation (e.g.
            overlap/throughput logging). Backends that don't need it may
            accept and ignore it; it must never affect the request itself.
        timeout:
            Optional per-call override for the request timeout, in
            seconds. None (default) means "use the client's configured
            default timeout" — every existing call site is unaffected by
            this parameter's addition. Intended for cheap, small-
            max_tokens calls (classifiers, gate checks) that should fail
            fast rather than share the same budget as a full-length main-
            dispatch answer (2026-07-17 — see mcp_tool_dispatcher.py's
            _RESEARCH_CLASSIFIER_TIMEOUT for the motivating incident: a
            max_tokens=10 gate-check call stalled for the full default
            timeout on a cloud-model-side hang, confirmed not a local
            issue since health-check polling stayed healthy throughout).
            A backend whose transport doesn't support a per-call timeout
            may accept and ignore this parameter rather than breaking.

        Returns
        -------
        str
            The model's complete response text.

        Raises
        ------
        RuntimeError
            On any transport, timeout, or decode failure.
            All concrete implementations must normalise backend-specific
            exceptions to RuntimeError so callers get a consistent type.
        """
        ...

    def embed(self, text: str) -> list[float]:
        """
        Request a dense embedding vector.

        Parameters
        ----------
        text:
            The input string to embed.

        Returns
        -------
        list[float]
            The embedding vector.  Dimensionality is backend-dependent.

        Raises
        ------
        RuntimeError
            On any transport or decode failure.
        """
        ...

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
        Request a streaming chat completion.

        Yields text chunks as they arrive from the model.  The caller
        accumulates chunks or relays them directly to a client (e.g. SSE).

        Backends without native streaming support must still implement this
        method; they may yield the full infer() result as one chunk:

            def infer_stream(self, prompt, system="", max_tokens=1024,
                             temperature=0.2, label="", timeout=None):
                yield self.infer(prompt, system, max_tokens, temperature,
                                  label, timeout)

        Parameters
        ----------
        prompt, system, max_tokens, temperature, label:
            Same semantics as infer().
        timeout:
            Same semantics as infer()'s `timeout` parameter — None means
            "use the client's configured default timeout".

        Yields
        ------
        str
            One text chunk per iteration.  Chunk granularity is
            backend-dependent (token, word, or sentence).

        Raises
        ------
        RuntimeError
            On any transport or decode failure.  May be raised after the
            generator has already yielded partial output.
        """
        ...
