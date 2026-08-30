"""
Localist — RuntimeFactory
======================
A single entry point that constructs the correct runtime client based on
a backend identifier string.

Layer placement
---------------
  main.py (lifespan startup)  →  runtime_factory.py  →  concrete clients
                                                         (FoundryRuntimeClient,
                                                          OMLXRuntimeClient, …)

Architectural contract
----------------------
- Called once at application startup inside main.py's lifespan function.
- Returns a BaseRuntimeClient-conforming object.
- The Controller, Planner, Synthesizer, and all sub-agents receive this
  object typed as BaseRuntimeClient — they never see the concrete class.
- Adding a new backend requires only: (a) writing a new concrete client
  that satisfies BaseRuntimeClient, (b) adding an entry to _REGISTRY below.
  Nothing else in the stack changes.

Usage in main.py
----------------
Replace the direct FoundryRuntimeClient construction in the lifespan
function with:

    from .runtime_factory import create_runtime

    runtime = create_runtime(
        backend  = settings.runtime_backend,   # "foundry" | "omlx"
        settings = settings,
    )

Settings integration
--------------------
Add one field to the Settings class in main.py:

    runtime_backend: str = "foundry"   # override with LOCALIST_RUNTIME_BACKEND

All other settings fields (chat_model, embedding_model, foundry_url, …)
remain as-is.  The factory reads only the fields relevant to the chosen
backend, so unused fields are silently ignored.
"""

from __future__ import annotations

import logging
from typing import Any

from .base_runtime_client import BaseRuntimeClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------
# Maps the backend string (from LOCALIST_RUNTIME_BACKEND / Settings) to a
# zero-argument factory callable that returns a configured client.
#
# Concrete client imports are deferred inside the factory functions so that
# importing runtime_factory.py never triggers a heavy import chain for a
# backend that isn't being used.  This keeps startup fast and keeps optional
# dependencies (e.g. oMLX's SDK) from failing the import if not installed.

def _make_foundry(kwargs: dict[str, Any]) -> BaseRuntimeClient:
    """Construct a FoundryRuntimeClient from flattened settings kwargs."""
    from .foundry_runtime_client import FoundryRuntimeClient
    return FoundryRuntimeClient(
        chat_model      = kwargs.get("chat_model") or "Phi-4-mini-instruct-generic-gpu:5",
        embedding_model = kwargs.get("embedding_model", "text-embedding-3-small"),
        base_url        = kwargs.get("foundry_url"),        # None → auto-resolve from CLI
        request_timeout = kwargs.get("request_timeout", 30.0),
        stream_timeout  = kwargs.get("stream_timeout",  60.0),
    )


def _make_omlx(kwargs: dict[str, Any]) -> BaseRuntimeClient:
    """
    Construct an OMLXRuntimeClient from flattened settings kwargs.

    embedding_model is threaded through like the other two backends —
    OMLXRuntimeClient itself already supports a configured embedding_model
    (its constructor and embed() work correctly); this factory function
    just passes the kwarg on. Defaults to "" ("not yet configured"),
    matching OMLXRuntimeClient.DEFAULT_EMBEDDING_MODEL.
    """
    from .omlx_runtime_client import OMLXRuntimeClient
    return OMLXRuntimeClient(
        chat_model      = kwargs.get("chat_model") or "gemma-4-e4b-it-4bit",
        embedding_model = kwargs.get("embedding_model", ""),
        base_url        = kwargs.get("omlx_url",        "http://127.0.0.1:8000"),
        request_timeout = kwargs.get("request_timeout", 30.0),
        stream_timeout  = kwargs.get("stream_timeout",  60.0),
    )


def _make_ollama(kwargs: dict[str, Any]) -> BaseRuntimeClient:
    """Construct an OllamaRuntimeClient from flattened settings kwargs."""
    from .ollama_runtime_client import OllamaRuntimeClient
    return OllamaRuntimeClient(
        chat_model      = kwargs.get("chat_model") or "",
        embedding_model = kwargs.get("embedding_model", ""),
        base_url        = kwargs.get("ollama_url",      "http://localhost:11434"),
        request_timeout = kwargs.get("request_timeout", 30.0),
        stream_timeout  = kwargs.get("stream_timeout",  60.0),
    )


# Registry maps backend name → factory function.
# All entries must return a BaseRuntimeClient-conforming object.
_REGISTRY: dict[str, Any] = {
    "foundry": _make_foundry,
    "omlx":    _make_omlx,
    "ollama":  _make_ollama,
}


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def create_runtime(
    backend:  str,
    **kwargs: Any,
) -> BaseRuntimeClient:
    """
    Construct and return the runtime client for the given backend.

    Parameters
    ----------
    backend:
        Backend identifier string.  Case-insensitive.
        Supported values: "foundry", "omlx".
        New backends are added by inserting an entry into _REGISTRY.

    **kwargs:
        Configuration forwarded to the concrete client constructor.
        The factory function for each backend extracts only the keys it
        needs; unrecognised keys are silently ignored.

        Common keys (used by all backends):
            chat_model (str)       — model ID for chat completions
            embedding_model (str)  — model ID for embeddings; honored by
                                      all three backends ("foundry",
                                      "omlx", "ollama")
            request_timeout (float)
            stream_timeout (float)

        Foundry-specific keys:
            foundry_url (str | None) — override auto-resolved base URL

        oMLX-specific keys:
            omlx_url (str) — base URL of the oMLX server

    Returns
    -------
    BaseRuntimeClient
        A configured, ready-to-use runtime client.

    Raises
    ------
    ValueError
        If the backend string is not in the registry.

    Example
    -------
    # In main.py lifespan:
    runtime = create_runtime(
        backend         = settings.runtime_backend,
        chat_model      = settings.chat_model,
        embedding_model = settings.embedding_model,
        foundry_url     = settings.foundry_url,
        omlx_url        = settings.omlx_url,
        request_timeout = settings.request_timeout,
        stream_timeout  = settings.stream_timeout,
    )
    """
    key = backend.strip().lower()

    if key not in _REGISTRY:
        supported = ", ".join(f'"{k}"' for k in sorted(_REGISTRY))
        raise ValueError(
            f"Unknown runtime backend: {backend!r}. "
            f"Supported backends: {supported}. "
            f"To add a new backend, insert an entry into _REGISTRY in runtime_factory.py."
        )

    logger.info("RuntimeFactory: constructing '%s' runtime client.", key)
    client = _REGISTRY[key](kwargs)

    # Sanity-check at construction time — catches misconfigured concrete
    # classes before the first request hits.
    assert isinstance(client, BaseRuntimeClient), (
        f"_REGISTRY['{key}'] returned an object that does not satisfy "
        f"BaseRuntimeClient.  Check the factory function for '{key}'."
    )

    logger.info("RuntimeFactory: '%s' client ready — %r", key, client)
    return client


# ---------------------------------------------------------------------------
# Registry introspection (used by GET /agents equivalent for runtimes)
# ---------------------------------------------------------------------------

def available_backends() -> list[str]:
    """Return the list of registered backend names in sorted order."""
    return sorted(_REGISTRY.keys())
