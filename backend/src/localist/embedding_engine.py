"""
Localist — EmbeddingEngine
======================
Standalone, backend-agnostic embedding engine powered by mlx-embeddings.

Motivation
----------
Embeddings must be stable and deterministic across Localist restarts.  Tying
them to OMLXRuntimeClient's inference model created a coupling that made
both the embedding and the oMLX integration fragile.  This module isolates
embedding into a single responsibility:

  EmbeddingEngine.embed(text) → list[float]   (768-dim, mlx-community/embeddinggemma-300m-4bit)

Library
-------
embeddinggemma-300m-4bit is an mlx-embeddings model, NOT an mlx-lm model.
The correct library is mlx-embeddings (already present in the venv as a
dependency of mlx-audio and mlx-vlm).

The mlx-embeddings API for this model:

    from mlx_embeddings import load
    import mlx.core as mx

    model, tokenizer = load("mlx-community/embeddinggemma-300m-4bit")
    encoded = tokenizer([text], padding=True, truncation=True, return_tensors="mlx")
    output   = model(encoded["input_ids"], encoded["attention_mask"])
    vector   = output.text_embeds   # mlx array, shape (1, 768), already normalised

Architecture
------------
- Loaded once at startup by main.py lifespan.
- ``engine.embed`` (a bound method) is passed as ``embed_fn`` to MemoryManager.
- OMLXRuntimeClient.embed() is NOT called anywhere for corpus embeddings.
- LOCALIST_EMBEDDING_ENGINE_ENABLED=false skips load; MemoryManager runs keyword-only.
- Graceful degradation: load failure sets available=False and logs a warning —
  MemoryManager falls back to keyword-only scoring automatically (embed_fn=None).

Usage in main.py lifespan
-------------------------
    from .embedding_engine import EmbeddingEngine

    engine = EmbeddingEngine()           # loads model (or logs warning)
    embed_fn = engine.embed if engine.available else None
    memory_manager = MemoryManager(..., embed_fn=embed_fn)

Backfill
--------
Run ``backfill_embeddings.py`` after first enabling this module to populate
the ``embedding`` column for documents already indexed without vectors.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Model identifier — embeddinggemma-300m-4bit, mlx-embeddings variant.
_DEFAULT_MODEL = "mlx-community/embeddinggemma-300m-4bit"

# Expected embedding dimensionality.
_EXPECTED_DIM = 768

# Task prefix used for retrieval/similarity queries.
# embeddinggemma supports task-specific prefixes; "sentence similarity" is
# appropriate for corpus retrieval use.
_TASK_PREFIX = "task: sentence similarity | query: "


class EmbeddingEngine:
    """
    Wraps mlx-embeddings for deterministic local embeddings.

    Parameters
    ----------
    model_path:
        HuggingFace / mlx-community model identifier or local directory path.
        Defaults to ``mlx-community/embeddinggemma-300m-4bit``.

    Attributes
    ----------
    available : bool
        True if the model loaded successfully and embed() is safe to call.
    model_path : str
        The model path that was (or was attempted to be) loaded.
    """

    def __init__(self, model_path: str = _DEFAULT_MODEL) -> None:
        self.model_path: str  = model_path
        self.available:  bool = False

        self._model     = None
        self._tokenizer = None

        self._load()

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """
        Embed ``text`` and return a 768-dim float list.

        Signature-compatible with MemoryManager's ``embed_fn`` parameter:

            MemoryManager(..., embed_fn=engine.embed)

        Parameters
        ----------
        text:
            Input string to embed.  Leading/trailing whitespace is stripped.
            The task prefix is prepended automatically.

        Returns
        -------
        list[float]
            Dense normalised embedding vector of length 768.

        Raises
        ------
        RuntimeError
            If the model is not available (load failed).
        RuntimeError
            If mlx-embeddings returns an unexpected output shape.
        """
        if not self.available:
            raise RuntimeError(
                "EmbeddingEngine is not available — model failed to load at startup. "
                f"Attempted path: {self.model_path!r}. "
                "Check logs for the load error.  MemoryManager will fall back to "
                "keyword-only scoring when embed_fn=None."
            )

        text = text.strip()
        if not text:
            logger.debug("embed() called with empty string — returning zero vector.")
            return [0.0] * _EXPECTED_DIM

        # Prepend task prefix — embeddinggemma is trained with these prefixes
        # and produces better retrieval vectors when they are present.
        prefixed = _TASK_PREFIX + text

        try:
            import mlx.core as mx  # type: ignore[import]
            encoded = self._tokenizer(
                [prefixed],
                padding    = True,
                truncation = True,
                return_tensors = "mlx",
            )
            output = self._model(encoded["input_ids"], encoded["attention_mask"])
            # output.text_embeds is shape (batch, 768) — already L2-normalised
            raw = output.text_embeds[0]   # first (only) item in batch
        except Exception as exc:
            raise RuntimeError(
                f"mlx-embeddings embed() call failed for model {self.model_path!r}: {exc}"
            ) from exc

        vector = _to_float_list(raw)

        if len(vector) != _EXPECTED_DIM:
            logger.warning(
                "embed() returned unexpected dimension %d (expected %d). "
                "MemoryManager cosine scores may be incorrect.",
                len(vector), _EXPECTED_DIM,
            )

        logger.debug("embed() ← dim=%d  input_chars=%d", len(vector), len(text))
        return vector

    @property
    def embed_fn(self) -> Callable[[str], list[float]] | None:
        """
        Convenience property — returns ``self.embed`` when available, else None.

            MemoryManager(..., embed_fn=engine.embed_fn)
        """
        return self.embed if self.available else None

    def __repr__(self) -> str:
        return (
            f"EmbeddingEngine("
            f"model_path={self.model_path!r}, "
            f"available={self.available})"
        )

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        """
        Attempt to load the model and tokenizer via mlx-embeddings.

        Sets ``self.available = True`` on success.
        Logs a warning (does not raise) on failure so the server can start
        in keyword-only mode.
        """
        logger.info("EmbeddingEngine: loading model %r …", self.model_path)
        try:
            from mlx_embeddings import load  # type: ignore[import]
            model, tokenizer = load(self.model_path)
            self._model     = model
            self._tokenizer = tokenizer
            self.available  = True
            logger.info(
                "EmbeddingEngine: model loaded successfully — %r", self.model_path
            )
        except ImportError:
            logger.warning(
                "EmbeddingEngine: mlx_embeddings is not installed. "
                "Install with: pip install mlx-embeddings  "
                "MemoryManager will run in keyword-only mode."
            )
        except Exception as exc:
            logger.warning(
                "EmbeddingEngine: failed to load model %r — %s. "
                "MemoryManager will run in keyword-only mode.",
                self.model_path, exc,
            )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_float_list(raw: object) -> list[float]:
    """
    Normalise mlx-embeddings output to list[float].

    Handles:
    - mlx.core.array  — .tolist()
    - numpy ndarray   — .tolist()
    - list[float]     — returned as-is
    - any iterable    — cast element-wise
    """
    if hasattr(raw, "tolist"):
        result = raw.tolist()  # type: ignore[union-attr]
        # mlx .tolist() on a 1-D array returns list[float] directly
        if isinstance(result, list):
            return [float(x) for x in result]
        # scalar edge case
        return [float(result)]

    if isinstance(raw, list):
        return [float(x) for x in raw]

    try:
        return [float(x) for x in raw]  # type: ignore[union-attr]
    except TypeError as exc:
        raise RuntimeError(
            f"EmbeddingEngine: cannot convert embed() output of type "
            f"{type(raw).__name__!r} to list[float]: {exc}"
        ) from exc