"""
OllamaRuntimeClient — constructor validation for chat_model.

Covers the fail-fast behaviour added to ollama_runtime_client.py: chat_model
has no default (DEFAULT_CHAT_MODEL == "") because Ollama serves models of
wildly different size, so silently falling back to a specific one (e.g. a
multi-gigabyte local pull) is never safe. Construction must raise ValueError
immediately when chat_model is empty, rather than deferring the failure to
the first request against Ollama.
"""

from __future__ import annotations

import pytest

from localist.ollama_runtime_client import DEFAULT_CHAT_MODEL, OllamaRuntimeClient


def test_default_chat_model_is_empty():
    assert DEFAULT_CHAT_MODEL == ""


def test_empty_chat_model_raises_value_error_at_construction():
    with pytest.raises(ValueError, match="chat_model"):
        OllamaRuntimeClient()


def test_explicit_empty_string_chat_model_raises_value_error():
    with pytest.raises(ValueError, match="chat_model"):
        OllamaRuntimeClient(chat_model="")


def test_explicit_chat_model_constructs_successfully():
    client = OllamaRuntimeClient(chat_model="gemma4:e4b-mlx")
    assert client._chat_model == "gemma4:e4b-mlx"
