"""
oMLX runtime client — overlap-detection logging + serialization lock.

Covers the two concerns added to omlx_runtime_client.py:
  1. RUNTIME_OVERLAP warning logging, correlated by `label`, whenever a
     call starts while another is already in flight.
  2. The module-level threading.Lock that brackets the HTTP call in
     infer_stream() (and, transitively, infer()) so two overlapping
     call sites (main_dispatch / implicit_extraction / working_state)
     can never both be talking to the single oMLX server at once.

Call sites into OMLXRuntimeClient are plain synchronous methods (the
FastAPI layer runs them in a worker thread via asyncio.to_thread), so the
lock under test is threading.Lock, not asyncio.Lock — see the module
docstring in omlx_runtime_client.py for why.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from unittest.mock import patch

import pytest

from localist import omlx_runtime_client as _omlx_mod
from localist.omlx_runtime_client import OMLXRuntimeClient
from localist.prompt_builder import Turn


def _fake_models_response(entries: list[dict]):
    """Build a fake requests.Response-like object for GET /v1/models."""

    class _FakeModelsResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": entries}

    return _FakeModelsResponse()


def _fake_400_response(detail: str):
    """Build a fake requests.Response-like object for a 400 error."""

    class _Fake400Response:
        status_code = 400
        text = json.dumps({"detail": detail})

        def json(self):
            return {"detail": detail}

    return _Fake400Response()


def _fake_prompt_too_long_response(num_tokens: int = 9000, max_ctx: int = 8192):
    """oMLX's server.py validate_context_window() 400 body shape."""
    return _fake_400_response(
        f"Prompt too long: {num_tokens} tokens exceeds max context window of {max_ctx} tokens"
    )


def _fake_response(chunks: list[str], delay_s: float = 0.0):
    """Build a fake requests.Response-like object for _iter_sse_chunks."""

    class _FakeResponse:
        status_code = 200

        def iter_lines(self, decode_unicode: bool = True):
            for chunk in chunks:
                if delay_s:
                    time.sleep(delay_s)
                envelope = {"choices": [{"delta": {"content": chunk}}]}
                yield f"data: {json.dumps(envelope)}"
            yield "data: [DONE]"

    return _FakeResponse()


class TestOverlapWarningLogging:
    def setup_method(self):
        # Guard against cross-test leakage if a prior test left the
        # counter non-zero due to an assertion failure mid-call.
        _omlx_mod._inflight_count = 0

    def test_no_warning_when_no_call_in_flight(self, caplog):
        client = OMLXRuntimeClient()
        with patch.object(_omlx_mod.requests, "post", return_value=_fake_response(["hi"])):
            with caplog.at_level(logging.WARNING, logger="localist.omlx_runtime_client"):
                result = client.infer(prompt="hello", label="main_dispatch")

        assert result == "hi"
        assert "RUNTIME_OVERLAP" not in caplog.text
        assert _omlx_mod._inflight_count == 0

    def test_warning_fires_and_names_label_when_counter_already_positive(self, caplog):
        client = OMLXRuntimeClient()
        # Simulate a call already in flight (e.g. a prior turn's background
        # write that hasn't finished) without needing real concurrency.
        _omlx_mod._inflight_count = 1
        try:
            with patch.object(_omlx_mod.requests, "post", return_value=_fake_response(["hi"])):
                with caplog.at_level(logging.WARNING, logger="localist.omlx_runtime_client"):
                    client.infer(prompt="hello", label="implicit_extraction")
        finally:
            pass

        assert "RUNTIME_OVERLAP detected" in caplog.text
        assert "label=implicit_extraction" in caplog.text
        # Decremented back to the pre-existing in-flight call's count (1),
        # not to 0 — this call didn't own that other in-flight slot.
        assert _omlx_mod._inflight_count == 1
        _omlx_mod._inflight_count = 0

    def test_counter_decrements_on_exception(self):
        client = OMLXRuntimeClient()
        with patch.object(_omlx_mod.requests, "post", side_effect=_omlx_mod.requests.ConnectionError("down")):
            try:
                client.infer(prompt="hello", label="working_state")
            except RuntimeError:
                pass

        assert _omlx_mod._inflight_count == 0


class _NullLock:
    """No-op stand-in for threading.Lock — lets two threads run the
    'locked' section concurrently, used only to prove the test below is
    actually capable of detecting overlap (rather than passing trivially)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class TestSerializationLock:
    def setup_method(self):
        _omlx_mod._inflight_count = 0

    def _concurrent_calls(self, client, delay_s: float):
        def _slow_post(*args, **kwargs):
            return _fake_response(["tok"], delay_s=delay_s)

        def _call(label: str):
            client.infer(prompt="hello", label=label)

        with patch.object(_omlx_mod.requests, "post", side_effect=_slow_post):
            t1 = threading.Thread(target=_call, args=("main_dispatch",))
            t2 = threading.Thread(target=_call, args=("implicit_extraction",))
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)

    def test_lock_serializes_concurrent_infer_calls(self, caplog):
        """
        Two threads (standing in for two call sites racing against the same
        oMLX server) must never trigger a RUNTIME_OVERLAP warning once the
        lock is in place — the second thread blocks on _inflight_lock until
        the first has fully finished (and decremented), so its own overlap
        check always sees 0.
        """
        client = OMLXRuntimeClient()
        with caplog.at_level(logging.WARNING, logger="localist.omlx_runtime_client"):
            self._concurrent_calls(client, delay_s=0.05)

        assert "RUNTIME_OVERLAP" not in caplog.text
        assert _omlx_mod._inflight_count == 0

    def test_sanity_overlap_is_detected_when_lock_disabled(self, caplog):
        """
        Proves the harness above isn't just insensitive to overlap: with
        the real lock swapped for a no-op, the same two concurrent calls
        DO trigger RUNTIME_OVERLAP (and the raw race can corrupt the
        counter, which is exactly why the lock — not just the counter's
        own bookkeeping — must bracket the call).
        """
        client = OMLXRuntimeClient()
        with patch.object(_omlx_mod, "_inflight_lock", _NullLock()):
            with caplog.at_level(logging.WARNING, logger="localist.omlx_runtime_client"):
                self._concurrent_calls(client, delay_s=0.05)

        assert "RUNTIME_OVERLAP" in caplog.text
        _omlx_mod._inflight_count = 0

    def test_label_reaches_debug_log(self, caplog):
        client = OMLXRuntimeClient()
        with patch.object(_omlx_mod.requests, "post", return_value=_fake_response(["hi"])):
            with caplog.at_level(logging.DEBUG, logger="localist.omlx_runtime_client"):
                client.infer(prompt="hello", label="working_state")

        assert "label=working_state" in caplog.text


class TestHealthCheckMaxModelLen:
    """
    health_check() reads GET /v1/models' oMLX-specific max_model_len field
    for the active chat model and caches it onto self.max_model_len, falling
    back to _DEFAULT_MAX_MODEL_LEN when the field is null or absent.
    """

    def test_real_integer_max_model_len_is_captured(self):
        client = OMLXRuntimeClient(chat_model="gemma-4-e4b-it-4bit")
        entries = [
            {"id": "gemma-4-e4b-it-4bit", "object": "model", "created": 1,
             "owned_by": "omlx", "max_model_len": 32768},
        ]
        with patch.object(
            _omlx_mod.requests, "get",
            return_value=_fake_models_response(entries),
        ):
            client.health_check()

        assert client.max_model_len == 32768

    def test_null_max_model_len_falls_back_to_default(self):
        client = OMLXRuntimeClient(chat_model="gemma-4-e4b-it-4bit")
        entries = [
            {"id": "gemma-4-e4b-it-4bit", "object": "model", "created": 1,
             "owned_by": "omlx", "max_model_len": None},
        ]
        with patch.object(
            _omlx_mod.requests, "get",
            return_value=_fake_models_response(entries),
        ):
            client.health_check()

        assert client.max_model_len == _omlx_mod._DEFAULT_MAX_MODEL_LEN

    def test_missing_max_model_len_field_falls_back_to_default(self):
        client = OMLXRuntimeClient(chat_model="gemma-4-e4b-it-4bit")
        entries = [
            {"id": "gemma-4-e4b-it-4bit", "object": "model", "created": 1,
             "owned_by": "omlx"},
        ]
        with patch.object(
            _omlx_mod.requests, "get",
            return_value=_fake_models_response(entries),
        ):
            client.health_check()

        assert client.max_model_len == _omlx_mod._DEFAULT_MAX_MODEL_LEN


class TestPromptTooLongRetry:
    """
    oMLX's specific 400 "Prompt too long" response (server.py's
    validate_context_window(), f"Prompt too long: {N} tokens exceeds max
    context window of {M} tokens") triggers a distinct path: drop the
    single oldest working-memory turn from the outgoing `messages` array
    and retry exactly once. Any other non-200 (including a differently-
    shaped 400) keeps the pre-existing generic-error behavior.
    """

    def setup_method(self):
        _omlx_mod._inflight_count = 0

    def _turns(self) -> list[Turn]:
        return [
            Turn("user",  "oldest turn"),
            Turn("agent", "middle turn"),
            Turn("user",  "newest turn"),
        ]

    def _sent_messages(self, call) -> list[dict]:
        return json.loads(call.kwargs["data"])["messages"]

    def test_normal_200_never_triggers_retry(self):
        """Baseline: a plain success must not touch the retry path at all."""
        client = OMLXRuntimeClient()
        with patch.object(
            _omlx_mod.requests, "post", return_value=_fake_response(["hi"]),
        ) as mock_post:
            result = client.infer(
                prompt = "current query",
                system = "sys",
                working_memory_turns = self._turns(),
            )

        assert result == "hi"
        assert mock_post.call_count == 1
        sent = self._sent_messages(mock_post.call_args)
        # system + 3 turns + current query = 5, all present on the one call.
        assert len(sent) == 5

    def test_prompt_too_long_drops_oldest_turn_and_retries_once(self):
        client = OMLXRuntimeClient()
        responses = [
            _fake_prompt_too_long_response(),
            _fake_response(["retried answer"]),
        ]
        with patch.object(
            _omlx_mod.requests, "post", side_effect=responses,
        ) as mock_post:
            result = client.infer(
                prompt = "current query",
                system = "sys",
                working_memory_turns = self._turns(),
            )

        # The successful retry's response is what the caller gets back.
        assert result == "retried answer"
        assert mock_post.call_count == 2

        first_messages  = self._sent_messages(mock_post.call_args_list[0])
        second_messages = self._sent_messages(mock_post.call_args_list[1])

        assert len(first_messages)  == 5   # system + 3 turns + query
        assert len(second_messages) == 4   # exactly one turn dropped

        first_contents  = [m["content"] for m in first_messages]
        second_contents = [m["content"] for m in second_messages]

        # The dropped turn is the oldest one specifically, not just "a" turn.
        assert "oldest turn" in first_contents
        assert "oldest turn" not in second_contents
        assert "middle turn" in second_contents
        assert "newest turn" in second_contents

        # System message and the current query survive untouched, in place.
        assert second_messages[0]  == {"role": "system", "content": "sys"}
        assert second_messages[-1] == {"role": "user", "content": "current query"}

    def test_prompt_too_long_on_retry_too_raises_distinct_error_no_third_attempt(self):
        client = OMLXRuntimeClient()
        responses = [
            _fake_prompt_too_long_response(),
            _fake_prompt_too_long_response(),
        ]
        with patch.object(
            _omlx_mod.requests, "post", side_effect=responses,
        ) as mock_post:
            with pytest.raises(RuntimeError) as exc_info:
                client.infer(
                    prompt = "current query",
                    working_memory_turns = self._turns(),
                )

        # Exactly the initial attempt + the one retry — never a third.
        assert mock_post.call_count == 2

        message = str(exc_info.value)
        # Distinguishable from the generic "oMLX returned HTTP 400: ..."
        # shape used for every other non-200 — a caller catching
        # RuntimeError can still tell these apart by message content.
        assert "still too long" in message.lower()
        assert "returned HTTP 400" not in message

    def test_non_prompt_too_long_400_does_not_trigger_retry(self):
        """
        A 400 for an unrelated reason (e.g. malformed request) must not be
        mistaken for oMLX's "Prompt too long" case — no turn gets dropped,
        no retry happens, and the pre-existing generic-error message shape
        is preserved.
        """
        client = OMLXRuntimeClient()
        with patch.object(
            _omlx_mod.requests, "post",
            return_value=_fake_400_response("Invalid request: missing field 'model'"),
        ) as mock_post:
            with pytest.raises(RuntimeError) as exc_info:
                client.infer(
                    prompt = "current query",
                    working_memory_turns = self._turns(),
                )

        assert mock_post.call_count == 1
        assert "returned HTTP 400" in str(exc_info.value)
