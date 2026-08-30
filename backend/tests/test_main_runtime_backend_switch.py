"""
Live-switchable runtime backend + per-backend chat-model pinning
(main.py: POST /settings/runtime-backend, GET .../models,
POST .../chat-model; docs/architecture/16-runtime-backend-layer.md §16.3).

Follows the established convention (test_main_memory_episodes.py,
test_main_embedding_source_selection.py): the real FastAPI lifespan() is
never triggered; AppState fields are swapped directly, and main.create_runtime
is monkeypatched so no real backend client (HTTP calls, SDKs) is ever
constructed. main._PROJECT_ROOT is monkeypatched to a tmp_path so .env
read-modify-write tests never touch the real backend/.env file.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.memory_manager import MemoryManager


def _settings(**overrides):
    defaults = dict(
        runtime_backend="omlx",
        chat_model=None,
        chat_model_omlx=None,
        chat_model_ollama=None,
        chat_model_foundry=None,
        embedding_model="",
        foundry_url=None,
        omlx_url="http://localhost:8000",
        ollama_url="http://localhost:11434",
        request_timeout=30.0,
        stream_timeout=60.0,
        episodic_write_approval=False,
    )
    defaults.update(overrides)
    return main.Settings(**defaults)


def _fake_create_runtime(*, reachable=True, base_url="http://fake", models=None,
                          chat_model_found=True, error=None):
    """
    A drop-in replacement for main.create_runtime that never constructs a
    real client — returns a MagicMock (fresh, distinct object per call, so
    identity checks on _state.runtime work) whose health_check() is fixed.
    """
    resolved_models = models if models is not None else ["model-a"]

    def _create(backend, **kwargs):
        client = MagicMock(name=f"fake-runtime-{backend}")
        client.health_check.return_value = {
            "reachable":         reachable,
            "base_url":          base_url,
            "models":            resolved_models,
            "chat_model_found":  chat_model_found,
            "embed_model_found": False,
            "error":             error,
        }
        return client

    return _create


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """
    TestClient against main.app with all runtime-backend-switch-relevant
    AppState fields swapped to isolated fakes, and _PROJECT_ROOT redirected
    to tmp_path so .env reads/writes never touch the real project .env.
    Restores everything afterward.
    """
    monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)

    prev_settings   = main._state.settings
    prev_runtime    = main._state.runtime
    prev_wiki_agent = main._state.wiki_agent
    prev_controller = main._state.controller
    prev_memory     = main._state.memory_manager
    prev_templates  = main._state.templates_dir

    main._state.settings       = _settings()
    main._state.memory_manager = MemoryManager(db_path=tmp_path / "main_runtime_switch.db")
    main._state.templates_dir  = tmp_path  # no warmup_fixture.md here — run_cache_warmup no-ops safely

    initial_runtime = MagicMock(name="initial-runtime")
    main._state.runtime    = initial_runtime
    main._state.wiki_agent = MagicMock(name="initial-wiki-agent")
    main._state.controller = MagicMock(name="initial-controller")

    yield TestClient(main.app), initial_runtime

    main._state.settings       = prev_settings
    main._state.runtime        = prev_runtime
    main._state.wiki_agent     = prev_wiki_agent
    main._state.controller     = prev_controller
    main._state.memory_manager = prev_memory
    main._state.templates_dir  = prev_templates


# ---------------------------------------------------------------------------
# Unknown backend rejected on all three endpoints
# ---------------------------------------------------------------------------

class TestUnknownBackendRejected:
    def test_switch_endpoint_rejects_and_touches_nothing(self, client, monkeypatch, tmp_path):
        test_client, initial_runtime = client
        monkeypatch.setattr(
            main, "create_runtime",
            MagicMock(side_effect=AssertionError("create_runtime must not be called")),
        )
        env_path = tmp_path / ".env"
        env_path.write_text("LOCALIST_RUNTIME_BACKEND=omlx\n")

        resp = test_client.post("/settings/runtime-backend", json={"backend": "not-a-real-backend"})

        assert resp.status_code == 422
        assert main._state.runtime is initial_runtime
        assert env_path.read_text() == "LOCALIST_RUNTIME_BACKEND=omlx\n"

    def test_models_endpoint_rejects_and_touches_nothing(self, client, monkeypatch):
        test_client, initial_runtime = client
        monkeypatch.setattr(
            main, "create_runtime",
            MagicMock(side_effect=AssertionError("create_runtime must not be called")),
        )

        resp = test_client.get("/settings/runtime-backend/not-a-real-backend/models")

        assert resp.status_code == 422
        assert main._state.runtime is initial_runtime

    def test_chat_model_endpoint_rejects_and_touches_nothing(self, client, monkeypatch, tmp_path):
        test_client, initial_runtime = client
        monkeypatch.setattr(
            main, "create_runtime",
            MagicMock(side_effect=AssertionError("create_runtime must not be called")),
        )
        env_path = tmp_path / ".env"

        resp = test_client.post(
            "/settings/runtime-backend/not-a-real-backend/chat-model",
            json={"chat_model": "some-model"},
        )

        assert resp.status_code == 422
        assert main._state.runtime is initial_runtime
        assert not env_path.exists()


# ---------------------------------------------------------------------------
# POST /settings/runtime-backend
# ---------------------------------------------------------------------------

class TestSwitchRuntimeBackend:
    def test_unreachable_target_leaves_state_and_env_untouched(self, client, monkeypatch, tmp_path):
        test_client, initial_runtime = client
        env_path = tmp_path / ".env"
        original = "LOCALIST_RUNTIME_BACKEND=omlx\n# a comment\n\nOTHER_KEY=1\n"
        env_path.write_text(original)

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(reachable=False, base_url="http://down:1234"),
        )

        resp = test_client.post("/settings/runtime-backend", json={"backend": "ollama"})

        assert resp.status_code == 502
        assert main._state.runtime is initial_runtime
        assert main._state.settings.runtime_backend == "omlx"
        assert env_path.read_text() == original

    def test_successful_switch_swaps_state_identity_and_updates_env(self, client, monkeypatch, tmp_path):
        test_client, initial_runtime = client
        prev_wiki_agent = main._state.wiki_agent
        prev_controller = main._state.controller

        env_path = tmp_path / ".env"
        original = "# a comment\nLOCALIST_RUNTIME_BACKEND=omlx\n\nOTHER_KEY=unchanged\n"
        env_path.write_text(original)

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(reachable=True, base_url="http://localhost:11434"),
        )

        resp = test_client.post("/settings/runtime-backend", json={"backend": "ollama"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["backend"] == "ollama"
        assert body["persisted"] is True
        assert body["reachable"] is True
        assert body["base_url"] == "http://localhost:11434"

        assert main._state.runtime is not initial_runtime
        assert main._state.wiki_agent is not prev_wiki_agent
        assert main._state.controller is not prev_controller
        assert main._state.settings.runtime_backend == "ollama"

        expected_env = original.replace(
            "LOCALIST_RUNTIME_BACKEND=omlx\n", "LOCALIST_RUNTIME_BACKEND=ollama\n",
        )
        assert env_path.read_text() == expected_env

    def test_successful_switch_leaves_embed_fn_untouched(self, client, monkeypatch, tmp_path):
        """
        docs/architecture/16-runtime-backend-layer.md §16.5: a chat-backend switch must never
        change which embedding source is in use. Assign a sentinel to _embed_fn, switch backends,
        and assert memory_manager.embed_fn is the exact same object afterward (identity, not
        equality) — proving the switch path never re-derives it from the new runtime.
        """
        test_client, _initial_runtime = client
        env_path = tmp_path / ".env"
        env_path.write_text("LOCALIST_RUNTIME_BACKEND=omlx\n")

        sentinel_embed_fn = lambda text: [0.0] * 768  # noqa: E731
        main._state.memory_manager._embed_fn = sentinel_embed_fn

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(reachable=True, base_url="http://localhost:11434"),
        )

        resp = test_client.post("/settings/runtime-backend", json={"backend": "ollama"})

        assert resp.status_code == 200
        assert main._state.memory_manager.embed_fn is sentinel_embed_fn

    def test_switch_with_chat_model_override_persists_both(self, client, monkeypatch, tmp_path):
        test_client, _initial_runtime = client
        env_path = tmp_path / ".env"
        env_path.write_text("LOCALIST_RUNTIME_BACKEND=omlx\n")

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(reachable=True, base_url="http://localhost:11434"),
        )

        resp = test_client.post(
            "/settings/runtime-backend",
            json={"backend": "ollama", "chat_model": "gemma4:e4b-mlx"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["chat_model"] == "gemma4:e4b-mlx"

        assert main._state.settings.chat_model_ollama == "gemma4:e4b-mlx"
        assert main._state.settings.runtime_backend == "ollama"

        env_text = env_path.read_text()
        assert "LOCALIST_CHAT_MODEL_OLLAMA=gemma4:e4b-mlx" in env_text
        assert "LOCALIST_RUNTIME_BACKEND=ollama" in env_text


# ---------------------------------------------------------------------------
# GET /settings/runtime-backend/{backend}/models
# ---------------------------------------------------------------------------

class TestGetRuntimeBackendModels:
    def test_returns_health_check_output_and_never_touches_state(self, client, monkeypatch):
        test_client, initial_runtime = client
        prev_controller = main._state.controller

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(
                reachable=True, base_url="http://foundry:9999",
                models=["m1", "m2"], chat_model_found=True,
            ),
        )

        resp = test_client.get("/settings/runtime-backend/foundry/models")

        assert resp.status_code == 200
        assert resp.json() == {
            "reachable":        True,
            "base_url":         "http://foundry:9999",
            "models":           ["m1", "m2"],
            "chat_model_found": True,
            "error":            None,
        }
        assert main._state.runtime is initial_runtime
        assert main._state.controller is prev_controller


# ---------------------------------------------------------------------------
# POST /settings/runtime-backend/{backend}/chat-model
# ---------------------------------------------------------------------------

class TestSetRuntimeBackendChatModel:
    def test_pin_for_inactive_backend_persists_without_rebuild(self, client, monkeypatch, tmp_path):
        test_client, initial_runtime = client
        prev_wiki_agent = main._state.wiki_agent
        prev_controller = main._state.controller

        env_path = tmp_path / ".env"
        env_path.write_text("LOCALIST_RUNTIME_BACKEND=omlx\n")

        # Active backend is "omlx" (fixture default) — pin targets "ollama",
        # so create_runtime must never be invoked.
        monkeypatch.setattr(
            main, "create_runtime",
            MagicMock(side_effect=AssertionError("create_runtime must not be called for an inactive-backend pin")),
        )

        resp = test_client.post(
            "/settings/runtime-backend/ollama/chat-model",
            json={"chat_model": "gemma4:e4b-mlx"},
        )

        assert resp.status_code == 200
        assert resp.json() == {
            "backend":      "ollama",
            "chat_model":   "gemma4:e4b-mlx",
            "persisted":    True,
            "applied_live": False,
        }

        assert main._state.settings.chat_model_ollama == "gemma4:e4b-mlx"
        assert main._state.runtime is initial_runtime
        assert main._state.wiki_agent is prev_wiki_agent
        assert main._state.controller is prev_controller
        assert "LOCALIST_CHAT_MODEL_OLLAMA=gemma4:e4b-mlx" in env_path.read_text()

    def test_pin_for_active_backend_persists_and_rebuilds(self, client, monkeypatch, tmp_path):
        test_client, initial_runtime = client
        prev_wiki_agent = main._state.wiki_agent
        prev_controller = main._state.controller

        env_path = tmp_path / ".env"
        env_path.write_text("LOCALIST_RUNTIME_BACKEND=omlx\n")

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(reachable=True, base_url="http://localhost:8000"),
        )

        resp = test_client.post(
            "/settings/runtime-backend/omlx/chat-model",
            json={"chat_model": "new-model"},
        )

        assert resp.status_code == 200
        assert resp.json() == {
            "backend":      "omlx",
            "chat_model":   "new-model",
            "persisted":    True,
            "applied_live": True,
        }

        assert main._state.settings.chat_model_omlx == "new-model"
        assert main._state.runtime is not initial_runtime
        assert main._state.wiki_agent is not prev_wiki_agent
        assert main._state.controller is not prev_controller
        assert "LOCALIST_CHAT_MODEL_OMLX=new-model" in env_path.read_text()

    def test_pin_for_active_backend_leaves_embed_fn_untouched(self, client, monkeypatch, tmp_path):
        """
        Same guarantee as test_successful_switch_leaves_embed_fn_untouched, but for the
        applied_live=True chat-model-pin path, which also calls _build_controller().
        """
        test_client, _initial_runtime = client
        env_path = tmp_path / ".env"
        env_path.write_text("LOCALIST_RUNTIME_BACKEND=omlx\n")

        sentinel_embed_fn = lambda text: [0.0] * 768  # noqa: E731
        main._state.memory_manager._embed_fn = sentinel_embed_fn

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(reachable=True, base_url="http://localhost:8000"),
        )

        resp = test_client.post(
            "/settings/runtime-backend/omlx/chat-model",
            json={"chat_model": "new-model"},
        )

        assert resp.status_code == 200
        assert resp.json()["applied_live"] is True
        assert main._state.memory_manager.embed_fn is sentinel_embed_fn

    def test_pin_for_active_but_unreachable_backend_still_persists(self, client, monkeypatch, tmp_path):
        test_client, initial_runtime = client
        env_path = tmp_path / ".env"
        env_path.write_text("LOCALIST_RUNTIME_BACKEND=omlx\n")

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(reachable=False, base_url="http://down:8000"),
        )

        resp = test_client.post(
            "/settings/runtime-backend/omlx/chat-model",
            json={"chat_model": "new-model"},
        )

        # Live rebuild fails (502), but the pin itself was already written
        # before the rebuild was attempted — "Always" persists per spec.
        assert resp.status_code == 502
        assert main._state.settings.chat_model_omlx == "new-model"
        assert "LOCALIST_CHAT_MODEL_OMLX=new-model" in env_path.read_text()
        assert main._state.runtime is initial_runtime


# ---------------------------------------------------------------------------
# _write_env_var() — direct unit tests, independent of the endpoints
# ---------------------------------------------------------------------------

class TestWriteEnvVar:
    def test_appends_new_key_when_absent(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("FOO=bar\n")

        main._write_env_var(tmp_path, "LOCALIST_RUNTIME_BACKEND", "ollama")

        assert env_path.read_text() == "FOO=bar\nLOCALIST_RUNTIME_BACKEND=ollama\n"

    def test_replaces_existing_key_preserving_every_other_line(self, tmp_path):
        env_path = tmp_path / ".env"
        original = (
            "# comment\n"
            "FOO=bar\n"
            "\n"
            "LOCALIST_RUNTIME_BACKEND=omlx\n"
            "BAZ=qux\n"
        )
        env_path.write_text(original)

        main._write_env_var(tmp_path, "LOCALIST_RUNTIME_BACKEND", "foundry")

        expected = original.replace(
            "LOCALIST_RUNTIME_BACKEND=omlx\n", "LOCALIST_RUNTIME_BACKEND=foundry\n",
        )
        assert env_path.read_text() == expected

    def test_does_not_match_a_commented_out_key(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("# LOCALIST_RUNTIME_BACKEND=omlx\n")

        main._write_env_var(tmp_path, "LOCALIST_RUNTIME_BACKEND", "ollama")

        content = env_path.read_text()
        assert "# LOCALIST_RUNTIME_BACKEND=omlx\n" in content
        assert "LOCALIST_RUNTIME_BACKEND=ollama\n" in content

    def test_creates_file_when_absent(self, tmp_path):
        env_path = tmp_path / ".env"
        assert not env_path.exists()

        main._write_env_var(tmp_path, "LOCALIST_RUNTIME_BACKEND", "ollama")

        assert env_path.read_text() == "LOCALIST_RUNTIME_BACKEND=ollama\n"

    def test_adds_missing_trailing_newline_before_appending(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("FOO=bar")  # no trailing newline

        main._write_env_var(tmp_path, "LOCALIST_RUNTIME_BACKEND", "ollama")

        assert env_path.read_text() == "FOO=bar\nLOCALIST_RUNTIME_BACKEND=ollama\n"


# ---------------------------------------------------------------------------
# _runtime_switch_lock — must be an asyncio.Lock, not a threading.Lock
# (docs/architecture/16-runtime-backend-layer.md §16.5, sessions-log.md §36/§37)
# ---------------------------------------------------------------------------

class TestConcurrentSwitchRequestsDoNotBlockEventLoop:
    def test_lock_is_an_asyncio_lock(self):
        """
        A plain threading.Lock held across `await asyncio.to_thread(...)` (the
        prior implementation) busy-blocks the whole event loop while a second
        overlapping request waits to acquire it — not just queues behind the
        first request's coroutine, but freezes every other coroutine on the
        loop too. asyncio.Lock lets that wait suspend cooperatively instead.
        """
        assert isinstance(main._runtime_switch_lock, asyncio.Lock)

    def test_two_concurrent_switches_serialize_without_freezing_the_loop(
        self, client, monkeypatch, tmp_path,
    ):
        """
        Fires two overlapping POST /settings/runtime-backend requests (called
        directly as coroutines, sidestepping TestClient's synchronous portal so
        real concurrency on one event loop is guaranteed) against different
        backends, with the first's health-check step held open for
        HOLD_SECONDS. A concurrent heartbeat coroutine ticks throughout; if the
        lock ever busy-blocks the loop (the threading.Lock bug), the heartbeat
        stalls for the full hold duration. With asyncio.Lock, the heartbeat
        keeps ticking while the second request's lock-acquire is suspended.
        """
        _test_client, _initial_runtime = client
        env_path = tmp_path / ".env"
        env_path.write_text("LOCALIST_RUNTIME_BACKEND=omlx\n")

        HOLD_SECONDS = 0.15

        def _slow_create_and_check(settings, backend):
            time.sleep(HOLD_SECONDS)
            resolved = {
                "reachable":         True,
                "base_url":          f"http://{backend}",
                "models":            ["m"],
                "chat_model_found":  True,
                "embed_model_found": False,
                "error":             None,
            }
            fake = MagicMock(name=f"fake-runtime-{backend}")
            fake.health_check.return_value = resolved
            return fake, resolved

        monkeypatch.setattr(main, "_create_and_check_backend", _slow_create_and_check)

        heartbeat_gaps: list[float] = []

        async def _heartbeat(duration: float) -> None:
            loop = asyncio.get_running_loop()
            last = loop.time()
            end = last + duration
            while True:
                await asyncio.sleep(0.01)
                now = loop.time()
                heartbeat_gaps.append(now - last)
                last = now
                if now >= end:
                    break

        async def _run():
            return await asyncio.gather(
                main.switch_runtime_backend(main.RuntimeBackendSwitchRequest(backend="ollama")),
                main.switch_runtime_backend(main.RuntimeBackendSwitchRequest(backend="foundry")),
                _heartbeat(HOLD_SECONDS * 2.5),
            )

        # A regression to threading.Lock doesn't just stall here — held across
        # `await asyncio.to_thread(...)`, a plain threading.Lock deadlocks the
        # whole event loop outright (confirmed by hand against the pre-fix
        # code: the loop can never process the to_thread completion callback
        # that would let the lock holder resume and release it). An in-process
        # asyncio timeout can't detect that, because the frozen event loop
        # can't run its own timer either — only an external thread's
        # `join(timeout=...)` can, since it doesn't depend on that loop at all.
        outcome: dict = {}

        def _target() -> None:
            try:
                outcome["results"] = asyncio.run(_run())
            except BaseException as exc:  # pragma: no cover - only on regression
                outcome["error"] = exc

        runner = threading.Thread(target=_target, daemon=True)
        runner.start()
        runner.join(timeout=HOLD_SECONDS * 20)
        if runner.is_alive():
            pytest.fail(
                "switch_runtime_backend() calls did not complete — the event "
                "loop deadlocked, which is what a threading.Lock held across "
                "`await asyncio.to_thread(...)` does."
            )
        if "error" in outcome:
            raise outcome["error"]

        switch_results = outcome["results"][:2]

        assert all(r.reachable for r in switch_results)
        assert {r.backend for r in switch_results} == {"ollama", "foundry"}
        assert len(heartbeat_gaps) > 5
        # The busy-blocking bug would stall the heartbeat for ~HOLD_SECONDS
        # straight; a healthy asyncio.Lock keeps every gap near the 10ms tick.
        assert max(heartbeat_gaps) < HOLD_SECONDS / 2


# ---------------------------------------------------------------------------
# _resolve_chat_model() — precedence unit tests
# ---------------------------------------------------------------------------

class TestResolveChatModel:
    def test_global_override_wins_over_per_backend_pin(self):
        settings = _settings(chat_model="global-model", chat_model_ollama="pinned-model")
        assert main._resolve_chat_model(settings, "ollama") == "global-model"

    def test_per_backend_pin_used_when_no_global_override(self):
        settings = _settings(chat_model=None, chat_model_ollama="pinned-model")
        assert main._resolve_chat_model(settings, "ollama") == "pinned-model"

    def test_none_when_neither_is_set(self):
        settings = _settings(chat_model=None, chat_model_ollama=None)
        assert main._resolve_chat_model(settings, "ollama") is None
