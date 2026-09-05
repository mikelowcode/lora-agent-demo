"""
Live-switchable embedding model (main.py: POST /settings/embedding-model;
docs/architecture/16-runtime-backend-layer.md §16.4/§16.5).

Follows the established convention (test_main_runtime_backend_switch.py): the
real FastAPI lifespan() is never triggered; AppState fields are swapped
directly, and main.create_runtime is monkeypatched so no real backend client
(HTTP calls, SDKs) is ever constructed. main._PROJECT_ROOT is monkeypatched
to a tmp_path so .env read-modify-write tests never touch the real
backend/.env file.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.memory_manager import MemoryManager


def _settings(**overrides):
    defaults = dict(
        runtime_backend="ollama",
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


def _fake_create_runtime(*, reachable=True, base_url="http://localhost:11434", models=None,
                          embed_model_found=False, error=None):
    """
    A drop-in replacement for main.create_runtime that never constructs a
    real client — returns a MagicMock (fresh, distinct object per call, so
    identity checks on _state.runtime work) whose health_check() is fixed.
    `.embed` is a MagicMock attribute, usable as a distinct identity target.
    """
    resolved_models = models if models is not None else ["nomic-embed-text"]

    def _create(backend, **kwargs):
        client = MagicMock(name=f"fake-runtime-{backend}")
        client.health_check.return_value = {
            "reachable":         reachable,
            "base_url":          base_url,
            "models":            resolved_models,
            "chat_model_found":  True,
            "embed_model_found": embed_model_found,
            "error":             error,
        }
        return client

    return _create


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """
    TestClient against main.app with all embedding-model-switch-relevant
    AppState fields swapped to isolated fakes, and _PROJECT_ROOT redirected
    to tmp_path so .env reads/writes never touch the real project .env.
    Restores everything afterward.
    """
    monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)

    prev_settings          = main._state.settings
    prev_runtime           = main._state.runtime
    prev_wiki_agent        = main._state.wiki_agent
    prev_controller        = main._state.controller
    prev_memory            = main._state.memory_manager
    prev_templates         = main._state.templates_dir
    prev_embedding_engine  = main._state.embedding_engine
    prev_active_model_name = main._state.active_embedding_model_name

    main._state.settings          = _settings()
    main._state.memory_manager    = MemoryManager(db_path=tmp_path / "main_embedding_switch.db")
    main._state.templates_dir     = tmp_path  # no warmup_fixture.md here — run_cache_warmup no-ops safely
    main._state.embedding_engine  = None

    initial_runtime = MagicMock(name="initial-runtime")
    main._state.runtime    = initial_runtime
    main._state.wiki_agent = MagicMock(name="initial-wiki-agent")
    main._state.controller = MagicMock(name="initial-controller")

    yield TestClient(main.app), initial_runtime

    main._state.settings                  = prev_settings
    main._state.runtime                   = prev_runtime
    main._state.wiki_agent                = prev_wiki_agent
    main._state.controller                = prev_controller
    main._state.memory_manager            = prev_memory
    main._state.templates_dir             = prev_templates
    main._state.embedding_engine          = prev_embedding_engine
    main._state.active_embedding_model_name = prev_active_model_name


class TestOmlxRejectedUpFront:
    def test_omlx_backend_rejected_without_touching_anything(self, client, monkeypatch, tmp_path):
        test_client, initial_runtime = client
        main._state.settings.runtime_backend = "omlx"
        monkeypatch.setattr(
            main, "create_runtime",
            MagicMock(side_effect=AssertionError("create_runtime must not be called")),
        )

        resp = test_client.post("/settings/embedding-model", json={"model": "nomic-embed-text"})

        assert resp.status_code == 409
        assert main._state.runtime is initial_runtime
        assert main._state.memory_manager.embed_fn is None


class TestUnreachableBackend:
    def test_unreachable_backend_leaves_state_and_env_untouched(self, client, monkeypatch, tmp_path):
        test_client, initial_runtime = client
        env_path = tmp_path / ".env"
        original = "LOCALIST_RUNTIME_BACKEND=ollama\n"
        env_path.write_text(original)

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(reachable=False, base_url="http://down:11434"),
        )

        resp = test_client.post("/settings/embedding-model", json={"model": "nomic-embed-text"})

        assert resp.status_code == 502
        assert main._state.runtime is initial_runtime
        assert main._state.memory_manager.embed_fn is None
        assert main._state.settings.embedding_model == ""
        assert env_path.read_text() == original


class TestModelNotFound:
    def test_model_not_on_backend_rejected_without_mutation(self, client, monkeypatch, tmp_path):
        test_client, initial_runtime = client
        env_path = tmp_path / ".env"
        env_path.write_text("LOCALIST_RUNTIME_BACKEND=ollama\n")

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(embed_model_found=False, models=["some-other-model"]),
        )

        resp = test_client.post("/settings/embedding-model", json={"model": "nomic-embed-text"})

        assert resp.status_code == 422
        assert "nomic-embed-text" in resp.json()["detail"]
        assert main._state.runtime is initial_runtime
        assert main._state.memory_manager.embed_fn is None


class TestSuccessfulSwitch:
    def test_setting_a_model_swaps_state_and_persists(self, client, monkeypatch, tmp_path):
        test_client, initial_runtime = client
        prev_wiki_agent = main._state.wiki_agent
        prev_controller = main._state.controller
        env_path = tmp_path / ".env"
        env_path.write_text("LOCALIST_RUNTIME_BACKEND=ollama\n")

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(embed_model_found=True, models=["nomic-embed-text"]),
        )

        resp = test_client.post("/settings/embedding-model", json={"model": "nomic-embed-text"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["backend"] == "ollama"
        assert body["model"] == "nomic-embed-text"
        assert body["persisted"] is True
        assert body["active"] is True

        assert main._state.runtime is not initial_runtime
        assert main._state.wiki_agent is not prev_wiki_agent
        assert main._state.controller is not prev_controller
        assert main._state.settings.embedding_model == "nomic-embed-text"
        assert main._state.memory_manager.embed_fn is main._state.runtime.embed
        assert main._state.active_embedding_model_name == "nomic-embed-text"

        assert "LOCALIST_EMBEDDING_MODEL=nomic-embed-text" in env_path.read_text()

    def test_setting_a_model_ignores_a_stale_embedding_engine_for_naming(
        self, client, monkeypatch, tmp_path,
    ):
        """
        _state.embedding_engine can be non-None left over from startup (tier
        2 engaged then, before this endpoint was ever called) even while
        setting a tier-1 model now — _derive_active_embedding_model_name()
        must not consult it once tier 1 is what's actually wired to
        embed_fn, or the Planner's tuned-threshold guard (§16.4) would
        compare against the wrong model name and silently pass a real
        mismatch.
        """
        test_client, _initial_runtime = client
        fake_engine = MagicMock(name="fake-embedding-engine")
        fake_engine.available = True
        fake_engine.model_path = "mlx-community/embeddinggemma-300m-4bit"
        main._state.embedding_engine = fake_engine

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(embed_model_found=True, models=["nomic-embed-text"]),
        )

        resp = test_client.post("/settings/embedding-model", json={"model": "nomic-embed-text"})

        assert resp.status_code == 200
        assert main._state.memory_manager.embed_fn is main._state.runtime.embed
        assert main._state.active_embedding_model_name == "nomic-embed-text"

    def test_clearing_the_model_disables_embed_fn(self, client, monkeypatch, tmp_path):
        test_client, _initial_runtime = client
        env_path = tmp_path / ".env"
        env_path.write_text("LOCALIST_RUNTIME_BACKEND=ollama\nLOCALIST_EMBEDDING_MODEL=nomic-embed-text\n")
        main._state.settings.embedding_model = "nomic-embed-text"

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(embed_model_found=False, models=["nomic-embed-text"]),
        )

        resp = test_client.post("/settings/embedding-model", json={"model": ""})

        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == ""
        assert body["active"] is False

        assert main._state.settings.embedding_model == ""
        assert main._state.memory_manager.embed_fn is None
        assert main._state.active_embedding_model_name is None
        assert "LOCALIST_EMBEDDING_MODEL=\n" in env_path.read_text()

    def test_clearing_falls_back_to_already_loaded_embedding_engine(self, client, monkeypatch, tmp_path):
        """
        The desktop-build (base-only PyInstaller freeze) case has no
        EmbeddingEngine to fall back to (_state.embedding_engine stays None,
        covered by test_clearing_the_model_disables_embed_fn above). On the
        full dev build, an EmbeddingEngine may already be loaded from
        startup (tier 2 engaged because LOCALIST_EMBEDDING_MODEL was unset
        then) — clearing a tier-1 override must fall back to that already-
        loaded instance, not drop straight to keyword-only and silently lose
        working embeddings that were available the whole time.
        """
        test_client, _initial_runtime = client
        env_path = tmp_path / ".env"
        env_path.write_text("LOCALIST_RUNTIME_BACKEND=ollama\nLOCALIST_EMBEDDING_MODEL=nomic-embed-text\n")
        main._state.settings.embedding_model = "nomic-embed-text"

        fake_engine = MagicMock(name="fake-embedding-engine")
        fake_engine.available = True
        fake_engine.model_path = "mlx-community/embeddinggemma-300m-4bit"
        main._state.embedding_engine = fake_engine

        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(embed_model_found=False, models=["nomic-embed-text"]),
        )

        resp = test_client.post("/settings/embedding-model", json={"model": ""})

        assert resp.status_code == 200
        body = resp.json()
        assert body["active"] is True  # embed_fn is set, just to the EmbeddingEngine now

        assert main._state.settings.embedding_model == ""
        assert main._state.memory_manager.embed_fn is fake_engine.embed
        assert main._state.active_embedding_model_name == "mlx-community/embeddinggemma-300m-4bit"

    def test_write_env_failure_reports_persisted_false_but_keeps_live_swap(
        self, client, monkeypatch, tmp_path,
    ):
        test_client, initial_runtime = client
        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(embed_model_found=True, models=["nomic-embed-text"]),
        )
        monkeypatch.setattr(
            main, "_write_env_var",
            MagicMock(side_effect=OSError("read-only filesystem")),
        )

        resp = test_client.post("/settings/embedding-model", json={"model": "nomic-embed-text"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["persisted"] is False
        assert body["error"] is not None
        # In-memory swap still applied despite the .env write failure.
        assert main._state.runtime is not initial_runtime
        assert main._state.memory_manager.embed_fn is main._state.runtime.embed
