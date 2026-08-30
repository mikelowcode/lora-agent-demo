"""
main._configure_embedding_source() — three-tier embedding source precedence.

Covers the platform-gating change to lifespan()'s embedding selection: the
MLX EmbeddingEngine fallback is only ever attempted on Apple Silicon, since
its mlx_lm dependency cannot run elsewhere.

This is a mocked/forced-condition test, not real cross-platform execution.
platform.system()/platform.machine() are monkeypatched to force each branch;
no test here runs on actual non-Apple-Silicon hardware. lifespan() itself is
never exercised (no existing pattern in this suite triggers the real FastAPI
lifespan — see test_main_memory_episodes.py's docstring), so the branch
selection was pulled out into _configure_embedding_source() specifically so
it's callable and assertable in isolation, without needing to run the full
startup sequence (real runtime construction, directory indexing, graph
build, etc.).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from localist import main


def _settings(*, embedding_model="", embedding_engine_enabled=True, runtime_backend="ollama"):
    return SimpleNamespace(
        embedding_model=embedding_model,
        embedding_engine_enabled=embedding_engine_enabled,
        runtime_backend=runtime_backend,
    )


def _runtime():
    return SimpleNamespace(embed=MagicMock(name="runtime.embed"))


class TestRuntimeBackendTier:
    def test_runtime_embed_selected_when_configured_and_found(self, monkeypatch, caplog):
        monkeypatch.setattr(main, "EmbeddingEngine", MagicMock(name="EmbeddingEngine"))
        settings = _settings(embedding_model="nomic-embed-text:latest")
        runtime = _runtime()
        health = {"embed_model_found": True}

        with caplog.at_level(logging.INFO, logger="localist.main"):
            embed_fn, embedding_engine = main._configure_embedding_source(settings, runtime, health)

        assert embed_fn is runtime.embed
        assert embedding_engine is None
        main.EmbeddingEngine.assert_not_called()
        assert "Runtime-backend embeddings ready" in caplog.text

    def test_runtime_tier_wins_even_when_embedding_engine_would_also_qualify(self, monkeypatch):
        # embedding_model set + found, on Apple Silicon, engine enabled —
        # tier 1 still takes precedence over tier 2.
        monkeypatch.setattr(main.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(main.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(main, "EmbeddingEngine", MagicMock(name="EmbeddingEngine"))
        settings = _settings(embedding_model="nomic-embed-text:latest", embedding_engine_enabled=True)
        runtime = _runtime()
        health = {"embed_model_found": True}

        embed_fn, embedding_engine = main._configure_embedding_source(settings, runtime, health)

        assert embed_fn is runtime.embed
        assert embedding_engine is None
        main.EmbeddingEngine.assert_not_called()


class TestEmbeddingEngineTier:
    def test_attempted_on_apple_silicon_when_enabled(self, monkeypatch, caplog):
        monkeypatch.setattr(main.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(main.platform, "machine", lambda: "arm64")

        fake_engine = SimpleNamespace(available=True, embed=MagicMock(name="engine.embed"))
        engine_cls = MagicMock(name="EmbeddingEngine", return_value=fake_engine)
        monkeypatch.setattr(main, "EmbeddingEngine", engine_cls)

        settings = _settings(embedding_model="", embedding_engine_enabled=True)
        runtime = _runtime()
        health = {"embed_model_found": None}

        with caplog.at_level(logging.INFO, logger="localist.main"):
            embed_fn, embedding_engine = main._configure_embedding_source(settings, runtime, health)

        engine_cls.assert_called_once()
        assert embed_fn is fake_engine.embed
        assert embedding_engine is fake_engine
        assert "EmbeddingEngine ready" in caplog.text

    def test_aarch64_machine_string_also_counts_as_apple_silicon(self, monkeypatch):
        # platform.machine() can report "aarch64" as well as "arm64"
        # depending on the Python build; both must be treated as Apple
        # Silicon per the is_apple_silicon check.
        monkeypatch.setattr(main.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(main.platform, "machine", lambda: "aarch64")

        fake_engine = SimpleNamespace(available=True, embed=MagicMock())
        engine_cls = MagicMock(return_value=fake_engine)
        monkeypatch.setattr(main, "EmbeddingEngine", engine_cls)

        settings = _settings(embedding_model="", embedding_engine_enabled=True)
        embed_fn, embedding_engine = main._configure_embedding_source(settings, _runtime(), {})

        engine_cls.assert_called_once()
        assert embed_fn is fake_engine.embed


class TestSkipOnNonAppleSilicon:
    def test_embedding_engine_never_constructed_and_skip_logged(self, monkeypatch, caplog):
        monkeypatch.setattr(main.platform, "system", lambda: "Linux")
        monkeypatch.setattr(main.platform, "machine", lambda: "x86_64")

        engine_cls = MagicMock(name="EmbeddingEngine")
        monkeypatch.setattr(main, "EmbeddingEngine", engine_cls)

        settings = _settings(embedding_model="", embedding_engine_enabled=True)
        runtime = _runtime()
        health = {"embed_model_found": None}

        with caplog.at_level(logging.INFO, logger="localist.main"):
            embed_fn, embedding_engine = main._configure_embedding_source(settings, runtime, health)

        engine_cls.assert_not_called()
        assert embed_fn is None
        assert embedding_engine is None
        assert "EmbeddingEngine skipped" in caplog.text
        assert "requires Apple Silicon" in caplog.text
        assert "Linux" in caplog.text
        assert "x86_64" in caplog.text

    def test_skip_log_is_info_not_warning(self, monkeypatch, caplog):
        # This is an expected skip, not a failure — must not be logged at
        # WARNING (which is what the "EmbeddingEngine failed to load" path
        # uses for genuine load failures).
        monkeypatch.setattr(main.platform, "system", lambda: "Windows")
        monkeypatch.setattr(main.platform, "machine", lambda: "AMD64")
        monkeypatch.setattr(main, "EmbeddingEngine", MagicMock())

        settings = _settings(embedding_model="", embedding_engine_enabled=True)

        with caplog.at_level(logging.INFO, logger="localist.main"):
            main._configure_embedding_source(settings, _runtime(), {})

        skip_records = [r for r in caplog.records if "EmbeddingEngine skipped" in r.message]
        assert len(skip_records) == 1
        assert skip_records[0].levelname == "INFO"


class TestDisabledTier:
    def test_disabled_flag_skips_regardless_of_platform(self, monkeypatch, caplog):
        monkeypatch.setattr(main.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(main.platform, "machine", lambda: "arm64")
        engine_cls = MagicMock(name="EmbeddingEngine")
        monkeypatch.setattr(main, "EmbeddingEngine", engine_cls)

        settings = _settings(embedding_model="", embedding_engine_enabled=False)

        with caplog.at_level(logging.INFO, logger="localist.main"):
            embed_fn, embedding_engine = main._configure_embedding_source(settings, _runtime(), {})

        engine_cls.assert_not_called()
        assert embed_fn is None
        assert embedding_engine is None
        assert "EmbeddingEngine disabled" in caplog.text


class TestDeriveActiveEmbeddingModelName:
    """
    main._derive_active_embedding_model_name() — the three-tier name
    derivation that feeds Planner's _TUNED_EMBEDDING_MODEL guard (docs/
    architecture/16-runtime-backend-layer.md §16.4). Mirrors
    _configure_embedding_source()'s own tier precedence, so each case here
    is fed the same (embed_fn, embedding_engine) shape that function would
    actually return for that tier.
    """

    def test_tier1_runtime_backend_embed_returns_settings_model(self):
        settings = _settings(embedding_model="nomic-embed-text:latest")
        embed_fn = MagicMock(name="runtime.embed")

        name = main._derive_active_embedding_model_name(settings, embed_fn, None)

        assert name == "nomic-embed-text:latest"

    def test_tier2_embedding_engine_available_returns_model_path(self):
        settings = _settings(embedding_model="")
        fake_engine = SimpleNamespace(
            available=True, model_path="mlx-community/embeddinggemma-300m-4bit",
            embed=MagicMock(name="engine.embed"),
        )

        name = main._derive_active_embedding_model_name(settings, fake_engine.embed, fake_engine)

        assert name == "mlx-community/embeddinggemma-300m-4bit"

    def test_tier2_embedding_engine_failed_to_load_returns_none(self):
        """embedding_engine was constructed (tier 2 attempted) but failed to
        load (available=False) — names no model, even if embed_fn happens
        to be non-None."""
        settings = _settings(embedding_model="")
        fake_engine = SimpleNamespace(available=False, model_path="mlx-community/embeddinggemma-300m-4bit")

        name = main._derive_active_embedding_model_name(settings, None, fake_engine)

        assert name is None

    def test_tier3_keyword_only_returns_none(self):
        settings = _settings(embedding_model="")

        name = main._derive_active_embedding_model_name(settings, None, None)

        assert name is None
