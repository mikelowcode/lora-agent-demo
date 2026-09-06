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
from localist.threshold_calibration import CalibrationResult, GateCalibration


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


def _fake_calibration_result() -> CalibrationResult:
    """A deliberately mixed result — two clean gates, two degenerate — so
    tests can assert both branches of the response/persistence shape at
    once, mirroring a real partial calibration outcome."""
    return CalibrationResult(gates={
        "explicit_search_action": GateCalibration(
            threshold=0.70, degenerate=False, reason=None,
            min_positive_score=0.75, max_negative_score=0.40,
            tpr_at_threshold=0.9, fpr_at_threshold=0.05,
            positive_count=3, negative_count=17,
        ),
        "lookup_request": GateCalibration(
            threshold=0.60, degenerate=False, reason=None,
            min_positive_score=0.65, max_negative_score=0.35,
            tpr_at_threshold=1.0, fpr_at_threshold=0.1,
            positive_count=3, negative_count=17,
        ),
        "research_intent": GateCalibration(
            threshold=None, degenerate=True,
            reason="best Youden's J (0.10) below floor (0.5)",
            min_positive_score=0.3, max_negative_score=0.35,
            tpr_at_threshold=0.2, fpr_at_threshold=0.1,
            positive_count=10, negative_count=17,
        ),
        "episodic_relevance": GateCalibration(
            threshold=None, degenerate=True,
            reason="empty positive or negative pool (embed_fn failures thinned the "
                   "battery, or this gate has no labeled positive pool to begin with)",
            min_positive_score=None, max_negative_score=None,
            tpr_at_threshold=None, fpr_at_threshold=None,
            positive_count=0, negative_count=20,
        ),
    })


class TestAutomaticFirstTimeCalibration:
    """
    POST /settings/embedding-model's automatic, zero-extra-clicks first-time
    calibration (PLAN_semantic_gating_calibration.md §5) — a brand-new,
    non-tuned, non-validated model gets calibrate_thresholds() run and
    persisted before the controller is rebuilt, so it never lands on fully-
    disabled semantic gating without at least trying to do better.

    calibrate_thresholds() itself is monkeypatched — its own correctness is
    threshold_calibration.py's job (see test_threshold_calibration.py); this
    file only tests that main.py wires the trigger/response/persistence
    correctly.
    """

    def test_new_unvalidated_model_triggers_calibration_and_persists(
        self, client, monkeypatch, tmp_path,
    ):
        test_client, _initial_runtime = client
        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(embed_model_found=True, models=["mystery-model"]),
        )
        monkeypatch.setattr(main, "calibrate_thresholds", MagicMock(return_value=_fake_calibration_result()))

        resp = test_client.post("/settings/embedding-model", json={"model": "mystery-model"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["calibration"] is not None
        assert body["calibration"]["model"] == "mystery-model"
        assert body["calibration"]["gates"]["lookup_request"]["threshold"] == 0.60
        assert body["calibration"]["gates"]["lookup_request"]["degenerate"] is False
        assert body["calibration"]["gates"]["research_intent"]["degenerate"] is True
        assert body["calibration"]["gates"]["research_intent"]["threshold"] is None

        # Only the two clean gates persisted — degenerate gates are NULL,
        # not fabricated, in the stored row (see set_calibrated_thresholds).
        persisted = main._state.memory_manager.get_calibrated_thresholds("mystery-model")
        assert persisted == {"explicit_search_action": 0.70, "lookup_request": 0.60}

    def test_already_calibrated_model_does_not_recalibrate_on_switch(
        self, client, monkeypatch, tmp_path,
    ):
        main._state.memory_manager.set_calibrated_thresholds("mystery-model", {"lookup_request": 0.5})
        test_client, _initial_runtime = client
        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(embed_model_found=True, models=["mystery-model"]),
        )
        fake_calibrate = MagicMock(side_effect=AssertionError("must not recalibrate on a plain switch — already attempted"))
        monkeypatch.setattr(main, "calibrate_thresholds", fake_calibrate)

        resp = test_client.post("/settings/embedding-model", json={"model": "mystery-model"})

        assert resp.status_code == 200
        assert resp.json()["calibration"] is None
        fake_calibrate.assert_not_called()

    def test_validated_model_never_triggers_calibration(self, client, monkeypatch, tmp_path):
        test_client, _initial_runtime = client
        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(embed_model_found=True, models=["nomic-embed-text:latest"]),
        )
        fake_calibrate = MagicMock(side_effect=AssertionError("must not calibrate an already-validated model"))
        monkeypatch.setattr(main, "calibrate_thresholds", fake_calibrate)

        resp = test_client.post("/settings/embedding-model", json={"model": "nomic-embed-text:latest"})

        assert resp.status_code == 200
        assert resp.json()["calibration"] is None
        fake_calibrate.assert_not_called()

    def test_tuned_model_never_triggers_calibration(self, client, monkeypatch, tmp_path):
        from localist.planner import _TUNED_EMBEDDING_MODEL

        test_client, _initial_runtime = client
        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(embed_model_found=True, models=[_TUNED_EMBEDDING_MODEL]),
        )
        fake_calibrate = MagicMock(side_effect=AssertionError("must not calibrate the tuned model"))
        monkeypatch.setattr(main, "calibrate_thresholds", fake_calibrate)

        resp = test_client.post("/settings/embedding-model", json={"model": _TUNED_EMBEDDING_MODEL})

        assert resp.status_code == 200
        assert resp.json()["calibration"] is None
        fake_calibrate.assert_not_called()
        assert main._state.memory_manager.embed_fn is main._state.runtime.embed


class TestReembedRecalibration:
    """
    POST /memory/reembed's second phase (PLAN_semantic_gating_calibration.md
    §5) — unlike the switch endpoint's first-time-only trigger, this ALWAYS
    re-runs calibration for the active non-tuned model, every call, and
    rebuilds the controller so the refreshed thresholds take effect
    immediately. Reuses this file's fixture (not test_main_memory_reembed.py's)
    since it needs a fully wired settings/runtime/templates_dir, which
    _build_controller requires.
    """

    def test_reembed_recalibrates_and_rebuilds_controller(self, client, monkeypatch, tmp_path):
        test_client, _initial_runtime = client
        monkeypatch.setattr(
            main, "create_runtime",
            _fake_create_runtime(embed_model_found=True, models=["mystery-model"]),
        )
        # First-time switch: calibrates once (result A).
        monkeypatch.setattr(main, "calibrate_thresholds", MagicMock(return_value=_fake_calibration_result()))
        switch_resp = test_client.post("/settings/embedding-model", json={"model": "mystery-model"})
        assert switch_resp.status_code == 200
        assert switch_resp.json()["calibration"] is not None
        controller_after_switch = main._state.controller

        # Second, DIFFERENT result — proves /memory/reembed actually re-runs
        # calibration rather than reusing the switch-time persisted row.
        improved_result = CalibrationResult(gates={
            "explicit_search_action": GateCalibration(
                threshold=0.72, degenerate=False, reason=None,
                min_positive_score=0.8, max_negative_score=0.3,
                tpr_at_threshold=1.0, fpr_at_threshold=0.0,
                positive_count=3, negative_count=17,
            ),
            "lookup_request": GateCalibration(
                threshold=0.61, degenerate=False, reason=None,
                min_positive_score=0.7, max_negative_score=0.3,
                tpr_at_threshold=1.0, fpr_at_threshold=0.0,
                positive_count=3, negative_count=17,
            ),
            "research_intent": GateCalibration(
                threshold=0.58, degenerate=False, reason=None,
                min_positive_score=0.6, max_negative_score=0.3,
                tpr_at_threshold=0.8, fpr_at_threshold=0.1,
                positive_count=10, negative_count=17,
            ),
            "episodic_relevance": GateCalibration(
                threshold=None, degenerate=True, reason="still degenerate",
                min_positive_score=0.3, max_negative_score=0.35,
                tpr_at_threshold=0.1, fpr_at_threshold=0.05,
                positive_count=13, negative_count=20,
            ),
        })
        fake_calibrate = MagicMock(return_value=improved_result)
        monkeypatch.setattr(main, "calibrate_thresholds", fake_calibrate)

        reembed_resp = test_client.post("/memory/reembed")

        assert reembed_resp.status_code == 200
        fake_calibrate.assert_called_once()
        body = reembed_resp.json()
        assert body["calibration"] is not None
        assert body["calibration"]["model"] == "mystery-model"
        assert body["calibration"]["gates"]["research_intent"]["threshold"] == 0.58
        assert body["calibration"]["gates"]["research_intent"]["degenerate"] is False

        persisted = main._state.memory_manager.get_calibrated_thresholds("mystery-model")
        assert persisted == {
            "explicit_search_action": 0.72,
            "lookup_request": 0.61,
            "research_intent": 0.58,
        }
        # Controller rebuilt so the new thresholds take effect immediately,
        # not just on next restart.
        assert main._state.controller is not controller_after_switch

    def test_reembed_does_not_recalibrate_for_tuned_model(self, client, monkeypatch, tmp_path):
        # Default fixture state: no embedding model switch has happened, so
        # _state.active_embedding_model_name is whatever the fixture leaves
        # it as (None) — nothing to recalibrate, and no embed_fn configured
        # either, so the endpoint 409s before ever reaching the calibration
        # branch. This just guards against a regression where recalibration
        # fires unconditionally regardless of active_embedding_model_name.
        test_client, _initial_runtime = client
        fake_calibrate = MagicMock(side_effect=AssertionError("must not calibrate with no active embedding model"))
        monkeypatch.setattr(main, "calibrate_thresholds", fake_calibrate)

        resp = test_client.post("/memory/reembed")

        assert resp.status_code == 409
        fake_calibrate.assert_not_called()
