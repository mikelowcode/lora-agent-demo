"""
`is_local` (base_runtime_client.py) — added 2026-07-18 alongside
context_profile.py to let every backend-tier-aware ceiling key off one
flag instead of three ad-hoc special cases.

- oMLX only ever runs on-device: always True.
- Foundry is local-only in this deployment (its own module docstring says
  "local only, never cloud"): always True.
- Ollama can serve either a local pull or an Ollama-Cloud-hosted model
  through the *same* local daemon (base_url stays localhost:11434 either
  way — see docs/architecture/16-runtime-backend-layer.md §16.4's
  live-verified "gemma4:31b-cloud, proxied through ollama.com" config), so
  its is_local derivation is covered separately and more thoroughly in
  tests/test_ollama_runtime_client.py::TestIsLocalAndNumCtx — this file
  only re-confirms the two straightforward cases for completeness.

Also covers context_profile.py's ContextProfile dataclass values and
profile_for() resolution directly, since every other test in this session
exercises them only indirectly through ControllerAgent/OllamaRuntimeClient.

LOCAL_PROFILE is no longer a fixed module-level constant (see
context_profile.py's module docstring, "Prior approach, retired") — it's
now resolved per-call by profile_for(runtime) from the runtime client's
live max_model_len, so these tests exercise it via a small fake runtime
double rather than importing a constant.
"""

from __future__ import annotations

from localist.omlx_runtime_client import OMLXRuntimeClient
from localist.foundry_runtime_client import FoundryRuntimeClient
from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.context_profile import (
    CLOUD_PROFILE,
    ContextProfile,
    profile_for,
    _LOCAL_RESERVED_HEADROOM_TOKENS,
    _LOCAL_WORKING_MEMORY_FLOOR_TOKENS,
    _LOCAL_MAX_MODEL_LEN_FALLBACK,
)


class _FakeRuntime:
    """Minimal runtime double exposing only what profile_for() reads."""

    def __init__(self, is_local: bool = True, max_model_len: object = None):
        self.is_local = is_local
        if max_model_len is not None:
            self.max_model_len = max_model_len


class TestIsLocalPerBackend:

    def test_omlx_is_always_local(self):
        assert OMLXRuntimeClient().is_local is True

    def test_foundry_is_always_local(self):
        client = FoundryRuntimeClient(base_url="http://127.0.0.1:59999")
        assert client.is_local is True

    def test_ollama_local_model(self):
        client = OllamaRuntimeClient(chat_model="gemma4:e4b-mlx")
        assert client.is_local is True

    def test_ollama_cloud_model(self):
        client = OllamaRuntimeClient(chat_model="gemma4:31b-cloud")
        assert client.is_local is False


class TestContextProfileValues:

    def test_local_profile_budgets_from_max_model_len(self):
        """
        working_memory_tokens = max_model_len - reserved headroom (27_000),
        for a max_model_len comfortably above the floor.
        """
        runtime = _FakeRuntime(is_local=True, max_model_len=32_768)
        profile = profile_for(runtime)
        assert profile.working_memory_tokens == 32_768 - _LOCAL_RESERVED_HEADROOM_TOKENS
        assert profile.working_memory_limit is None

    def test_local_profile_floors_when_max_model_len_too_small(self):
        """
        A max_model_len small enough that the reservation would drive the
        budget to zero or negative must floor at 300, not go negative.
        """
        runtime = _FakeRuntime(is_local=True, max_model_len=8_192)
        profile = profile_for(runtime)
        assert profile.working_memory_tokens == _LOCAL_WORKING_MEMORY_FLOOR_TOKENS

    def test_local_profile_falls_back_when_max_model_len_missing(self):
        """
        A runtime client that doesn't expose max_model_len at all (e.g. an
        older client, or one that hasn't implemented this yet) must fall
        back to the conservative default, not raise.
        """
        runtime = _FakeRuntime(is_local=True)  # no max_model_len attribute
        profile = profile_for(runtime)
        expected = max(
            _LOCAL_MAX_MODEL_LEN_FALLBACK - _LOCAL_RESERVED_HEADROOM_TOKENS,
            _LOCAL_WORKING_MEMORY_FLOOR_TOKENS,
        )
        assert profile.working_memory_tokens == expected

    def test_local_profile_falls_back_when_max_model_len_not_an_int(self):
        """
        A non-int max_model_len (e.g. an unconfigured MagicMock attribute
        on a test double) must fall back to the conservative default
        rather than raising a TypeError during arithmetic.
        """
        runtime = _FakeRuntime(is_local=True, max_model_len="not-a-number")
        profile = profile_for(runtime)
        expected = max(
            _LOCAL_MAX_MODEL_LEN_FALLBACK - _LOCAL_RESERVED_HEADROOM_TOKENS,
            _LOCAL_WORKING_MEMORY_FLOOR_TOKENS,
        )
        assert profile.working_memory_tokens == expected

    def test_cloud_profile_removes_turn_cap(self):
        assert CLOUD_PROFILE.working_memory_limit is None

    def test_cloud_profile_budget_larger_than_local(self):
        local = profile_for(_FakeRuntime(is_local=True, max_model_len=32_768))
        assert CLOUD_PROFILE.working_memory_tokens > local.working_memory_tokens

    def test_profile_for_local(self):
        runtime = _FakeRuntime(is_local=True, max_model_len=32_768)
        profile = profile_for(runtime)
        assert profile.working_memory_limit is None
        assert profile is not CLOUD_PROFILE

    def test_profile_for_cloud(self):
        assert profile_for(_FakeRuntime(is_local=False)) is CLOUD_PROFILE

    def test_profile_for_defaults_to_local_when_is_local_absent(self):
        """
        A runtime double lacking `is_local` entirely must be treated as
        local (fail toward the more conservative tier), not silently
        upgraded to the cloud budget.
        """
        class _BareRuntime:
            pass

        profile = profile_for(_BareRuntime())
        assert profile.working_memory_tokens == _LOCAL_WORKING_MEMORY_FLOOR_TOKENS

    def test_context_profile_is_frozen(self):
        profile = ContextProfile(working_memory_tokens=1, working_memory_limit=1)
        try:
            profile.working_memory_tokens = 2  # type: ignore[misc]
            assert False, "ContextProfile must be immutable"
        except AttributeError:
            pass

    def test_context_profile_no_longer_has_total_context_tokens(self):
        """
        total_context_tokens was removed 2026-07-19 (dead parameter — see
        diagnostics/reports/ollama_cloud_num_ctx_findings.md). Guards
        against silently reintroducing it.
        """
        local = profile_for(_FakeRuntime(is_local=True, max_model_len=32_768))
        assert not hasattr(local, "total_context_tokens")
        assert not hasattr(CLOUD_PROFILE, "total_context_tokens")


class TestOmlxMaxModelLenIntegration:
    """
    Bridges Prompt 2 (omlx_runtime_client.py's max_model_len) and Prompt 3
    (context_profile.py reading it): a freshly constructed OMLXRuntimeClient
    that hasn't run health_check() yet exposes its own conservative
    fallback default, which profile_for() then floors to 300.
    """

    def test_fresh_omlx_client_before_health_check_floors_to_300(self):
        client = OMLXRuntimeClient()
        profile = profile_for(client)
        assert profile.working_memory_tokens == _LOCAL_WORKING_MEMORY_FLOOR_TOKENS
        assert profile.working_memory_limit is None
