"""
LORA — ContextProfile
=======================
Backend-tier-aware ceilings for how much conversation history flows into
each turn's prompt.

Layer placement
---------------
  ControllerAgent (_execute_plan)  →  ContextProfile  →  MemoryManager.get_context_window()
                                                       →  PromptBuilder.build()

Why two profiles, not one global constant
------------------------------------------
A cloud-hosted model (e.g. Gemma-4-31B via Ollama Cloud) runs on someone
else's hardware with a much larger real context window (~128K) than a
local Apple Silicon chat model; the host Mac's RAM is no longer the
constraint for the cloud tier, so there is no reason to truncate history to
the same degree as the local tier. `is_local` (see base_runtime_client.py)
is the single flag every ceiling below keys off — no per-backend special-
casing beyond that one flag.

Both profiles are budgeted against each tier's real native context window,
reserving headroom for the other prompt slots rather than spending the
literal ceiling:

LOCAL_PROFILE — budgeted against the oMLX-reported max_model_len for the
  currently active chat model (backend/omlx_runtime_client.py's
  health_check(), which exposes it live as `runtime.max_model_len` —
  falling back to its own conservative constant there when oMLX doesn't
  report the field, or before the first health check has run):
    - working_memory_tokens = max_model_len - _LOCAL_RESERVED_HEADROOM_TOKENS
      (27_000), floored at _LOCAL_WORKING_MEMORY_FLOOR_TOKENS (300). The
      27_000-token reservation covers the other slots' worst case: session
      files up to 20K (PromptBuilder._CEIL_SESSION_FILES_TOTAL),
      persona/RAG/tool/graph/working-state slots combined ~3K, output
      generation ~4K — the same breakdown CLOUD_PROFILE uses below, since
      those slot ceilings are shared PromptBuilder constants, not
      backend-tier-dependent. Unlike CLOUD_PROFILE there is no further
      "inexact context-length" safety margin stacked on top of that,
      because max_model_len is a value the server reports for the model it
      actually loaded, not an assumption about one.
    - The 300-token floor exists so a small or unreported max_model_len
      (e.g. omlx_runtime_client's own conservative fallback default, used
      before the first health check or on an oMLX version that never
      reports the field) can't drive the reservation past the window and
      leave working memory at zero or negative. 300 is the same figure
      this profile used historically under the retired RAM-tier approach
      (see "Prior approach" below), so that degraded case lands on a
      familiar, previously-live number rather than an untested new one.
    - working_memory_limit=None — same as CLOUD_PROFILE; the token budget
      is the only real limiter now. The previous fixed 5-turn cap wasn't
      derived from anything about the model or the machine, just carried
      over unexamined from the original 300-token/5-turn pair.
    - max_model_len is a live value read off the active runtime client at
      call time (it can change after a runtime-backend swap via
      POST /settings/runtime-backend, or once the first health check
      completes) — so, unlike before, LOCAL_PROFILE is no longer a fixed
      module-level constant. `profile_for(runtime)` takes the runtime
      client instance itself and resolves the profile fresh on every call
      — see its docstring.

CLOUD_PROFILE — budgeted against Gemma-4-31B's ~128K window, reserving
  headroom rather than spending the literal ceiling:
    - working_memory_tokens=60_000 is budgeted against that ~128K native
      ceiling with headroom reserved for the other slots' worst case
      (session files up to 20K, persona/RAG/tool/graph/working-state
      slots combined ~3K, output generation ~4K) plus a safety margin
      against an inexact context-length assumption.
    - working_memory_limit=None removes the turn-count cap entirely — the
      token ceiling above is the only thing trimming history under this
      profile, so a request with many *short* turns doesn't get cut off by
      turn count before it ever approaches the token budget.

Known trade-off, out of scope here: a 60K-token cloud prompt has a real
$/latency cost per request. This module does not add a cost-based
guardrail — RAM-bloat protection only ever applied to the local tier, and
under the cloud tier there's no RAM to protect. A cost/latency guardrail,
if ever wanted, is a separate concern.

Prior approach, retired: RAM-tiered lookup
--------------------------------------------
Before this change, LOCAL_PROFILE.working_memory_tokens was resolved at
import time from a hardcoded table keyed on detected host RAM
(`_VALIDATED_LOCAL_TIERS = {16: 300}`, measured 2026-07-19 — see
diagnostics/reports/local_working_memory_ram_findings.md for that
investigation's full data). That approach under-reports the real
constraint: oMLX runs as a separate process with its own paged KV cache
and memory enforcer that already manages RAM dynamically per-machine, so
host-RAM tiering was measuring the wrong layer. It's replaced outright by
the max_model_len-based budget above, not re-measured or extended to new
RAM tiers — the RAM-tier mechanism (`_VALIDATED_LOCAL_TIERS`,
`resolve_local_working_memory_tokens()`) is removed rather than kept as a
fallback path.

check_local_ram_headroom() below still exists, but purely as a startup
observability log line now (see its docstring) — nothing that computes
working_memory_tokens reads its return value any more.

2026-07-19 — `total_context_tokens` removed (dead parameter)
--------------------------------------------------------------
`total_context_tokens` only ever existed to feed `OllamaRuntimeClient`'s
`options.num_ctx` request field. `diagnostics/reports/
ollama_cloud_num_ctx_findings.md` confirmed `num_ctx` has zero observed
effect on Ollama (local or cloud, across a 25x sweep of values) — the
real enforced ceiling is always each model's hardcoded native context
length. Removed the field and stopped sending `num_ctx` entirely rather
than leave a dead parameter (and its now-unjustifiable "~28K headroom"
budget math) sitting in the profile as if it did something.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextProfile:
    working_memory_tokens: int          # PromptBuilder Slot 6 ceiling / MemoryManager max_tokens
    working_memory_limit:  int | None   # rows fetched by get_context_window() before token trim; None = unbounded


# ---------------------------------------------------------------------------
# LOCAL_PROFILE: budgeted from the active model's real max_model_len
# ---------------------------------------------------------------------------
# See the module docstring's "LOCAL_PROFILE" section for the full reasoning
# behind these two numbers.

# session files up to 20K (PromptBuilder._CEIL_SESSION_FILES_TOTAL) +
# persona/RAG/tool/graph/working-state slots combined ~3K + output
# generation ~4K.
_LOCAL_RESERVED_HEADROOM_TOKENS = 27_000

# Floor so a small/unreported max_model_len can't drive the budget to zero
# or negative — matches this profile's own pre-retirement value.
_LOCAL_WORKING_MEMORY_FLOOR_TOKENS = 300

# Used only when the active runtime doesn't expose a real max_model_len
# (missing attribute, or present but not an int — e.g. a MagicMock test
# double, or a runtime client that hasn't implemented this yet). Matches
# OMLXRuntimeClient's own _DEFAULT_MAX_MODEL_LEN fallback (see
# omlx_runtime_client.py) so an unreported model_len degrades to the same
# number regardless of which side (client or profile resolution) is the one
# missing the real figure.
_LOCAL_MAX_MODEL_LEN_FALLBACK = 8192


def _build_local_profile(max_model_len: int) -> ContextProfile:
    """
    Build the local-tier ContextProfile from the active chat model's real
    max_model_len, in place of the retired RAM-tier lookup (see module
    docstring's "Prior approach, retired" section).
    """
    working_memory_tokens = max(
        max_model_len - _LOCAL_RESERVED_HEADROOM_TOKENS,
        _LOCAL_WORKING_MEMORY_FLOOR_TOKENS,
    )
    return ContextProfile(
        working_memory_tokens = working_memory_tokens,
        working_memory_limit  = None,
    )


CLOUD_PROFILE = ContextProfile(
    working_memory_tokens = 60_000,
    working_memory_limit  = None,
)


def profile_for(runtime: object) -> ContextProfile:
    """
    Return the ContextProfile for a runtime client instance.

    Takes the runtime client itself (not just its `is_local` flag) because
    the local tier's budget now depends on a second live value,
    `runtime.max_model_len` (backend/omlx_runtime_client.py's
    health_check()) — this can change after a runtime-backend swap
    (POST /settings/runtime-backend) or once the first health check
    completes, so LOCAL_PROFILE is no longer a fixed module-level constant
    and must be resolved fresh on every call, never memoized.

    Both attributes are read via getattr with a default rather than direct
    access, so a runtime client that predates one of them (or a test
    double, e.g. a bare MagicMock) degrades gracefully instead of raising:
      - is_local defaults to True when absent, so any test double lacking
        the attribute keeps today's local-tier behavior rather than
        silently falling into the cloud tier.
      - max_model_len falls back to _LOCAL_MAX_MODEL_LEN_FALLBACK when
        absent, or when present but not an int (e.g. an unconfigured
        MagicMock attribute) — the same defensive `isinstance` check
        omlx_runtime_client.py's health_check() already applies to the
        raw value from GET /v1/models.
    """
    is_local = getattr(runtime, "is_local", True)
    if not is_local:
        return CLOUD_PROFILE

    raw_max_model_len = getattr(runtime, "max_model_len", _LOCAL_MAX_MODEL_LEN_FALLBACK)
    max_model_len = (
        raw_max_model_len
        if isinstance(raw_max_model_len, int)
        else _LOCAL_MAX_MODEL_LEN_FALLBACK
    )
    return _build_local_profile(max_model_len)


# ---------------------------------------------------------------------------
# Startup RAM guard (local tier only)
# ---------------------------------------------------------------------------
# See the module docstring's "Prior approach, retired" section for why this
# no longer feeds working_memory_tokens. Thresholds below are set from the
# 2026-07-19 RAM investigation's measured numbers (see
# diagnostics/reports/local_working_memory_ram_findings.md) — observability
# only, never raises or blocks startup, matching lifespan()'s existing
# reachable/not-reachable warn-and-continue pattern in main.py.

_AVAILABLE_GB_WARN_THRESHOLD = 8.0
_SWAP_PCT_WARN_THRESHOLD     = 40.0


def check_local_ram_headroom() -> dict:
    """
    Measure current real RAM headroom and flag whether it's already thin
    enough to risk swapping under ordinary background load.

    Startup observability only — logs a warning for an operator to see.
    Nothing that computes working_memory_tokens reads this return value;
    LOCAL_PROFILE is budgeted from the active model's max_model_len instead
    (see module docstring). Thresholds are set from the 2026-07-19 RAM
    investigation's measured numbers (available=6.60GB / swap=61.3%
    produced swapping throughout that probe), with margin added above them
    — not picked to always-fire or never-fire.

    Only meaningful for local backends (oMLX / non-cloud Ollama); callers
    should gate this on the active runtime's `is_local` flag.
    """
    import psutil

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    available_gb = vm.available / 1e9
    swap_pct = sw.percent

    warning = (
        available_gb < _AVAILABLE_GB_WARN_THRESHOLD
        or swap_pct > _SWAP_PCT_WARN_THRESHOLD
    )
    message = (
        f"available={available_gb:.2f}GB (warn below {_AVAILABLE_GB_WARN_THRESHOLD}GB), "
        f"swap={swap_pct:.1f}% (warn above {_SWAP_PCT_WARN_THRESHOLD}%)"
    )
    return {
        "available_gb": available_gb,
        "swap_pct": swap_pct,
        "warning": warning,
        "message": message,
    }
