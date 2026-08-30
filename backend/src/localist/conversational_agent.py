"""
Localist — ConversationalAgent (Primary RAG Engine)
================================================
The single reasoning path for all non-ingest user queries.

Architecture
------------
ConversationalAgent is the primary query engine for Localist.  Every query that
is not a wiki ingest flows here.  The pipeline is:

  1. Query MemoryManager — semantic retrieval over the full corpus
     (wiki pages + raw docs, cosine re-ranked when embeddings available)
  2. Select top-k results and inject into the system prompt as grounded context
  3. Single inference call → answer

This replaces the multi-step ResearchAgent pipeline entirely.  The model
(gemma-4-e4b-it-4bit) is strong enough to reason over retrieved wiki context
in a single call.  The structured wiki pages produced by WikiAgent provide
clean, well-organised source material.

Design principles
-----------------
- Single inference call.  No loops, no sub-queries, no synthesis step.
- Corpus-first.  Every query hits MemoryManager before the model.
- Graceful degradation.  If MemoryManager is absent or corpus is empty,
  the model answers from its own knowledge — no crash, no error.
- Wiki-first context.  query_corpus() searches both wiki and raw doc_types;
  wiki pages are preferred because they are structured and agent-verified.
- Source transparency.  AgentResult.output["sources"] lists every document
  path that contributed to the answer.

AgentInterface compliance
--------------------------
    name        → "conversational_agent"
    can_handle  → True for all non-ingest instructions
    run(subtask)→ AgentResult with output["answer"], output["sources"],
                  output["grounded"]

SubTask.context keys (all optional)
------------------------------------
    max_tokens     : int   — model max tokens.              Default 1024.
    temperature    : float — sampling temperature.          Default 0.3.
    max_results    : int   — total corpus docs to inject.   Default 4.
    wiki_threshold : int   — min wiki hits before raw       Default 2.
                             fallback is skipped entirely.
    system         : str   — override system prompt.        Default: _DEFAULT_SYSTEM.

AgentResult.output schema
--------------------------
    answer     : str       — the model's response
    sources    : list[str] — document paths used as context
    grounded   : bool      — True when corpus context was injected

query_corpus() contract (memory_manager.py)
--------------------------------------------
Returns list[DocumentResult] with __slots__:
    .name            str
    .path            Path
    .doc_type        str   — "wiki" or "raw"
    .content         str
    .relevance_score float
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from typing import Any

from . import session_files as _session_files

from .episodic_extractor import _log_infer_throughput
from .prompt_builder import PromptBuilder, RagSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fabricated tool-call detection (§8.8 Open Item 11)
# ---------------------------------------------------------------------------

_FABRICATED_TOOLCALL_PATTERN = re.compile(
    r"<\|?tool_?call.*?call:web.*?tool_?call\|>",
    re.IGNORECASE | re.DOTALL,
)


def _is_fabricated_toolcall(text: str) -> bool:
    """
    Detect the malformed tool-call-shaped string this model has been observed
    to emit on tools=[] turns (§8.8 Open Item 11). No real tool-calling
    contract exists in this codebase for any runtime client — any match is
    fabrication, never a legitimate format to parse or honour.
    """
    return bool(_FABRICATED_TOOLCALL_PATTERN.search(text))


# Substituted when fabricated tool-call syntax is detected.  Grounded=False
# and sources=[] are forced alongside this message — a substituted fallback
# is never grounded in real tool or RAG content.
_SEARCH_UNAVAILABLE_FALLBACK = (
    "I don't have live search results for that — here's what I know "
    "from training, which may be stale or incomplete."
)

# Substituted when the model returns a well-formed but empty completion
# (confirmed live: Ollama can return "done": true with zero content in
# between, typically when the model has no tool grounding for a query it
# structurally can't answer from training alone — e.g. a specific recent
# or future date). Deliberately a different string from
# _SEARCH_UNAVAILABLE_FALLBACK: that message promises training-knowledge
# content to follow ("here's what I know from training"), which would be
# a non-sequitur here — there is no content to follow. Grounded=False and
# sources=[] are forced alongside this message for the same reason as
# above. This string itself must never be empty — that's the actual
# invariant the empty-completion guard exists to protect.
_EMPTY_RESPONSE_FALLBACK = (
    "I wasn't able to put together an answer for that — could you "
    "rephrase, or ask again?"
)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

# Instructions containing these keywords are ingest operations — they belong
# to WikiAgent.  can_handle() returns False for these so the Planner routes
# them to wiki_agent instead.
_INGEST_KEYWORDS: frozenset[str] = frozenset({
    "ingest", "update wiki", "create page", "apply diff",
})


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Characters of each document's content to include in the prompt.
# 2000 chars × 4 docs ≈ 8000 chars of context — comfortable for
# gemma-4-e4b-it-4bit alongside a system prompt.
_MAX_SNIPPET_CHARS = 2000

_PROMPT_BUILDER = PromptBuilder()


# ---------------------------------------------------------------------------
# ConversationalAgent
# ---------------------------------------------------------------------------

class ConversationalAgent:
    """
    Corpus-aware single-inference agent.  Primary query engine for Localist.

    Parameters
    ----------
    runtime :
        A RuntimeClient instance (OMLXRuntimeClient or FoundryRuntimeClient).
    memory_manager :
        MemoryManager instance.  When provided, query_corpus() is called
        before every inference to inject grounded wiki context.
        When None, the model answers without corpus context.
    project_root :
        Unused; present for constructor-signature parity with WikiAgent.
    """

    def __init__(
        self,
        runtime:        Any,
        memory_manager: Any | None = None,
        project_root:   Any | None = None,
    ) -> None:
        self._runtime        = runtime
        self._memory_manager = memory_manager
        self._project_root   = project_root  # parity only; not used

    # -----------------------------------------------------------------------
    # AgentInterface
    # -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "conversational_agent"

    def can_handle(self, instruction: str) -> bool:
        """
        Return True for all non-ingest instructions.

        ConversationalAgent is the default handler — it accepts everything
        except explicit wiki ingest requests (which belong to WikiAgent).
        """
        lowered = instruction.lower()
        return not any(kw in lowered for kw in _INGEST_KEYWORDS)

    def run(self, subtask: Any) -> Any:
        """
        RAG pipeline: retrieve → inject → infer → return.

        Steps
        -----
        1. Extract parameters from subtask.context.
        2. Query MemoryManager for top-k relevant documents.
        3. Build context block from retrieved documents.
        4. Assemble final prompt (context + question).
        5. Single runtime.infer() call.
        6. Return AgentResult.
        """
        from .controller_agent import AgentResult, TaskStatus

        instruction = subtask.instruction
        context     = subtask.context or {}

        max_tokens     = int(context.get("max_tokens",   1024))
        temperature    = float(context.get("temperature",  0.3))
        max_results    = int(context.get("max_results",   4))
        assistant_name = (
            self._memory_manager.get_assistant_name()
            if self._memory_manager is not None else None
        )
        system = str(
            context.get("system") or _PROMPT_BUILDER._system_message(assistant_name)
        )

        logger.info(
            "ConversationalAgent.run() — subtask=%s  chars=%d  max_results=%d  session_files=%d",
            subtask.subtask_id, len(instruction), max_results,
            len(_session_files.get_files()),
        )

        # -- Prebuilt prompt passthrough (Phase 4 controller integration) --------
        # When the controller has pre-assembled a full 6-slot prompt via
        # PromptBuilder, it passes it here. Skip internal RAG and prompt
        # assembly entirely — use the prebuilt prompt directly.
        on_token         = context.get("_on_token")
        prebuilt_prompt  = context.get("_prebuilt_prompt")
        prebuilt_system  = context.get("_prebuilt_system")
        # Only ever set by controller_agent.py when the active runtime is
        # OMLXRuntimeClient (see its emit_structured_working_memory wiring,
        # prompt_builder.py). Every other runtime path leaves this key
        # absent, so the kwarg below is simply never passed for them —
        # their infer()/infer_stream() signatures are untouched.
        prebuilt_working_memory_turns = context.get("_prebuilt_working_memory_turns")

        if prebuilt_prompt is not None:
            logger.debug(
                "ConversationalAgent: using prebuilt prompt from controller "
                "(chars=%d).", len(prebuilt_prompt),
            )
            infer_kwargs: dict[str, Any] = dict(
                prompt      = prebuilt_prompt,
                system      = prebuilt_system or system,
                max_tokens  = max_tokens,
                temperature = temperature,
                label       = "main_dispatch",
            )
            if prebuilt_working_memory_turns is not None:
                infer_kwargs["working_memory_turns"] = prebuilt_working_memory_turns
            try:
                if on_token is not None:
                    chunks: list[str] = []
                    _t0 = time.monotonic()
                    for chunk in self._runtime.infer_stream(**infer_kwargs):
                        on_token(chunk)
                        chunks.append(chunk)
                    answer = "".join(chunks)
                    _log_infer_throughput("main_dispatch", time.monotonic() - _t0, len(answer))
                else:
                    answer = self._runtime.infer(**infer_kwargs)
            except Exception as exc:
                logger.error(
                    "ConversationalAgent: inference failed (prebuilt path) "
                    "for subtask %s: %s", subtask.subtask_id, exc,
                )
                return AgentResult(
                    subtask_id = subtask.subtask_id,
                    agent_name = self.name,
                    status     = TaskStatus.FAILED,
                    output     = {},
                    error      = f"Inference error: {exc}",
                )
            if _is_fabricated_toolcall(answer):
                logger.warning(
                    "ConversationalAgent: fabricated tool-call syntax detected "
                    "(prebuilt path) for subtask %s — %d raw chars discarded.",
                    subtask.subtask_id, len(answer),
                )
                return AgentResult(
                    subtask_id = subtask.subtask_id,
                    agent_name = self.name,
                    status     = TaskStatus.COMPLETE,
                    output     = {
                        "answer":   _SEARCH_UNAVAILABLE_FALLBACK,
                        "sources":  [],
                        "grounded": False,
                    },
                )
            return AgentResult(
                subtask_id = subtask.subtask_id,
                agent_name = self.name,
                status     = TaskStatus.COMPLETE,
                output     = {
                    "answer":   answer,
                    "sources":  context.get("_prebuilt_sources", []),
                    "grounded": bool(context.get("_routing", {}).get("fetch_rag")),
                },
            )
        # -- End prebuilt prompt passthrough ------------------------------------

        # -- Step 2: Corpus retrieval (wiki-first) ---------------------------
        #
        # Strategy:
        #   1. Query wiki pages only (structured, agent-verified content).
        #   2. If wiki results >= wiki_threshold, use them exclusively —
        #      raw docs would only add duplicate content.
        #   3. If wiki results < wiki_threshold, top up with raw docs to
        #      fill the remaining slots up to max_results.
        #
        # This prevents the duplicate-content problem that occurs when both
        # "project-outline.md" (wiki) and
        # "Project Outline.md" (raw) are returned for the same
        # query, wasting context window space on identical material.

        sources:  list[str] = []
        grounded: bool      = False
        results:  list[Any] = []

        # Minimum wiki hits before raw fallback is skipped entirely.
        wiki_threshold = int(context.get("wiki_threshold", 2))

        if self._memory_manager is not None:
            try:
                # -- Pass 1: wiki pages --
                wiki_results = self._memory_manager.query_corpus(
                    instruction,
                    max_results    = max_results,
                    use_embeddings = True,
                )
                wiki_results = [r for r in wiki_results if r.doc_type == "wiki"]

                if len(wiki_results) >= wiki_threshold:
                    # Wiki corpus is sufficient — use it exclusively.
                    results = wiki_results[:max_results]
                    logger.debug(
                        "Corpus: wiki-only path (%d wiki results, threshold=%d).",
                        len(wiki_results), wiki_threshold,
                    )
                else:
                    # -- Pass 2: top up with raw docs --
                    remaining = max_results - len(wiki_results)
                    raw_results = self._memory_manager.query_corpus(
                        instruction,
                        max_results    = remaining,
                        use_embeddings = True,
                    )
                    raw_results = [r for r in raw_results if r.doc_type == "raw"]
                    results = wiki_results + raw_results[:remaining]
                    logger.debug(
                        "Corpus: wiki+raw fallback path "
                        "(%d wiki, %d raw, threshold=%d).",
                        len(wiki_results), len(raw_results), wiki_threshold,
                    )

                if results:
                    for doc in results:
                        sources.append(str(doc.path))
                    grounded = True
                    logger.debug(
                        "Corpus: %d doc(s) injected — %s",
                        len(sources), [os.path.basename(s) for s in sources],
                    )
                else:
                    logger.debug(
                        "Corpus: no results (max_results=%d) "
                        "— answering without context.",
                        max_results,
                    )

            except Exception as exc:
                logger.warning(
                    "ConversationalAgent: corpus query failed (%s) — "
                    "proceeding without context.", exc,
                )

        # -- Step 3: Assemble prompt via PromptBuilder -----------------------
        rag_sources = (
            [RagSource(path=str(doc.path), content=doc.content[:_MAX_SNIPPET_CHARS])
             for doc in results]
            if results else None
        )
        system, prompt = _PROMPT_BUILDER.build(
            instruction      = instruction,
            current_datetime = datetime.now().astimezone(),
            session_files    = _session_files.get_files(),
            rag_snippets     = rag_sources,
            assistant_name   = assistant_name,
        )

        # -- Step 4: Inference -----------------------------------------------
        try:
            if on_token is not None:
                chunks = []
                _t0 = time.monotonic()
                for chunk in self._runtime.infer_stream(
                    prompt      = prompt,
                    system      = system,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                    label       = "main_dispatch",
                ):
                    on_token(chunk)
                    chunks.append(chunk)
                answer = "".join(chunks)
                _log_infer_throughput("main_dispatch", time.monotonic() - _t0, len(answer))
            else:
                answer = self._runtime.infer(
                    prompt      = prompt,
                    system      = system,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                    label       = "main_dispatch",
                )
        except Exception as exc:
            logger.error(
                "ConversationalAgent: inference failed for subtask %s: %s",
                subtask.subtask_id, exc,
            )
            return AgentResult(
                subtask_id = subtask.subtask_id,
                agent_name = self.name,
                status     = TaskStatus.FAILED,
                output     = {},
                error      = f"Inference error: {exc}",
            )

        if _is_fabricated_toolcall(answer):
            logger.warning(
                "ConversationalAgent: fabricated tool-call syntax detected "
                "(legacy RAG path) for subtask %s — %d raw chars discarded.",
                subtask.subtask_id, len(answer),
            )
            return AgentResult(
                subtask_id = subtask.subtask_id,
                agent_name = self.name,
                status     = TaskStatus.COMPLETE,
                output     = {
                    "answer":   _SEARCH_UNAVAILABLE_FALLBACK,
                    "sources":  [],
                    "grounded": False,
                },
            )

        # Empty-completion floor (legacy RAG path only — the prebuilt-prompt
        # path above is exclusively reached via ControllerAgent._execute_plan(),
        # which owns a smarter forced-tool retry before falling back; this
        # path has no tool-dispatch capability of its own and nothing wraps
        # it with a retry, so a single substitution is the best available
        # guarantee that TaskStatus.COMPLETE never carries an empty answer).
        if not answer.strip():
            logger.warning(
                "ConversationalAgent: empty completion (legacy RAG path) "
                "for subtask %s — substituting fallback.", subtask.subtask_id,
            )
            return AgentResult(
                subtask_id = subtask.subtask_id,
                agent_name = self.name,
                status     = TaskStatus.COMPLETE,
                output     = {
                    "answer":   _EMPTY_RESPONSE_FALLBACK,
                    "sources":  [],
                    "grounded": False,
                },
            )

        sources += [f"session://{sf.filename}" for sf in _session_files.get_files()]

        logger.info(
            "ConversationalAgent.run() complete — answer_chars=%d  "
            "grounded=%s  sources=%d",
            len(answer), grounded, len(sources),
        )

        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = self.name,
            status     = TaskStatus.COMPLETE,
            output     = {
                "answer":   answer,
                "sources":  sources,
                "grounded": grounded,
            },
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _path_to_title(path: str) -> str:
    """
    Convert a file path to a readable title.

    "wiki/project-outline.md"        → "Project Outline"
    "/abs/path/raw/Build Order.md"   → "Build Order"
    """
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    name = name.replace("-", " ").replace("_", " ")
    words = [w if w.isupper() and len(w) > 1 else w.title() for w in name.split()]
    return " ".join(words)