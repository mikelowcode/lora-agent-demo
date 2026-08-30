"""
LORA — PromptBuilder
====================
Single point of prompt assembly for all LORA agents.
Implements the 7-slot prompt contract defined in §3 of
LOCALIST-Architecture.md. Every agent calls PromptBuilder.build();
no agent assembles its own prompt string.

This module has no dependencies on FastAPI, SQLite, or any runtime
client. It is pure Python and safe to import anywhere in the backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Input dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role:    str         # "user" | "assistant" | "tool"
    content: str
    label:   str | None = None   # tool name, if role == "tool"


@dataclass
class EpisodeBullet:
    content:      str
    episode_type: str
    confidence:   float


@dataclass
class UserProfileFact:
    content: str   # single fact line, already scored and selected by caller


@dataclass
class RagSource:
    path:    str
    content: str


@dataclass
class SessionFile:
    filename: str   # original filename, used as the label in the slot
    content:  str   # full extracted text; truncation enforced at render-time in slot builder
    source:   Literal["upload", "wiki_pin"] = "upload"


@dataclass
class ToolResult:
    tool_name:  str
    parameters: str
    result:     str
    success:    bool = True
    # Non-prompt-facing payload for tools whose full output must never reach
    # Slot 5 ([TOOL RESULTS]) — e.g. chart's png_path/chart_config, which
    # would blow the 500-token ceiling and aren't meant for the model to see
    # at all. _slot5_tools() only ever reads .tool_name/.parameters/.result,
    # so this field is naturally excluded from the prompt; callers that need
    # it (controller_agent.py, to populate ControllerResult.metadata) read it
    # directly off the ToolResult instead.
    artifact:   dict[str, Any] | None = None
    # Correlation key grouping every ToolResult produced by one bounded
    # multi-step tool call (currently only MCPToolDispatcher._run_research_loop)
    # into a single "workflow" — read by controller_agent.py to populate
    # ControllerResult.metadata["workflow_id"]/["workflow_steps"] for the
    # Episode Browsing UI's step-chain view. None for every other tool (a
    # single web_search or url_fetch is not a workflow on its own).
    workflow_id: str | None = None


@dataclass
class GraphLinkEntry:
    name:     str    # resolved page's stem, or raw link_text if unresolved
    resolved: bool   # True if this entry points to a real page


@dataclass
class GraphQueryResult:
    direction: str                  # "incoming" | "outgoing"
    page_name: str                  # display name of the resolved page
    links:     list[GraphLinkEntry] # may be empty — zero results is valid


@dataclass
class GraphNeighborhood:
    """
    1-hop graph neighborhood for the Phase A RAG-assist slot — distinct
    from GraphQueryResult (Planner P3c's explicit structural-query result).

    Opportunistic, not a query result: silently omitted when both lists are
    empty, unlike GraphQueryResult's "always render, even with zero edges"
    contract (that contract exists because a P3c query is a direct answer
    to an explicit question; this is supplementary context nobody asked
    for, so silence on "nothing to add" is correct here, not a regression).
    """
    page_name: str
    backlinks: list[GraphLinkEntry]
    outgoing:  list[GraphLinkEntry]


@dataclass
class WorkingMemoryState:
    current_project:  str | None = None
    active_artifacts: list[str]  = field(default_factory=list)


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Assembles the canonical 7-slot prompt defined in §3 of
    LOCALIST-Architecture.md.

    Slot layout
    -----------
    SYSTEM MESSAGE
      Slot 1a [identity]        — _SYSTEM constant; always present; ~50 tokens
      Slot 1b [persona]         — wiki persona doc; injected when provided;
                                  500-token ceiling; appended to system msg

    USER MESSAGE (static-first ordering for KV-cache prefix reuse)
      Slot DT [CURRENT DATETIME] — current local datetime + trust directive;
                                  always present, always first; unnumbered
                                  (see §3.2 — deliberately placed ahead of
                                  the stable-prefix ordering below for
                                  model salience, not cache reuse)
      Slot SF [SESSION FILES]   — uploaded file content; conditional; before
                                  Slot 3, after Slot DT
      Slot 3  [EPISODIC MEMORY] — durable facts; conditional; 150-token ceiling
      Slot 4  [CONTEXT]         — RAG snippets; conditional; 800-token ceiling
      Slot 4b [RELATED CONTEXT] — Phase A RAG-assist graph neighborhood (1-hop
                                  backlinks/outgoing of the top RAG hit, when
                                  it's a wiki graph node); conditional, clean
                                  omission on zero edges (unlike Slot 5b); flat
                                  500-char ceiling, not token-based — see
                                  _slot_graph_neighborhood()
      Slot 5  [TOOL RESULTS]    — tool output; conditional; 500-token ceiling
      Slot 5a [TOOL FAILED]     — failed tool calls; conditional; own
                                  150-token ceiling, kept separate from Slot 5
                                  so a verbose successful result can never
                                  crowd a failure signal out of the prompt
      Slot 5b [GRAPH RESULT]    — graph query result; emitted whenever a graph
                                  query resolved, even with zero edges;
                                  300-token ceiling
      Slot 6A [WORKING STATE]   — structured working state; conditional;
                                  100-token ceiling
      Slot 6  [WORKING MEMORY]  — recent turns; conditional; 300-token ceiling
      Slot 7  [INSTRUCTION]     — raw user instruction; always present; uncapped

    Slots are ordered most-stable-first so that the KV-cache prefix
    is maximally reused across consecutive turns. [INSTRUCTION] is always
    last because it changes on every turn.

    Token estimation: 1 token ≈ 4 characters (len(text) // 4).
    This is consistent with the convention used elsewhere in memory_manager.py.

    Returns
    -------
    build() → (system_prompt: str, user_prompt: str)
        system_prompt : Slot 1a + optional 1b. Pass as the `system=` argument
                        to runtime.
        user_prompt   : Slots 3–7 assembled in order. Most empty slots are
                        cleanly omitted (no label, no whitespace placeholder).
                        Exception: Slot 5b ([GRAPH RESULT]) is always rendered
                        when graph_result is not None, even with zero links —
                        see _slot_graph() for the full contract.

    Opt-in third return value: passing `emit_structured_working_memory=True`
    omits Slot 6 ([WORKING MEMORY]) from user_prompt and instead returns it
    as a third tuple element — a trimmed, chronological list[Turn] — so a
    caller can send each turn as its own discrete message instead of
    flattened text. Default is False, which reproduces today's 2-tuple
    behavior exactly; see build()'s own docstring for the full contract.

    Invariants
    ----------
    - Stateless. Safe to call from multiple threads concurrently.
    - Callers pass full content; PromptBuilder enforces all token ceilings.
    - Empty optional slots produce no output whatsoever — not even a newline.
    - Slot 1a's template is a constant, never overridden by callers. The
      assistant name it's parameterized with is not a constant — it's a
      user setting (MemoryManager.get_assistant_name()) callers resolve and
      pass in as `assistant_name`; omitting it falls back to
      _DEFAULT_ASSISTANT_NAME.
    """

    # -----------------------------------------------------------------------
    # Slot 1a — canonical system prompt (§3.2, Slot 1)
    # Ceiling: 50 tokens = 200 chars. This value is 174 chars / ~43 tokens
    # at the default name's length; a longer configured name eats into that
    # budget but the ceiling stays advisory, same as before.
    # -----------------------------------------------------------------------
    _SYSTEM_TEMPLATE: str = (
        "You are {name}, a local research assistant. "
        "You reason carefully, cite your sources, and acknowledge when you "
        "don't know something. You do not simulate certainty."
    )
    _DEFAULT_ASSISTANT_NAME: str = "Localist"

    # -----------------------------------------------------------------------
    # Token ceilings (in tokens; multiply by 4 for char equivalent)
    # -----------------------------------------------------------------------
    _CEIL_SYSTEM:   int = 50    # hard; slot 1a is a constant so this is advisory
    _CEIL_PERSONA:  int = 500   # slot 1b; persona injected into system msg
    _CEIL_EPISODIC: int = 150   # slot 3a; episodic bullets
    _CEIL_PROFILE:  int = 100   # slot 3b; user profile facts
    _CEIL_RAG:      int = 800   # slot 4
    _CEIL_TOOL:     int = 1500   # slot 5
    _CEIL_TOOL_FAILURE: int = 150   # slot 5a; own budget — see _slot5a_tool_failures
    _CEIL_GRAPH:    int = 300   # slot 5b
    # Phase A [RELATED CONTEXT] slot — a flat 500-*char* slice, not a token
    # ceiling like every other slot above. Deliberately matches
    # MemoryManager.index_document()'s embedding-truncation convention
    # (content[:500]) rather than introducing a new token budget: this is
    # the same category of problem (bounding an auxiliary/supplementary
    # signal) already empirically tuned once, and re-deriving a token
    # ceiling here would just re-litigate a settled question. See
    # _slot_graph_neighborhood().
    _CEIL_GRAPH_NEIGHBORHOOD_CHARS: int = 500
    _CEIL_WORKING_STATE: int = 100   # slot 6a
    _CEIL_WORKING:       int = 300   # slot 6
    _CEIL_SESSION_FILES_EACH:  int = 4000   # per-file ceiling, tokens
    _CEIL_SESSION_FILES_TOTAL: int = 20000  # total slot ceiling, tokens

    # -----------------------------------------------------------------------
    # Internal token helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count as len(text) // 4."""
        return len(text) // 4

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """
        Hard-truncate text to max_tokens (estimated).
        Truncation appends '… [truncated]' and never cuts mid-word.
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars].rsplit(" ", 1)[0]
        return cut + "… [truncated]"

    # -----------------------------------------------------------------------
    # Slot builders (private)
    # -----------------------------------------------------------------------

    def _system_message(self, assistant_name: str | None = None) -> str:
        """
        Return Slot 1a's identity string, interpolated with assistant_name.

        Falls back to _DEFAULT_ASSISTANT_NAME when assistant_name is None
        or empty — callers that don't resolve a name (or have no
        MemoryManager to resolve one from) get today's default behavior.
        """
        return self._SYSTEM_TEMPLATE.format(
            name=assistant_name or self._DEFAULT_ASSISTANT_NAME
        )

    def _slot1_system(
        self,
        persona: str | None = None,
        assistant_name: str | None = None,
    ) -> str:
        """
        Return Slot 1: system prompt, optionally with persona appended.

        When persona is None or empty, returns the identity string only.
        Otherwise appends the truncated persona (500-token ceiling) raw,
        separated by a double newline — no label is added.
        """
        system = self._system_message(assistant_name)
        if not persona:
            return system
        truncated = self._truncate_to_tokens(persona, self._CEIL_PERSONA)
        return system + "\n\n" + truncated

    def _slot_datetime(self, current_datetime: datetime) -> str:
        """
        Return the [CURRENT DATETIME] slot: ISO-8601 timestamp + weekday
        (+ tz abbreviation when the datetime carries one), followed by a
        trust-hierarchy directive.

        Unconditional — always rendered, unlike every other slot in the
        user message. There is no "empty" current time. Positioned first
        in the user message (ahead of [SESSION FILES]) so it is the first
        thing the model reads, deliberately trading a small, unavoidable
        cache cost (this content changes every call and can never be
        cached regardless of position) for maximum salience against the
        model's training-cutoff prior. See LOCALIST-Architecture.md §3.2.
        """
        weekday  = current_datetime.strftime("%A")
        tz_label = current_datetime.tzname()
        tz_part  = f", {tz_label}" if tz_label else ""
        timestamp = current_datetime.isoformat(timespec="seconds")
        return (
            "[CURRENT DATETIME]\n"
            f"{timestamp} ({weekday}{tz_part})\n"
            "This is ground truth for \"now.\" Tool results dated at or "
            "after your training cutoff are not errors — trust this "
            "timestamp and tool output over your training prior."
        )

    def _slot_session_files(
        self,
        files: list[SessionFile] | None,
    ) -> str:
        """
        Return the [SESSION FILES] slot, or "" if files is None/empty.

        This slot is unnumbered and positioned before all other user-message
        slots (before Slot 3 / [EPISODIC MEMORY]) so that uploaded file
        content sits as early as possible in the KV-cache prefix. It is
        populated directly from the backend ephemeral session file cache and
        bypasses the Planner routing ladder entirely.

        Per-file ceiling: 4,000 tokens (16,000 chars).
        Total slot ceiling: 20,000 tokens.

        Each file is rendered as:
            [SESSION FILES]
            --- filename.ext ---
            {content}
            --- end filename.ext ---

        Files with source == "wiki_pin" get "(from the vault)" appended to
        the opening label only, so the model can distinguish a pinned wiki
        page from an ad hoc upload for citation purposes.

        Multiple files are separated by a single blank line.
        Truncation appends the standard '… [truncated]' marker via
        _truncate_to_tokens(), consistent with all other slot builders.
        """
        if not files:
            return ""

        rendered_files: list[str] = []
        total_tokens = 0

        for f in files:
            truncated = self._truncate_to_tokens(f.content, self._CEIL_SESSION_FILES_EACH)
            label = f"{f.filename} (from the vault)" if f.source == "wiki_pin" else f.filename
            block = f"--- {label} ---\n{truncated}\n--- end {f.filename} ---"
            block_tokens = self._estimate_tokens(block)
            if total_tokens + block_tokens > self._CEIL_SESSION_FILES_TOTAL:
                break
            rendered_files.append(block)
            total_tokens += block_tokens

        if not rendered_files:
            return ""

        body = "\n\n".join(rendered_files)
        return f"[SESSION FILES]\n{body}"

    def _slot3_combined(
        self,
        bullets:       list[EpisodeBullet] | None,
        profile_facts: list[UserProfileFact] | None,
    ) -> str:
        """
        Return Slot 3: episodic memory block and/or user profile facts,
        or "" if both are None/empty.

        Two independent sub-budgets:
          Episodic bullets : 150-token ceiling (unchanged)
          User profile     : 100-token ceiling (new)

        Each sub-block is omitted cleanly when empty.
        Combined output is returned as a single string.

        Format:
            [EPISODIC MEMORY]
            - {content} ({episode_type}, {confidence:.1f})

            [USER PROFILE]
            - {fact line}
        """
        parts: list[str] = []

        # -- Episodic sub-block (150-token ceiling) ---------------------------
        if bullets:
            lines = ["[EPISODIC MEMORY]"]
            for bullet in bullets:
                line = (
                    f"- {bullet.content} "
                    f"({bullet.episode_type}, {bullet.confidence:.1f})"
                )
                candidate = "\n".join(lines + [line])
                if self._estimate_tokens(candidate) > self._CEIL_EPISODIC:
                    break
                lines.append(line)
            if len(lines) > 1:
                parts.append("\n".join(lines))

        # -- User profile sub-block (100-token ceiling) -----------------------
        if profile_facts:
            lines = ["[USER PROFILE]"]
            max_chars = self._CEIL_PROFILE * 4
            used_chars = len("[USER PROFILE]\n")
            for fact in profile_facts:
                line = f"- {fact.content}"
                if used_chars + len(line) + 1 > max_chars:
                    break
                lines.append(line)
                used_chars += len(line) + 1
            if len(lines) > 1:
                parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def _slot4_rag(
        self,
        sources: list[RagSource] | None,
    ) -> str:
        """
        Return Slot 4: RAG context block, or "" if None/empty.

        Format:
            [CONTEXT]
            Source: {path}
            {content snippet}

            Source: {path}
            {content snippet}

        The 800-token ceiling is enforced across all sources combined.
        Each source's content is truncated at a sentence boundary when
        possible. Maximum 3 sources (callers should pre-rank; this method
        takes the first 3).
        """
        if not sources:
            return ""

        MAX_SOURCES = 3
        max_chars   = self._CEIL_RAG * 4

        lines   = ["[CONTEXT]"]
        budget  = max_chars - len("[CONTEXT]\n")

        for source in sources[:MAX_SOURCES]:
            header  = f"Source: {source.path}"
            content = source.content.strip()

            # Truncate content at sentence boundary
            entry_budget = budget - len(header) - 2  # 2 for newlines
            if entry_budget <= 0:
                break
            if len(content) > entry_budget:
                # Try to cut at last sentence boundary within budget
                truncated = content[:entry_budget]
                last_period = max(
                    truncated.rfind("."),
                    truncated.rfind("!"),
                    truncated.rfind("?"),
                )
                if last_period > entry_budget // 2:
                    content = truncated[: last_period + 1] + " … [truncated]"
                else:
                    content = truncated + "… [truncated]"

            block  = f"{header}\n{content}"
            lines.append(block)
            budget -= len(block) + 1   # +1 for separator newline

            if budget <= 0:
                break

        if len(lines) == 1:
            return ""

        return "\n\n".join(lines[:1] + lines[1:])

    def _slot5_tools(
        self,
        tool_results: list[ToolResult] | None,
    ) -> str:
        """
        Return Slot 5: tool results block, or "" if None/empty.

        Format:
            [TOOL RESULTS]
            {tool_name}({parameters}):
              {result}

        The 1500-token ceiling is enforced. Each result is truncated to fit
        within the remaining budget before being appended.
        """
        if not tool_results:
            return ""

        max_chars = self._CEIL_TOOL * 4
        lines     = ["[TOOL RESULTS]"]
        budget    = max_chars - len("[TOOL RESULTS]\n")

        for tr in tool_results:
            header = f"{tr.tool_name}({tr.parameters}):"
            result = tr.result.strip()

            entry_budget = budget - len(header) - 4  # 4 for "\n  " + "\n"
            if entry_budget <= 0:
                break
            if len(result) > entry_budget:
                result = result[:entry_budget] + "… [truncated]"

            block  = f"{header}\n  {result}"
            lines.append(block)
            budget -= len(block) + 1

            if budget <= 0:
                break

        if len(lines) == 1:
            return ""

        return "\n".join(lines)

    def _slot5a_tool_failures(
        self,
        tool_failures: list[ToolResult] | None,
    ) -> str:
        """
        Return Slot 5a: tool failure block, or "" if None/empty.

        Deliberately separate from Slot 5 ([TOOL RESULTS]) rather than a
        raw "ERROR: ..." string folded into normal tool-result text — the
        model gets one consistent, unambiguous failure shape to learn to
        hedge against instead of an error string it might inconsistently
        narrate around.

        Format (one line per failure):
            [TOOL FAILED]
            {tool_name}({parameters}): FAILED — {reason}

        {reason} is tr.result with a leading "ERROR:" stripped.

        Given its own 150-token ceiling, deliberately separate from Slot
        5's 500-token budget: a verbose successful tool result should never
        be able to crowd a failure signal out of the prompt via shared
        budget exhaustion — the entire point of this slot is that it
        survives truncation pressure that ordinary Slot 5 entries do not.
        """
        if not tool_failures:
            return ""

        max_chars = self._CEIL_TOOL_FAILURE * 4
        lines     = ["[TOOL FAILED]"]
        budget    = max_chars - len("[TOOL FAILED]\n")

        for tr in tool_failures:
            reason = tr.result.strip()
            if reason.startswith("ERROR:"):
                reason = reason[len("ERROR:"):].strip()

            line = f"{tr.tool_name}({tr.parameters}): FAILED — {reason}"

            entry_budget = budget - 1  # 1 for separator newline
            if entry_budget <= 0:
                break
            if len(line) > entry_budget:
                line = line[:entry_budget] + "… [truncated]"

            lines.append(line)
            budget -= len(line) + 1

            if budget <= 0:
                break

        if len(lines) == 1:
            return ""

        return "\n".join(lines)

    def _slot_graph(self, graph_result: GraphQueryResult | None) -> str:
        """
        Return Slot 5b: graph query result, or "" if graph_result is None.

        This slot does NOT follow the clean-omission rule of other slots.
        When graph_result is not None the slot is always rendered — even when
        graph_result.links is an empty list. Zero edges is a real, correct
        answer that must be visible to the model. The only omission case is
        graph_result=None, meaning no graph query was resolved this turn.

        300-token ceiling enforced as a single post-render truncation.
        """
        if graph_result is None:
            return ""

        page  = graph_result.page_name
        links = graph_result.links

        if graph_result.direction == "incoming":
            if not links:
                content = f"No pages link to {page}."
            else:
                body    = "\n".join(f"- {e.name}" for e in links)
                content = f"Pages linking to {page}:\n{body}"
        else:  # outgoing
            if not links:
                content = f"{page} does not link to any other pages."
            else:
                resolved   = [e for e in links if e.resolved]
                unresolved = [e for e in links if not e.resolved]
                sections: list[str] = []
                if resolved:
                    res_body = "\n".join(f"- {e.name}" for e in resolved)
                    sections.append(f"{page} links to:\n{res_body}")
                if unresolved:
                    header = (
                        f"{page} also references a page that does not exist:"
                        if resolved
                        else f"{page} references a page that does not exist:"
                    )
                    unres_body = "\n".join(
                        f'- "{e.name}" (no matching page found)'
                        for e in unresolved
                    )
                    sections.append(f"{header}\n{unres_body}")
                content = "\n\n".join(sections)

        rendered = f"[GRAPH RESULT]\n{content}"
        return self._truncate_to_tokens(rendered, self._CEIL_GRAPH)

    def _slot_graph_neighborhood(
        self, neighborhood: GraphNeighborhood | None,
    ) -> str:
        """
        Return the Phase A [RELATED CONTEXT] slot, or "" to omit.

        Unlike _slot_graph() (Slot 5b), this slot follows the ordinary
        clean-omission rule: omitted when neighborhood is None *or* when
        both backlinks and outgoing are empty — a RAG turn whose top hit
        happens to be an unconnected wiki page should look exactly like a
        RAG turn with no graph data at all, not announce "zero edges" the
        way an explicit P3c query must.

        Truncation is a flat _CEIL_GRAPH_NEIGHBORHOOD_CHARS-char slice with
        no word-boundary trimming and no "…[truncated]" suffix — a
        deliberate deviation from _truncate_to_tokens()'s convention, to
        match memory_manager.index_document()'s embedding-truncation
        convention instead (see the constant's definition for the
        rationale). Label is distinct from Slot 5b's "[GRAPH RESULT]" so
        the model never confuses an opportunistic hint with a direct
        answer to a structural question it asked.
        """
        if neighborhood is None:
            return ""
        if not neighborhood.backlinks and not neighborhood.outgoing:
            return ""

        page = neighborhood.page_name
        sections: list[str] = []
        if neighborhood.backlinks:
            body = "\n".join(f"- {e.name}" for e in neighborhood.backlinks)
            sections.append(f"Referenced by:\n{body}")
        if neighborhood.outgoing:
            body = "\n".join(f"- {e.name}" for e in neighborhood.outgoing)
            sections.append(f"Links to:\n{body}")
        content = "\n\n".join(sections)

        rendered = f"[RELATED CONTEXT: {page}]\n{content}"
        return rendered[: self._CEIL_GRAPH_NEIGHBORHOOD_CHARS]

    def _slot6a_working_state(
        self,
        state: WorkingMemoryState | None,
    ) -> str:
        """
        Return Slot 6A: structured working state, or "" if state is None/empty.

        Clean-omission: returns "" when state is None or both current_project
        is falsy and active_artifacts is empty — no label, no placeholder.

        active_artifacts is truncated by dropping entries from the end until
        the block fits within the 100-token ceiling. current_project is emitted
        as-is (ceiling is soft for this single line).

        Format:
            [WORKING STATE]
            current_project: {current_project}
            active_artifacts: {artifact1}, {artifact2}, ...

        Each line is only emitted when its field is non-empty/non-None.
        """
        if state is None:
            return ""
        if not state.current_project and not state.active_artifacts:
            return ""

        lines = ["[WORKING STATE]"]

        if state.current_project:
            lines.append(f"current_project: {state.current_project}")

        if state.active_artifacts:
            artifacts = list(state.active_artifacts)
            while artifacts:
                line = f"active_artifacts: {', '.join(artifacts)}"
                candidate = "\n".join(lines + [line])
                if self._estimate_tokens(candidate) <= self._CEIL_WORKING_STATE:
                    lines.append(line)
                    break
                artifacts.pop()

        if len(lines) == 1:
            return ""

        return "\n".join(lines)

    def _trim_working_memory(
        self,
        turns:   list[Turn] | None,
        ceiling: int | None = None,
    ) -> list[Turn]:
        """
        Return the chronologically-ordered (oldest surviving first) subset
        of `turns` that fits within Slot 6's token ceiling — the exact same
        drop-oldest-first algorithm _slot6_working_memory() uses to build
        its flattened text, factored out so build()'s structured-turns
        return path (see `emit_structured_working_memory`) trims identically
        without ever rendering to text.

        Each turn is formatted the same way (`Turn {offset} [{role}]:
        {content}`) purely to measure size for the ceiling check; the
        returned list is the original Turn objects, not formatted strings.

        ceiling : None (default) uses the class's `_CEIL_WORKING` constant
            (300). Same parameter contract as _slot6_working_memory()'s
            `ceiling` — see that method's docstring.

        Returns [] if turns is None or empty.
        """
        if not turns:
            return []

        formatted: list[str] = []
        for i, turn in enumerate(turns):
            offset = -(len(turns) - i)   # -N … -1
            if turn.role == "tool" and turn.label:
                role_str = f"tool:{turn.label}"
            else:
                role_str = turn.role
            formatted.append(f"Turn {offset} [{role_str}]: {turn.content}")

        surviving = list(turns)
        effective_ceiling = ceiling if ceiling is not None else self._CEIL_WORKING
        max_chars = effective_ceiling * 4
        while formatted:
            total = sum(len(f) for f in formatted)
            if total <= max_chars:
                break
            formatted.pop(0)
            surviving.pop(0)

        return surviving

    def _slot6_working_memory(
        self,
        turns:   list[Turn] | None,
        ceiling: int | None = None,
    ) -> str:
        """
        Return Slot 6: working memory block, or empty string if no turns.

        Format:
            [WORKING MEMORY]
            Turn -N [role]: content
            ...
            Turn -1 [role]: content

        Turns are listed chronologically (oldest surviving first, newest last).
        The token ceiling is enforced by dropping oldest turns first (see
        _trim_working_memory(), which this method delegates the trimming to).
        Each turn is formatted as a single line before ceiling enforcement.

        Parameters
        ----------
        ceiling :
            Token ceiling for this slot. None (default) uses the class's
            `_CEIL_WORKING` constant (300) — today's behavior, unchanged.
            Callers pass a larger value (e.g. from a cloud ContextProfile,
            context_profile.py) to let more history survive into the
            prompt; this must scale in lockstep with whatever `max_tokens`
            the caller already passed into `get_context_window()`, since
            this is a second, independent truncation pass — raising one
            without the other silently re-truncates back down.

        Returns "" if turns is None or empty.
        """
        surviving = self._trim_working_memory(turns, ceiling)
        if not surviving:
            return ""

        lines: list[str] = []
        for i, turn in enumerate(surviving):
            offset = -(len(surviving) - i)   # -N … -1
            if turn.role == "tool" and turn.label:
                role_str = f"tool:{turn.label}"
            else:
                role_str = turn.role
            lines.append(f"Turn {offset} [{role_str}]: {turn.content}")

        body = "\n".join(lines)
        return f"[WORKING MEMORY]\n{body}"

    def _slot7_instruction(self, instruction: str) -> str:
        """
        Return Slot 7: the raw user instruction. Always present; uncapped.

        Format:
            [INSTRUCTION]
            {instruction}
        """
        return f"[INSTRUCTION]\n{instruction}"

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def build(
        self,
        instruction:      str,
        current_datetime: datetime,
        session_files:    list[SessionFile]        | None = None,
        persona:          str | None              = None,
        assistant_name:   str | None              = None,
        episodic_summary: list[EpisodeBullet]     | None = None,
        profile_facts:    list[UserProfileFact]   | None = None,
        rag_snippets:     list[RagSource]          | None = None,
        tool_results:     list[ToolResult]         | None = None,
        tool_failures:    list[ToolResult]         | None = None,
        graph_result:     GraphQueryResult         | None = None,
        graph_neighborhood: GraphNeighborhood       | None = None,
        working_state:    WorkingMemoryState       | None = None,
        working_memory:   list[Turn]               | None = None,
        working_memory_ceiling: int                | None = None,
        emit_structured_working_memory: bool = False,
    ) -> tuple[str, str] | tuple[str, str, list[Turn]]:
        """
        Assemble the canonical 7-slot prompt.

        Parameters
        ----------
        instruction :
            The raw user instruction (Slot 7). Never transformed.
        current_datetime :
            The current local datetime (unnumbered [CURRENT DATETIME] slot),
            computed by the caller — never by PromptBuilder itself, so the
            builder stays free of a system-clock dependency and remains
            trivially testable with a fixed value. Required; always rendered
            first in the user message, ahead of session_files. Callers should
            pass `datetime.now().astimezone()` freshly on every build() call —
            never memoized/cached across turns.
        session_files :
            Uploaded session files ([SESSION FILES], unnumbered slot before
            Slot 3). Per-file ceiling 4,000 tokens; total slot ceiling
            20,000 tokens. Pass None or [] to omit.
        persona :
            Optional persona string (Slot 1b). Injected into the system
            message when provided; 500-token ceiling enforced.
        assistant_name :
            Optional assistant name (Slot 1a). Interpolated into the
            identity template — callers resolve this from
            MemoryManager.get_assistant_name() and pass it in; None falls
            back to _DEFAULT_ASSISTANT_NAME ("Localist").
        episodic_summary :
            Pre-sorted EpisodeBullet list (Slot 3a). Caller is responsible
            for priority ordering. Pass None or [] to omit.
        profile_facts :
            Pre-scored UserProfileFact list (Slot 3b). Caller is responsible
            for relevance scoring and line selection. Pass None or [] to omit.
            100-token sub-budget within Slot 3.
        rag_snippets :
            Ranked RagSource list (Slot 4). At most 3 used. Pass None or
            [] to omit.
        tool_results :
            ToolResult list in dispatch order (Slot 5). Pass None or []
            to omit.
        tool_failures :
            ToolResult list of failed tool calls (Slot 5a, [TOOL FAILED]).
            Rendered as {tool_name}({parameters}): FAILED — {reason},
            distinct from Slot 5's success-path formatting. Own 150-token
            ceiling, independent of Slot 5's 500-token budget. Pass None
            or [] to omit.
        graph_result :
            GraphQueryResult for the current turn (Slot 5b). When not None,
            always rendered — even with zero links. Pass None to omit.
        graph_neighborhood :
            GraphNeighborhood for the current turn ([RELATED CONTEXT],
            Phase A RAG-assist — see memory-graph-inference-plan). Rendered
            right after Slot 4 [CONTEXT]. Unlike graph_result, cleanly
            omitted whenever there is nothing to add (None, or zero
            backlinks and outgoing edges). Pass None to omit.
        working_state :
            Structured working state (Slot 6A). Deterministic; not inferred.
            Rendered between tool-results and working-memory slots.
            100-token ceiling; active_artifacts truncated from the end.
            Pass None to omit (zero-impact on output when absent).
        working_memory :
            Recent conversation turns (Slot 6). Oldest dropped first when
            the ceiling (see working_memory_ceiling) is exceeded. Pass
            None or [] to omit.
        working_memory_ceiling :
            Token ceiling for Slot 6. None (default) uses the class's
            300-token `_CEIL_WORKING` constant — unchanged local-tier
            behavior. Pass a ContextProfile's `working_memory_tokens`
            (context_profile.py) to scale this with the caller's runtime
            tier; must match whatever `max_tokens` the caller already
            passed into `MemoryManager.get_context_window()`, since that
            call and this ceiling are two independent truncation passes
            over the same data.
        emit_structured_working_memory :
            Opt-in, default False. When False, behavior is completely
            unchanged from before this parameter existed: Slot 6 flattens
            `working_memory` into the [WORKING MEMORY] block inside
            `user_prompt`, exactly as always, and build() returns the
            (system_prompt, user_prompt) 2-tuple below. When True, Slot 6
            is OMITTED from `user_prompt` entirely (so a caller that also
            wants the turns structured — e.g. to send each as its own
            message — doesn't get the same content twice), and build()
            returns a 3-tuple instead, with the trimmed working-memory
            turns as the third element. Every existing caller that doesn't
            pass this flag is unaffected — see Returns below.

        Returns
        -------
        (system_prompt, user_prompt) : tuple[str, str]
            Returned when emit_structured_working_memory=False (default).
            system_prompt : Slot 1a + optional 1b. Pass as `system=` to
                            the runtime.
            user_prompt   : [SESSION FILES] (when present) followed by
                            Slots 3–7 joined with double newlines.
                            Empty slots are cleanly absent. Slot 3 includes
                            an optional [USER PROFILE] sub-block (Slot 3b)
                            when profile_facts are provided. Slot 5b
                            ([GRAPH RESULT]) is always rendered when
                            graph_result is not None; see _slot_graph().
        (system_prompt, user_prompt, working_memory_turns) : tuple[str, str, list[Turn]]
            Returned when emit_structured_working_memory=True.
            system_prompt : same as above.
            user_prompt   : same as above, EXCEPT Slot 6 ([WORKING MEMORY])
                            is never included.
            working_memory_turns : the same turns Slot 6 would otherwise
                            have flattened, trimmed against the same
                            ceiling (working_memory_ceiling, or
                            `_CEIL_WORKING` when None), chronologically
                            ordered oldest-first. [] when `working_memory`
                            is None/empty or nothing survives the ceiling.
        """
        system_prompt = self._slot1_system(persona, assistant_name)

        working_memory_turns: list[Turn] | None = None
        if emit_structured_working_memory:
            working_memory_turns = self._trim_working_memory(working_memory, working_memory_ceiling)
            working_memory_slot = ""
        else:
            working_memory_slot = self._slot6_working_memory(working_memory, working_memory_ceiling)

        slots = [
            self._slot_datetime(current_datetime),      # unnumbered — always first
            self._slot_session_files(session_files),   # unnumbered — before Slot 3
            self._slot3_combined(episodic_summary, profile_facts),
            self._slot4_rag(rag_snippets),
            self._slot_graph_neighborhood(graph_neighborhood),
            self._slot5_tools(tool_results),
            self._slot5a_tool_failures(tool_failures),
            self._slot_graph(graph_result),
            self._slot6a_working_state(working_state),
            working_memory_slot,
            self._slot7_instruction(instruction),
        ]

        user_prompt = "\n\n".join(s for s in slots if s)

        if emit_structured_working_memory:
            return system_prompt, user_prompt, working_memory_turns
        return system_prompt, user_prompt
