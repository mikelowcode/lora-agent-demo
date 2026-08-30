"""
Localist — MCP Tool Dispatcher
============================
Was originally a drop-in replacement for the now-deleted ToolDispatcher
(tool_dispatcher.py) at the controller_agent.py dispatch seam; as of Phase
4 (cleanup, 2026-07-03) that legacy class is gone entirely — "file_op",
"url_fetch", "web_search", "research", "chart", "news_search",
"github_search"/"github_read"/"github_release", and "hacker_news_search"
are the only tool names Planner ever routes to tools_to_call (see
planner.py's P3/P3-news/P3-github/P3-github-release/P3-hacker-news/P3b),
and all are served over the
localist-mcp service (mcp_server/, port 8003) via the MCP SSE transport
(research is a client-side loop over the same web_search/url_fetch MCP
tools, not a distinct MCP tool of its own). Any other tool name is
unrecognized and produces an inline error ToolResult — the one remaining
piece of what used to be ToolDispatcher's "else" branch, ported inline
rather than kept as an excuse to hold onto a whole extra class.

github_search/github_read/github_release: public-repo GitHub REST reads
(search, file/README/directory content, and release notes) — the GitHub
counterpart to news_search, minus a fallback tier (GitHub's Search API
has no NewsAPI-style miss/error split worth a second provider).
github_search and github_release are routed by Planner keyword gates
(P3-github, P3-github-release); github_read is reached only when a
caller supplies context["github_repo"] directly (same
caller-supplied-pin convention as news_search's context["news_article_url"]),
since Planner's keyword gates can only detect that a GitHub-shaped
request was made, not which repo — github_release works around this for
its own case by chaining a github_search call to resolve a bare project
name to owner/repo (see _run_github_release). See mcp_server/github.py
for the tool implementations — all three are read-only (GET-only), no
archive/CLI paths.

news_search (news-query-routing plan, 2026-07-22): tries the NewsAPI-backed
news_search MCP tool first; on a miss (NewsAPI empty/errored result — see
mcp_server/news_search.py's is_miss) or on localist-mcp being unreachable,
falls through to the same _execute_web_search_query() helper web_search
already uses, reusing the original derived query rather than re-deriving a
generic one. Both the failed tier-1 attempt and the fallback attempt are
returned (not just the winner) so Slot 5 and controller_agent.py's Step 3b
corpus fallback see the full picture — same "return every ToolResult
produced, not just the winning one" convention _run_research_loop already
established. The fallback ToolResult is retagged tool_name="news_search:
brave_fallback" for provenance (so a transcript/log makes it visible when
NewsAPI's key/quota needs attention); Step 3b's web_search-failure check
was extended with an `r.tool_name.startswith("news_search")` clause to
keep matching it after the rename, mirroring the `or r.tool_name ==
"research"` clause already added there for the same reason.

url_fetch (Phase 2): extracts the first http(s):// URL from the
instruction (or context["fetch_url"] if already resolved upstream), calls
the fetch_url MCP tool, and formats the result the same way the legacy
ToolDispatcher._run_url_fetch did. This retired the standalone Fetcher
microservice (port 8002) — fetch_url ports its /extract path in-process on
localist-mcp instead.

url_fetch / Ollama Cloud (ollama-web-search-mcp-tool-scoping.md §5,
2026-07-31): Ollama Cloud's web_fetch API is a fallback-only tier
underneath fetch_url — no primary mode, no config toggle, no Planner
visibility (§5.4), unlike web_search's WEB_SEARCH_PROVIDER. When
_execute_fetcher_url_fetch's result has success=False — and the failure
isn't localist-mcp itself being unreachable (_MCP_UNREACHABLE_PREFIX, same
suppression rule as web_search's fallback) — _run_url_fetch falls through
to the shared _run_ollama_url_fetch() helper, retagging the result
tool_name="url_fetch:ollama_fallback". Note there is no distinct
"successful but empty" case to detect here the way web_search has
result_count==0: mcp_server/url_fetch.py's _extract() already raises
ValueError on empty extraction (paywalls/login walls), which surfaces as
is_error=True indistinguishable from any other tool-level failure by the
time it reaches this dispatcher — so the fallback trigger is simply "not
success", confirmed by reading the actual extraction code rather than
assumed. ollama_web_fetch's MCP-layer response is pre-normalized to
fetch_url's own dict shape (mcp_server/ollama_web_fetch.py), so both
tiers share one result-formatting helper (_format_fetch_result_text) —
nothing downstream needs to branch on which tier answered. Since
_run_url_fetch is the single call point shared by the direct "url_fetch"
tool dispatch, _enrich_top_result's web_search follow-up fetch, and
_run_research_loop's candidate-URL fetch, all three transparently gain
Ollama coverage from this one change.

web_search (Phase 3): ports ToolDispatcher._run_web_search's query
resolution verbatim (explicit context["web_search_queries"], else derived
from the instruction) and calls the web_search MCP tool once per query, up
to _MAX_WEB_QUERIES. Locked decision: the legacy runtime.infer()
hallucination fallback for a missing LANGSEARCH_API_KEY is gone — that
path now produces a clean success=False ToolResult, same as any other
web_search failure, so controller_agent.py's existing corpus fallback
(Step 3b) is what grounds the answer instead.

web_search / Ollama Cloud (ollama-web-search-mcp-tool-scoping.md,
2026-07-31): Ollama Cloud is a second provider underneath web_search,
selected by WEB_SEARCH_PROVIDER (default "brave", read fresh on every
call — see _execute_web_search_query). "ollama" makes it the sole
primary — Brave is never called, works with BRAVE_API_KEY entirely
absent. "brave" (default) tries the existing web_search tool first and
falls through to the same _run_ollama_web_search() helper when Brave's
result is empty (result_count == 0) or failed, retagging the fallback
tool_name="web_search:ollama_fallback" (primary-Ollama results are
tagged "web_search:ollama_primary") — the Planner-facing tool identity
never changes; tools_to_call is still only ever "web_search". Since
_execute_web_search_query is the single per-query call point shared by
_run_web_search, _run_news_search's tier-2 fallback, and
_run_research_loop, all three transparently gained Ollama coverage from
one change.

research: a bounded search/evaluate/reformulate/fetch loop
(_run_research_loop) that Planner routes to instead of "web_search" when
the instruction's cosine similarity to planner.py's research_intent
template group clears _RESEARCH_INTENT_THRESHOLD (gated behind
LOCALIST_RESEARCH_LOOP_ENABLED, off by default) — for requests that need a
specific, extractable fact (price, spec, plan tier) run down rather than a
single search-and-answer. Up to _MAX_RESEARCH_ITERATIONS rounds of
web_search, each followed by a cheap runtime.infer() yes/no gate check
(and, if inconclusive, a url_fetch of the top candidate result re-checked
against the same gate) and, on failure, a runtime.infer() query
reformulation before retrying. Every ToolResult produced along the way is
returned — not just the winning one — so it drops into the same
dispatched_tool_results handling web_search already uses.

chart (_run_chart): promotes diagnostics/diag_shadow_chart_toolcall_v4_full.
py's measured pipeline to production — a bounded runtime.infer() call with
chart_tool_schema.SYSTEM_PROMPT_FEWSHOT at temperature=0.0 to extract
generate_chart arguments from the instruction, repaired via
json_envelope_repair.repair_envelope() and validated via
chart_tool_schema.validate_chart_arguments(); on a malformed envelope, one
retry at temperature=0.3 (final). On success, dispatches to the
generate_chart MCP tool and returns a ToolResult whose .result is only the
tool's "summary" field (png_path/chart_config ride in .artifact instead —
see prompt_builder.ToolResult.artifact — never in the prompt-facing
.result). On failure (post-retry still malformed, schema-invalid, or the
model legitimately declines), returns None rather than an ERROR-shaped
ToolResult — a deliberate deviation from every other tool here, see
_dispatch_async's chart branch for why.

Session lifecycle: dispatch() opens one MCP ClientSession (SSE transport)
and reuses it for every tool invocation made during that dispatch() call —
including multiple web_search queries and a research loop's internal
search/fetch calls — closing it on the way out regardless of outcome.
Session reuse is scoped to a single dispatch() call only; it is not
persisted across separate HTTP requests/dispatch() invocations (see
MCPToolDispatcher._dispatch_async's docstring).

Reference: §6 of LOCALIST-Architecture.md; Phase 1 MCP migration; Phase 2
url_fetch wiring + Fetcher retirement; Phase 3 web_search migration; Phase
4 cleanup (ToolDispatcher deletion); MCP follow-up (session reuse); research
loop addition (2026-07-16).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.sse import sse_client

from .chart_tool_schema import KNOWN_TOOL_NAMES, SYSTEM_PROMPT_FEWSHOT, validate_chart_arguments
from .json_envelope_repair import repair_envelope
from .prompt_builder import ToolResult

logger = logging.getLogger(__name__)

# localist-mcp service endpoint (standalone service on port 8003)
_MCP_SERVER_URL: str = os.environ.get(
    "LOCALIST_MCP_URL", "http://localhost:8003"
) + "/sse"

# file_op_action -> MCP tool name
_FILE_OP_TOOL_MAP: dict[str, str] = {
    "read":   "read_file",
    "write":  "write_file",
    "append": "append_file",
}

# Straightforward http(s):// URL extraction from an instruction — same
# pattern legacy ToolDispatcher._run_url_fetch used.
#
# 2026-07-16: ] and ) added to the excluded-character class after a live
# research loop run confirmed mcp_server/web_search.py's result formatting
# (f"• {title}\n  {body}\n  [{url}]") — every URL wrapped in literal
# [...] — caused this regex to capture the trailing "]" as part of the URL
# (e.g. ".../apple]"), which then 404'd when passed to url_fetch. Markdown
# links and parenthetical citations wrap URLs in ()/[] the same way, so
# this is a shared-regex fix, not a research-loop-specific one — this
# pattern also backs _run_url_fetch's instruction-text extraction, where a
# user pasting a bracket- or paren-wrapped URL would hit the identical bug,
# just not yet observed live.
_URL_RE = re.compile(r"https?://[^\s\"'>\]\)]+")

# github_release repo resolution — specifically the *bracketed* [{url}]
# line search_format.py's docstring documents as load-bearing, not "the
# first URL anywhere in the text" the way _URL_RE/_extract_first_url()
# scan (an existing, narrower vulnerability those two share: a hit's own
# description/summary text can legitimately contain an unrelated URL —
# e.g. a project linking its Twitter/docs page — that appears earlier in
# the string than the bracketed repo URL itself). Caught live, 2026-07-29:
# searching "httpie" surfaced a top hit whose description embedded
# "https://twitter.com/httpie" before its own "[https://github.com/
# httpie/http-prompt]" line, and _URL_RE.search() (unanchored) grabbed
# the Twitter URL first, which parses to only one path segment and
# correctly fails the owner/repo check — but for the wrong reason, and
# would have silently resolved to a bogus repo entirely if the stray URL
# had happened to have two path segments instead of one. Anchoring to the
# `[...]` wrapper specifically avoids this whole class of false match.
_GITHUB_SEARCH_BRACKETED_URL_RE = re.compile(r"\[(https?://[^\s\]]+)\]")

# Maximum number of web_search queries per dispatch call — same cap as
# legacy ToolDispatcher._run_web_search.
_MAX_WEB_QUERIES: int = 3

# Bounded research loop — hard cap on search+evaluate+reformulate cycles.
# Same rationale as _MAX_WEB_QUERIES: an unbounded loop against a live
# search provider is a cost and latency risk, not just a correctness one.
_MAX_RESEARCH_ITERATIONS: int = 3

# Cap on one enrichment fetch_url's contribution to Slot 5's shared token
# budget (see _enrich_top_result) — a full article's cleaned_text could
# otherwise dominate the whole [TOOL RESULTS] block on its own.
_ENRICH_EXCERPT_CHARS: int = 3000

# 2026-07-25: live testing showed a top search result whose top URL was
# bot-blocked (HTTP 403 from a news portal's anti-scraping defense) meant
# zero enrichment for that turn, even though the same result had other,
# perfectly fetchable URLs a few lines below. _enrich_top_result now tries
# up to this many candidate URLs (in result order) before giving up.
_ENRICH_MAX_ATTEMPTS: int = 3

# Chart argument extraction — matches diag_shadow_chart_toolcall_v4_full.py's
# measured pipeline exactly. _CHART_RETRY_TEMPERATURE is a second,
# independent inference sample on a MALFORMED_ENVELOPE first pass, not a
# repeat of the deterministic temperature=0.0 attempt.
_CHART_MAX_TOKENS:        int   = 400
_CHART_RETRY_TEMPERATURE: float = 0.3

# 2026-07-17: live testing showed a gate-check call inside
# _evaluate_pricing_gate (max_tokens=10) stall for the full 60s
# LOCALIST_STREAM_TIMEOUT before timing out — confirmed via logs that the
# Ollama daemon itself stayed responsive throughout (health-check polling
# to /api/tags kept succeeding every 15s during the stall), so this was a
# cloud-model-side stall, not a local hang. The 60s default is sized for
# the full 1024-token main-dispatch answer; a max_tokens=10/40 classifier
# call sharing that same budget means a stuck one burns a full minute
# before the loop can recover, when it should fail fast and let the loop
# reformulate instead. Applied only to _evaluate_pricing_gate and
# _reformulate_query — every other infer()/infer_stream() call site in the
# codebase keeps the default timeout unchanged.
_RESEARCH_CLASSIFIER_TIMEOUT: float = 15.0

_RESEARCH_GATE_SYSTEM_PROMPT: str = (
    "You are a fact-extraction classifier, not a conversational assistant. "
    "You will be given the ORIGINAL QUESTION the user asked, and a block of "
    "search-result or page TEXT. Decide whether the TEXT contains concrete, "
    "specific information that directly answers the ORIGINAL QUESTION — not "
    "just any pricing-shaped or spec-shaped content in general. A page that "
    "mentions a price or number for a DIFFERENT product, a different "
    "tier/trim than the one asked about, or that only says pricing exists "
    "without stating the number, does NOT count. If the question asks about "
    "a specific tier, trim, plan, or variant, the text must address THAT "
    "specific one, not a different one. Respond with exactly one word: yes "
    "or no."
)

_RESEARCH_REFORMULATE_SYSTEM_PROMPT: str = (
    "You are a search-query rewriter, not a conversational assistant. "
    "The previous web search did not surface concrete pricing information. "
    "Given the original request and the queries already tried, write ONE "
    "new, more specific search query likely to surface a pricing page "
    "with actual numbers (e.g. add \"pricing\", \"plans\", \"per month\", "
    "or the vendor's likely domain). Respond with the query text only, "
    "nothing else."
)

# Filler prefixes stripped when deriving a single query from the
# instruction — ported verbatim from ToolDispatcher._run_web_search.
_WEB_SEARCH_FILLER_PREFIXES: tuple[str, ...] = (
    "what are the ", "what is the ", "what is ", "find the ",
    "search for ", "look up ", "tell me about ",
)

# github_release — bare version/tag extraction from an instruction, e.g.
# "fetch the oMLX 0.5.3 release notes" -> "0.5.3". Optional leading "v"
# (some repos tag "v0.5.3", others "0.5.3" — github.py's github_release()
# retries with the other form on a 404, so either is fine here) followed
# by a 2- or 3-segment dotted numeric version.
_GITHUB_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b", re.IGNORECASE)

# github_release query derivation — two strategies, tried in order:
#
# 1. _GITHUB_RELEASE_SUBJECT_RE: a trailing "for/of/on/about X" object at
#    the very end of the instruction — the common "release notes FOR X"/
#    "changelog OF X"/"latest release ON X" phrasing, where the project
#    name comes *after* the release marker. Tried first since it's an
#    unambiguous, deliberately-named subject when present.
# 2. _GITHUB_RELEASE_MARKER_RE (fallback, see _derive_github_release_query):
#    truncate at the first release-marker/version match, assuming the
#    project name *precedes* it instead — "fetch the oMLX 0.5.3 release
#    notes ..." truncates before "0.5.3", leaving "oMLX" once filler is
#    stripped.
#
# Between the two, most natural phrasings of "get me the release notes
# for/of X" or "X's release notes" are covered. Neither is a general
# parser — a phrasing this doesn't handle (e.g. the project name and a
# version both trailing a marker in the same clause, "release notes for X
# 0.5.3") derives an imperfect query. Accepted as a known limitation, same
# posture as this file's other simple, heuristic derivation helpers
# (_derive_file_op_path, etc.).
_GITHUB_RELEASE_SUBJECT_RE = re.compile(
    r"\b(?:for|of|on|about|in)\s+([A-Za-z][\w.\-]*(?:\s+[A-Za-z][\w.\-]*){0,2})\s*[?.!]?\s*$",
    re.IGNORECASE,
)

_GITHUB_RELEASE_MARKER_RE = re.compile(
    r"\b(release notes|changelog|latest release|new release|release)\b",
    re.IGNORECASE,
)

# github_release query derivation — filler prefixes stripped from the
# candidate left after truncating at the first _GITHUB_RELEASE_MARKER_RE/
# _GITHUB_VERSION_RE match. Kept separate from _WEB_SEARCH_FILLER_PREFIXES
# (which web_search/news_search/research also use) rather than extending
# that shared list, since "fetch"/"get" phrasing is specific to how people
# ask for a release and has no reason to affect unrelated tools' query
# derivation.
_GITHUB_RELEASE_FILLER_PREFIXES: tuple[str, ...] = (
    "fetch the ", "fetch ", "get the ", "get ", "find the ", "look up ",
    "what are the ", "what is ", "show me ", "summarize the ", "summarize ",
    "check the ", "check ",
)

# file_op action derivation — keyword groups checked in this priority order
# (append > write > read: append/write are less ambiguous signals than a
# bare "read"). Checked against the lowercased instruction; only used when
# context["file_op_action"] is absent.
_FILE_OP_APPEND_KEYWORDS: tuple[str, ...] = ("append", "add to the file", "add this to")
_FILE_OP_WRITE_KEYWORDS:  tuple[str, ...] = ("write", "create", "save", "make a file")
_FILE_OP_READ_KEYWORDS:   tuple[str, ...] = ("read", "open", "show me the file", "what's in the file")

# file_op path derivation — patterns tried in order, first match wins; falls
# back to a bare filename token anywhere in the instruction. Only used when
# context["file_op_path"] is absent.
_FILE_OP_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"name it (?:.*?\s)?([\w\-]+\.\w+)", re.IGNORECASE),
    re.compile(r"call it (?:.*?\s)?([\w\-]+\.\w+)", re.IGNORECASE),
    re.compile(r"save (?:it |this )?as (?:.*?\s)?([\w\-]+\.\w+)", re.IGNORECASE),
)
_FILE_OP_PATH_FALLBACK_RE = re.compile(r"\b[\w\-]+\.\w+\b")

# file_op content derivation — patterns tried in order, first match wins.
# Only used when context["file_op_content"] is absent.
_FILE_OP_CONTENT_CODEBLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)
_FILE_OP_CONTENT_QUOTED_RE    = re.compile(r'"([^"]*)"|\'([^\']*)\'')
_FILE_OP_CONTENT_PHRASE_RE    = re.compile(
    r"(?:with the content|containing|that says)\s+(.*)$", re.IGNORECASE
)


def _derive_file_op_action(instruction: str) -> str:
    lowered = instruction.lower()
    if any(kw in lowered for kw in _FILE_OP_APPEND_KEYWORDS):
        return "append"
    if any(kw in lowered for kw in _FILE_OP_WRITE_KEYWORDS):
        return "write"
    if any(kw in lowered for kw in _FILE_OP_READ_KEYWORDS):
        return "read"
    return "read"


def _derive_file_op_path(instruction: str) -> str:
    for pattern in _FILE_OP_PATH_PATTERNS:
        match = pattern.search(instruction)
        if match:
            return match.group(1).strip()
    match = _FILE_OP_PATH_FALLBACK_RE.search(instruction)
    return match.group(0) if match else ""


def _derive_file_op_content(instruction: str) -> str:
    match = _FILE_OP_CONTENT_CODEBLOCK_RE.search(instruction)
    if match:
        return match.group(1).strip()
    match = _FILE_OP_CONTENT_QUOTED_RE.search(instruction)
    if match:
        return match.group(1) if match.group(1) is not None else match.group(2)
    match = _FILE_OP_CONTENT_PHRASE_RE.search(instruction)
    if match:
        return match.group(1).strip()
    return ""

# FastMCP wraps every raised tool exception as "Error executing tool <name>: <msg>"
# (mcp/server/fastmcp/tools/base.py). Our tool implementations always raise
# ValueError("ERROR: ..."), so stripping this wrapper recovers the exact
# "ERROR: ..." shape ToolDispatcher used to produce — which is what
# controller_agent.py's startswith("ERROR:") slot-6 filter matches on.
_FASTMCP_ERROR_WRAPPER_RE = re.compile(r"^Error executing tool \S+: ")


def _normalize_mcp_error_text(text: str) -> str:
    stripped = _FASTMCP_ERROR_WRAPPER_RE.sub("", text, count=1)
    return stripped if stripped.startswith("ERROR:") else text


# Literal prefix every "session is None" / "_call_mcp_tool raised" branch in
# this file uses for its result text (ad hoc f-string at each of ~15 call
# sites, not currently a shared constant elsewhere — introduced for
# _execute_web_search_query's use, see its docstring, and reused by
# _run_url_fetch's Ollama fallback for the identical reason: distinguishes
# "localist-mcp itself is unreachable" from "the tool call reached
# localist-mcp and failed there", since only the latter should trigger a
# fallback).
_MCP_UNREACHABLE_PREFIX = "ERROR: localist-mcp unreachable"


def _format_fetch_result_text(url: str, data: dict) -> str:
    """
    Render a fetch_url-shaped response dict ({"title", "url", "word_count",
    "cleaned_text", ...}) as the "Title/Source/Words/body" text every
    url_fetch ToolResult uses. Shared verbatim between
    _execute_fetcher_url_fetch (the in-process fetch_url tool) and
    _run_ollama_url_fetch (the Ollama Cloud fallback tier) — the latter's
    MCP-layer response is pre-normalized to this same dict shape (see
    mcp_server/ollama_web_fetch.py) specifically so this one function can
    format both tiers identically, with nothing downstream able to tell
    which tier answered.
    """
    return (
        f"Title: {data.get('title', '')}\n"
        f"Source: {data.get('url', url)}\n"
        f"Words: {data.get('word_count', 0)}\n\n"
        f"{data.get('cleaned_text', '')}"
    )


class MCPToolDispatcher:
    """
    "file_op", "url_fetch", "web_search", "chart", and
    "github_search"/"github_read"/"github_release" are served by the
    localist-mcp MCP server; any other tool name is unrecognized (Planner
    never routes tools_to_call to anything else — see planner.py's
    P3/P3b/P3-github/P3-github-release) and produces an inline error
    ToolResult, same shape the legacy
    ToolDispatcher's "else" branch used to produce.

    "ocr_extract" is also served by localist-mcp but is never planner-routed
    at all — Planner's tools_to_call never names it. It exists solely for
    backend/main.py's POST /chat/files to call directly at upload time
    (dispatcher.dispatch(["ocr_extract"], "", {"ocr_file_path": ...,
    "ocr_mime_type": ...})), reusing this same dispatch() entry point since
    it already resolves real tool arguments from context when present (see
    _run_file_op below) rather than requiring a natural-language instruction
    to parse. See mcp_server/ocr.py and
    docs/architecture/22-local-ocr-service.md.

    Parameters
    ----------
    runtime :
        RuntimeClient. Used by the research loop's pricing-gate evaluation
        and query reformulation (see _run_research_loop) and by chart
        argument extraction (see _run_chart) via a blocking runtime.infer()
        call — the same synchronous-call-from-async-context pattern
        planner.py's _classify_tool_fallback already uses, accepted here for
        the same reason (single-user, non-production app). Prior to the
        research loop's addition, this parameter was accepted but never
        stored — web_search's runtime.infer() hallucination fallback was
        removed in Phase 3 and nothing else here used it.
    project_root :
        Accepted for interface stability; not currently used by any tool
        path here. (file_op's actual sandbox root lives on the MCP server
        — see mcp_server/file_ops.py — and is configured independently via
        LOCALIST_MCP_PROJECT_ROOT.)
    mcp_server_url :
        Override the localist-mcp SSE endpoint. Defaults to
        LOCALIST_MCP_URL env var or http://localhost:8003/sse.
    """

    def __init__(
        self,
        runtime:        Any,
        project_root:   Path | str | None = None,
        mcp_server_url: str | None = None,
    ) -> None:
        self._runtime         = runtime
        self._mcp_server_url  = mcp_server_url or _MCP_SERVER_URL

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def dispatch(
        self,
        tools_to_call: list[str],
        instruction:   str,
        context:       dict[str, Any] | None = None,
    ) -> list[ToolResult]:
        """
        Execute the requested tools and return results. Never raises.

        "file_op", "url_fetch", and "web_search" are served over MCP; any
        other tool name is unrecognized and produces an error ToolResult.
        """
        ctx = context or {}
        return asyncio.run(self._dispatch_async(tools_to_call, instruction, ctx))

    async def _dispatch_async(
        self,
        tools_to_call: list[str],
        instruction:   str,
        ctx:           dict[str, Any],
    ) -> list[ToolResult]:
        """
        Open one MCP ClientSession for this dispatch() call, reuse it for
        every tool invocation made during it, and close it cleanly on the
        way out — happy path or not. Scoped to a single dispatch() call;
        not persisted across separate HTTP requests (see module docstring).
        """
        session:       ClientSession | None = None
        connect_error: Exception | None     = None

        async with AsyncExitStack() as stack:
            try:
                session = await self._open_session(stack)
            except Exception as exc:
                logger.warning(
                    "MCPToolDispatcher: localist-mcp unreachable — %s", exc
                )
                session, connect_error = None, exc

            results: list[ToolResult] = []
            for tool_name in tools_to_call:
                if tool_name == "file_op":
                    results.append(
                        await self._run_file_op(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "ocr_extract":
                    results.append(
                        await self._run_ocr_extract(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "url_fetch":
                    results.append(
                        await self._run_url_fetch(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "web_search":
                    results.extend(
                        await self._run_web_search(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "news_search":
                    results.extend(
                        await self._run_news_search(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "github_search":
                    results.append(
                        await self._run_github_search(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "github_read":
                    results.append(
                        await self._run_github_read(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "github_release":
                    results.extend(
                        await self._run_github_release(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "hacker_news_search":
                    results.append(
                        await self._run_hacker_news_search(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "research":
                    results.extend(
                        await self._run_research_loop(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "chart":
                    chart_result = await self._run_chart(session, connect_error, instruction, ctx)
                    # A failed chart extraction/dispatch degrades to a normal
                    # prose answer rather than surfacing an ERROR: result to
                    # the model — see _run_chart's docstring and
                    # claude/chart-mcp-tool-scoping.md's "Failure handling"
                    # section. Every other tool branch here appends an
                    # ERROR-shaped ToolResult on failure (visible to the
                    # model in Slot 5a); chart deliberately does not.
                    if chart_result is not None:
                        results.append(chart_result)
                    else:
                        logger.warning(
                            "MCPToolDispatcher: chart — extraction/dispatch "
                            "failed; no ToolResult appended (accepted "
                            "residual failure rate — see "
                            "claude/chart-mcp-tool-scoping.md)."
                        )
                else:
                    logger.warning(
                        "MCPToolDispatcher: unknown tool %r — skipping.", tool_name
                    )
                    results.append(ToolResult(
                        tool_name  = tool_name,
                        parameters = "",
                        result     = f"ERROR: unknown tool '{tool_name}'",
                        success    = False,
                    ))

            _succeeded = sum(1 for r in results if r.success)
            _failed    = len(results) - _succeeded
            logger.info(
                "MCPToolDispatcher: dispatch complete — tools=%s succeeded=%d failed=%d",
                tools_to_call, _succeeded, _failed,
            )
            return results

    async def _open_session(self, stack: AsyncExitStack) -> ClientSession:
        """
        Open the SSE transport + ClientSession for this dispatch() call,
        registering both on `stack` so AsyncExitStack tears them down
        together when the dispatch finishes. Split out from
        _dispatch_async as its own method purely as a test seam — tests
        patch this to hand back a fake session without touching the
        network, while still exercising real connect/teardown behavior in
        live verification.
        """
        read, write = await stack.enter_async_context(sse_client(self._mcp_server_url))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    # -----------------------------------------------------------------------
    # file_op — served by localist-mcp
    # -----------------------------------------------------------------------

    async def _run_file_op(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> ToolResult:
        action = (
            context["file_op_action"] if "file_op_action" in context
            else _derive_file_op_action(instruction)
        )
        rel_path = (
            context["file_op_path"] if "file_op_path" in context
            else _derive_file_op_path(instruction)
        )
        content = (
            context["file_op_content"] if "file_op_content" in context
            else _derive_file_op_content(instruction)
        )
        params_str = f"action={action!r} path={rel_path!r}"

        mcp_tool = _FILE_OP_TOOL_MAP.get(action)
        if mcp_tool is None:
            return ToolResult(
                tool_name  = "file_op",
                parameters = params_str,
                result     = f"ERROR: unknown file_op action '{action}'",
                success    = False,
            )

        if not rel_path:
            return ToolResult(
                tool_name  = "file_op",
                parameters = params_str,
                result     = "ERROR: file_op_path not provided in context",
                success    = False,
            )

        if session is None:
            return ToolResult(
                tool_name  = "file_op",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        arguments: dict[str, Any] = {"path": rel_path}
        if mcp_tool in ("write_file", "append_file"):
            arguments["content"] = content
        if mcp_tool == "append_file":
            arguments["turn_id"] = context.get("task_id")

        try:
            text, is_error = await self._call_mcp_tool(session, mcp_tool, arguments)
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for action=%r path=%r: %s",
                action, rel_path, exc,
            )
            return ToolResult(
                tool_name  = "file_op",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            text = _normalize_mcp_error_text(text)

        return ToolResult(
            tool_name  = "file_op",
            parameters = params_str,
            result     = text,
            success    = not is_error,
        )

    # -----------------------------------------------------------------------
    # ocr_extract — served by localist-mcp. Never planner-routed (see class
    # docstring); the caller always supplies both context values, never an
    # instruction to parse.
    # -----------------------------------------------------------------------

    async def _run_ocr_extract(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> ToolResult:
        file_path = context.get("ocr_file_path", "")
        mime_type = context.get("ocr_mime_type", "")
        params_str = f"path={file_path!r} mime_type={mime_type!r}"

        if not file_path or not mime_type:
            return ToolResult(
                tool_name  = "ocr_extract",
                parameters = params_str,
                result     = "ERROR: ocr_file_path and ocr_mime_type must both be provided in context",
                success    = False,
            )

        if session is None:
            return ToolResult(
                tool_name  = "ocr_extract",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        try:
            text, is_error = await self._call_mcp_tool(
                session, "ocr_extract", {"path": file_path, "mime_type": mime_type}
            )
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for ocr_extract path=%r: %s",
                file_path, exc,
            )
            return ToolResult(
                tool_name  = "ocr_extract",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            text = _normalize_mcp_error_text(text)

        return ToolResult(
            tool_name  = "ocr_extract",
            parameters = params_str,
            result     = text,
            success    = not is_error,
        )

    # -----------------------------------------------------------------------
    # url_fetch — served by localist-mcp
    # -----------------------------------------------------------------------

    async def _run_url_fetch(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> ToolResult:
        """
        Resolve the target URL (context["fetch_url"] override, else the
        first http(s):// URL in the instruction), then fetch it via the
        in-process fetch_url tool, falling back to Ollama Cloud
        (_run_ollama_url_fetch) when that fails — see this module's
        docstring ("url_fetch / Ollama Cloud") for the precise fallback
        trigger and why there's no separate empty-result check needed
        here. The fallback is skipped when the failure is localist-mcp
        itself being unreachable, same suppression rule
        _execute_web_search_query uses.
        """
        url: str = context.get("fetch_url", "")
        if not url:
            match = _URL_RE.search(instruction)
            url = match.group(0) if match else ""

        if not url:
            logger.warning(
                "MCPToolDispatcher: url_fetch — no URL found in instruction or context."
            )
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = "",
                result     = "ERROR: no URL found in instruction",
                success    = False,
            )

        fetch_result = await self._execute_fetcher_url_fetch(session, connect_error, url)
        if fetch_result.success or fetch_result.result.startswith(_MCP_UNREACHABLE_PREFIX):
            return fetch_result

        logger.info(
            "MCPToolDispatcher: url_fetch (fetcher) failed for url=%r — "
            "falling back to Ollama Cloud.",
            url,
        )
        fallback = await self._run_ollama_url_fetch(session, connect_error, url)
        return replace(fallback, tool_name="url_fetch:ollama_fallback")

    async def _execute_fetcher_url_fetch(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        url:           str,
    ) -> ToolResult:
        """
        Execute one fetch against the in-process fetch_url MCP tool (ports
        the retired port-8002 Fetcher microservice's /extract path — see
        mcp_server/url_fetch.py). Empty extraction (paywalls/login walls)
        already raises ValueError inside that tool's own _extract(),
        surfacing here as is_error=True indistinguishable from a
        transport/HTTP failure — confirmed by reading the actual
        extraction code, not assumed. So there is no separate "successful
        but empty" branch to check for, unlike
        _execute_brave_web_search_query's result_count==0.
        """
        params_str = f"url={url!r}"

        if session is None:
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        try:
            text, is_error = await self._call_mcp_tool(session, "fetch_url", {"url": url})
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for url_fetch url=%r: %s",
                url, exc,
            )
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = params_str,
                result     = _normalize_mcp_error_text(text),
                success    = False,
            )

        try:
            data = json.loads(text)
        except Exception as exc:
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = params_str,
                result     = f"ERROR: failed to parse fetch_url response — {exc}",
                success    = False,
            )

        result_text = _format_fetch_result_text(url, data)

        logger.info(
            "MCPToolDispatcher: url_fetch complete — url=%r  words=%d  chars=%d",
            url, data.get("word_count", 0), len(result_text),
        )
        return ToolResult(
            tool_name  = "url_fetch",
            parameters = params_str,
            result     = result_text,
            success    = True,
        )

    async def _run_ollama_url_fetch(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        url:           str,
    ) -> ToolResult:
        """
        Call the ollama_web_fetch MCP tool (Ollama Cloud) for one URL —
        the fallback tier when _execute_fetcher_url_fetch fails. Unlike
        web_search's Ollama helper, there is no "primary" mode here: the
        scoping doc's Track B (§5) is fallback-only, no config toggle, no
        Planner visibility (§5.4). ollama_web_fetch's MCP-layer response
        is already normalized to fetch_url's own dict shape (see
        mcp_server/ollama_web_fetch.py), so _format_fetch_result_text
        formats both tiers identically — this method has no opinion on
        being the fallback tier; the caller retags .tool_name afterward
        ("url_fetch:ollama_fallback").
        """
        params_str = f"url={url!r}"

        if session is None:
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        try:
            text, is_error = await self._call_mcp_tool(session, "ollama_web_fetch", {"url": url})
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for ollama_web_fetch url=%r: %s",
                url, exc,
            )
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = params_str,
                result     = _normalize_mcp_error_text(text),
                success    = False,
            )

        try:
            data = json.loads(text)
        except Exception as exc:
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = params_str,
                result     = f"ERROR: failed to parse ollama_web_fetch response — {exc}",
                success    = False,
            )

        result_text = _format_fetch_result_text(url, data)

        logger.info(
            "MCPToolDispatcher: ollama_web_fetch complete — url=%r  words=%d  chars=%d",
            url, data.get("word_count", 0), len(result_text),
        )
        return ToolResult(
            tool_name  = "url_fetch",
            parameters = params_str,
            result     = result_text,
            success    = True,
        )

    # -----------------------------------------------------------------------
    # web_search — served by localist-mcp
    # -----------------------------------------------------------------------

    async def _enrich_top_result(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        search_result: ToolResult,
        tried_urls:    set[str],
    ) -> ToolResult | None:
        """
        Best-effort depth boost for one successful web_search/news_search
        result: follow up with a single fetch_url call on the first URL
        found in its result text, so the model gets real article text
        instead of relying solely on the search provider's own short
        snippet (Brave's `description` in particular is a short meta
        description, not a summary — no truncation-cap tuning can lengthen
        it, only a real fetch can).

        Silent-fail by design — unlike _run_research_loop's candidate
        fetch (whose failure is itself diagnostic information for the
        pricing gate), a blocked/slow/paywalled fetch here just means no
        bonus depth this round: the search snippet already grounds the
        answer, so there's nothing for the model to hedge against. Tries
        up to _ENRICH_MAX_ATTEMPTS candidate URLs, in result order, before
        giving up entirely — a single bot-blocked/timed-out URL (news
        portals routinely 403 non-browser fetches) shouldn't cost the
        whole enrichment when other, perfectly fetchable URLs are sitting
        right below it in the same result. Returns None (nothing to
        append) when there's no URL left to try, the search itself didn't
        succeed, or every candidate's fetch failed.
        """
        if not search_result.success:
            return None

        for _ in range(_ENRICH_MAX_ATTEMPTS):
            url = self._extract_first_url(search_result.result, tried_urls)
            if not url:
                return None
            tried_urls.add(url)

            fetch_result = await self._run_url_fetch(
                session, connect_error, "", {"fetch_url": url}
            )
            if not fetch_result.success:
                logger.info(
                    "MCPToolDispatcher: enrichment fetch failed for url=%r — "
                    "trying next candidate, if any (%s).",
                    url, fetch_result.result,
                )
                continue

            if len(fetch_result.result) > _ENRICH_EXCERPT_CHARS:
                excerpt = fetch_result.result[:_ENRICH_EXCERPT_CHARS].rsplit(" ", 1)[0] + "…"
                fetch_result = replace(fetch_result, result=excerpt)

            return fetch_result

        return None

    async def _run_web_search(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> list[ToolResult]:
        """
        Execute web_search for up to _MAX_WEB_QUERIES queries.

        Query resolution order (ported verbatim from
        ToolDispatcher._run_web_search):
          1. context["web_search_queries"] — explicit list (max 3 used)
          2. Derive a single query from the instruction by stripping known
             filler phrases and taking the first 120 characters.

        Each query's search result that succeeds and contains a URL gets
        one follow-up fetch_url enrichment (see _enrich_top_result).
        tried_urls is shared across every query in this call so two
        similar queries surfacing the same top hit don't fetch it twice.
        A workflow_id is stamped across every result in this call, but
        only when at least one enrichment actually happened — a plain,
        un-enriched search stays a single unwrapped ToolResult exactly as
        before, so the Episode Browsing UI's step-chain view only appears
        when there's a real multi-step chain to show.
        """
        raw_queries: list[str] = context.get("web_search_queries") or []

        if not raw_queries:
            derived = instruction.strip()
            for filler in _WEB_SEARCH_FILLER_PREFIXES:
                if derived.lower().startswith(filler):
                    derived = derived[len(filler):]
                    break
            raw_queries = [derived[:120]]

        queries = raw_queries[:_MAX_WEB_QUERIES]
        tried_urls: set[str] = set()
        results: list[ToolResult] = []
        for query in queries:
            search_result = await self._execute_web_search_query(session, connect_error, query)
            results.append(search_result)
            enrichment = await self._enrich_top_result(session, connect_error, search_result, tried_urls)
            if enrichment is not None:
                results.append(enrichment)

        if len(results) > len(queries):
            workflow_id = str(uuid.uuid4())
            results = [replace(r, workflow_id=workflow_id) for r in results]

        return results

    async def _execute_web_search_query(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        query:         str,
    ) -> ToolResult:
        """
        Execute one web_search query via the configured provider.

        WEB_SEARCH_PROVIDER (default "brave") is read fresh on every call,
        not cached at import time — same convention already used for
        SEARCH_PROVIDER (controller_agent.py, mcp_server/web_search.py):
        this is live-swappable config, so it must be resolved at request
        time rather than captured once and held onto (CLAUDE.md's runtime
        backend rule applies to any config that can change without a
        process restart, not just LOCALIST_RUNTIME_BACKEND).

        provider == "ollama": Ollama Cloud answers directly as primary via
        _run_ollama_web_search() — Brave is never called, not even as a
        fallback, so this works correctly even with BRAVE_API_KEY entirely
        absent from the environment. Result is retagged tool_name=
        "web_search:ollama_primary".

        provider == "brave" (default): Brave answers first via
        _execute_brave_web_search_query(). If that result is empty or
        failed (success=False — see that method's docstring for the exact
        "empty or failed" definition, using Brave/LangSearch's actual
        result_count field, not an assumed shape), falls through to the
        same _run_ollama_web_search() helper the primary-mode branch above
        uses — one shared helper, not two separate Ollama-calling code
        paths — and retags the fallback's result tool_name="web_search:
        ollama_fallback". Mirrors _run_news_search's NewsAPI→Brave
        tiered-fallback shape, just one layer further down inside
        web_search itself (ollama-web-search-mcp-tool-scoping.md §0.2). On
        a genuine Brave success, the result is returned completely
        unchanged — tool_name stays "web_search", no new tagging for the
        unchanged common-case path.

        Exception: when Brave's failure is localist-mcp itself being
        unreachable (session is None, or _call_mcp_tool raised — see
        _MCP_UNREACHABLE_PREFIX) rather than a Brave-specific problem, the
        Ollama fallback is skipped — Ollama is served by the same
        localist-mcp process, so retrying through an already-known-dead
        session would just fail identically. This preserves the existing
        "one call, fail fast" contract for a total outage (see
        TestSessionReuse.test_connection_down_degrades_every_tool_call_not_
        just_first / TestWebSearch.test_connection_failure_returns_graceful_
        error / TestResearchLoop.test_connectivity_failure_stops_
        immediately_without_synthetic_result) — the fallback exists to
        route around a Brave-side problem, not to double up on retries
        during an infrastructure outage.

        Because this is the single point _run_web_search(), _run_news_search
        (tier-2 fallback), and _run_research_loop (each iteration) all call
        for one query, all three transparently gain Ollama coverage from
        this one change — none of their own bodies need to change.
        """
        provider = os.environ.get("WEB_SEARCH_PROVIDER", "brave").lower()

        if provider == "ollama":
            primary = await self._run_ollama_web_search(session, connect_error, query)
            return replace(primary, tool_name="web_search:ollama_primary")

        brave_result = await self._execute_brave_web_search_query(session, connect_error, query)
        if brave_result.success or brave_result.result.startswith(_MCP_UNREACHABLE_PREFIX):
            return brave_result

        logger.info(
            "MCPToolDispatcher: web_search (Brave) empty/failed for query=%r — "
            "falling back to Ollama Cloud.",
            query,
        )
        fallback = await self._run_ollama_web_search(session, connect_error, query)
        return replace(fallback, tool_name="web_search:ollama_fallback")

    async def _execute_brave_web_search_query(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        query:         str,
    ) -> ToolResult:
        """
        Execute one query against the generic "web_search" MCP tool — the
        "brave" branch of _execute_web_search_query()'s provider dispatch.

        Note this calls the MCP tool literally named "web_search"
        regardless of whether that tool is itself backed by Brave or
        LangSearch — SEARCH_PROVIDER (mcp_server/web_search.py) is a
        separate, MCP-server-local setting this dispatcher doesn't
        inspect. WEB_SEARCH_PROVIDER's "brave" branch name just means
        "try the existing web_search tool before reaching for Ollama" —
        it happens to be Brave in this deployment's current .env
        (SEARCH_PROVIDER=brave), same as the scoping doc's framing.

        A successful call with zero results — result_count == 0, both
        mcp_server/web_search.py's _web_search_brave and
        _web_search_langsearch return
        {"result_text": "No results found.", "result_count": 0} on an
        empty response, grepped directly rather than assumed — is treated
        as a miss: success=False, result_text preserved as-is. This is
        the new "successful-but-zero-results" case
        _execute_web_search_query()'s fallback needs to catch, distinct
        from the transport/MCP/parse failures below (unchanged, already
        existing) which also produce success=False but with an
        "ERROR: ..." result_text instead of "No results found.".
        """
        params_str = f"query={query!r}"

        if session is None:
            return ToolResult(
                tool_name  = "web_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        try:
            text, is_error = await self._call_mcp_tool(session, "web_search", {"query": query})
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for web_search query=%r: %s",
                query, exc,
            )
            return ToolResult(
                tool_name  = "web_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            return ToolResult(
                tool_name  = "web_search",
                parameters = params_str,
                result     = _normalize_mcp_error_text(text),
                success    = False,
            )

        try:
            data = json.loads(text)
        except Exception as exc:
            return ToolResult(
                tool_name  = "web_search",
                parameters = params_str,
                result     = f"ERROR: failed to parse web_search response — {exc}",
                success    = False,
            )

        return ToolResult(
            tool_name  = "web_search",
            parameters = params_str,
            result     = data.get("result_text", ""),
            success    = data.get("result_count", 0) > 0,
        )

    async def _run_ollama_web_search(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        query:         str,
    ) -> ToolResult:
        """
        Call the ollama_web_search MCP tool (Ollama Cloud) for one query.

        Used identically by _execute_web_search_query() whether Ollama is
        running as the configured primary (WEB_SEARCH_PROVIDER=ollama) or
        as the fallback tier when Brave is empty/failed (default config)
        — this method has no opinion on which case it's being called for;
        the caller retags .tool_name afterward ("web_search:ollama_primary"
        / "web_search:ollama_fallback") to distinguish the two in logs and
        transcripts. Same shape as _execute_brave_web_search_query, just
        against the "ollama_web_search" MCP tool instead of "web_search".
        """
        params_str = f"query={query!r}"

        if session is None:
            return ToolResult(
                tool_name  = "web_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        try:
            text, is_error = await self._call_mcp_tool(session, "ollama_web_search", {"query": query})
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for ollama_web_search query=%r: %s",
                query, exc,
            )
            return ToolResult(
                tool_name  = "web_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            return ToolResult(
                tool_name  = "web_search",
                parameters = params_str,
                result     = _normalize_mcp_error_text(text),
                success    = False,
            )

        try:
            data = json.loads(text)
        except Exception as exc:
            return ToolResult(
                tool_name  = "web_search",
                parameters = params_str,
                result     = f"ERROR: failed to parse ollama_web_search response — {exc}",
                success    = False,
            )

        return ToolResult(
            tool_name  = "web_search",
            parameters = params_str,
            result     = data.get("result_text", ""),
            success    = data.get("result_count", 0) > 0,
        )

    # -----------------------------------------------------------------------
    # news_search — NewsAPI first, falling back to web_search (Brave) on a miss
    # -----------------------------------------------------------------------

    async def _run_news_search(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> list[ToolResult]:
        """
        Execute news_search: NewsAPI (tier 1) first, falling through to the
        existing Brave-backed web_search tool (tier 2) on a miss — see
        news-query-routing plan §4.1/§4.2. Single query only (unlike
        web_search's up-to-_MAX_WEB_QUERIES loop) — reuses
        _derive_initial_query so both tiers see the same derived text; a
        miss on "latest news on X" still asks the fallback tier about "X",
        not a generic re-derivation.

        Always returns the tier-1 ToolResult. Only calls tier 2 (and
        appends its result) when tier 1 didn't succeed — mirrors
        _run_research_loop's "return every ToolResult produced, not just
        the winning one" convention so Slot 5 and controller_agent.py's
        Step 3b corpus fallback see the full picture.

        Whichever tier actually succeeds gets one follow-up fetch_url
        enrichment (see _enrich_top_result) — never both, since a tier-1
        success means tier 2 never runs. This also transparently benefits
        the §14.10 URL-pinning case (Live Feed "Ask about this"): the
        pinned article's own URL is already in news_result.result, so
        enrichment fetches that same article's full text on top of
        NewsAPI's own short `content` field.
        """
        query = self._derive_initial_query(instruction, context)
        article_url = context.get("news_article_url") or None
        news_result = await self._execute_news_search_query(session, connect_error, query, article_url)
        tried_urls: set[str] = set()
        results: list[ToolResult] = [news_result]

        if news_result.success:
            enrichment = await self._enrich_top_result(session, connect_error, news_result, tried_urls)
            if enrichment is not None:
                results.append(enrichment)
            if len(results) > 1:
                workflow_id = str(uuid.uuid4())
                results = [replace(r, workflow_id=workflow_id) for r in results]
            return results

        logger.info(
            "MCPToolDispatcher: news_search miss/failure for query=%r — "
            "falling back to web_search (Brave).",
            query,
        )
        brave_result = await self._execute_web_search_query(session, connect_error, query)
        # Retag for provenance (§4.3) — Step 3b's corpus-fallback check in
        # controller_agent.py matches on tool_name.startswith("news_search"),
        # so this stays visible to it even after the rename.
        brave_result = replace(brave_result, tool_name="news_search:brave_fallback")
        results.append(brave_result)

        enrichment = await self._enrich_top_result(session, connect_error, brave_result, tried_urls)
        if enrichment is not None:
            results.append(enrichment)

        if len(results) > 2:
            workflow_id = str(uuid.uuid4())
            results = [replace(r, workflow_id=workflow_id) for r in results]

        return results

    async def _execute_news_search_query(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        query:         str,
        url:           str | None = None,
    ) -> ToolResult:
        params_str = f"query={query!r}" if not url else f"query={query!r}, url={url!r}"

        if session is None:
            return ToolResult(
                tool_name  = "news_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        tool_args: dict[str, Any] = {"query": query}
        if url:
            tool_args["url"] = url

        try:
            text, is_error = await self._call_mcp_tool(session, "news_search", tool_args)
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for news_search query=%r: %s",
                query, exc,
            )
            return ToolResult(
                tool_name  = "news_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            return ToolResult(
                tool_name  = "news_search",
                parameters = params_str,
                result     = _normalize_mcp_error_text(text),
                success    = False,
            )

        try:
            data = json.loads(text)
        except Exception as exc:
            return ToolResult(
                tool_name  = "news_search",
                parameters = params_str,
                result     = f"ERROR: failed to parse news_search response — {exc}",
                success    = False,
            )

        if data.get("is_miss", False):
            return ToolResult(
                tool_name  = "news_search",
                parameters = params_str,
                result     = "",
                success    = False,
            )

        return ToolResult(
            tool_name  = "news_search",
            parameters = params_str,
            result     = data.get("result_text", ""),
            success    = True,
        )

    # -----------------------------------------------------------------------
    # github_search / github_read / github_release — served by localist-mcp
    # -----------------------------------------------------------------------

    async def _run_github_search(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> ToolResult:
        """
        Execute one github_search query — public repo/code search via the
        GitHub Search API. Single query only, no _MAX_WEB_QUERIES-style
        loop and no NewsAPI-style fallback tier (GitHub's Search API has no
        "miss vs. error" distinction worth a second provider) — reuses the
        same query resolution web_search/news_search already use via
        _derive_initial_query, so a "search github for X" instruction and a
        plain "look up X" instruction derive the same query text.
        """
        query = self._derive_initial_query(instruction, context)
        kind  = context.get("github_search_kind") or "repositories"
        return await self._execute_github_search_query(session, connect_error, query, kind)

    async def _execute_github_search_query(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        query:         str,
        kind:          str = "repositories",
    ) -> ToolResult:
        """
        Run one github_search call and wrap the result as a ToolResult.
        Split out from _run_github_search so _run_github_release() can
        reuse this same single-call path to resolve a bare repo name (e.g.
        "oMLX") to a concrete owner/repo before fetching a release — same
        "extracted, reusable single-query helper" shape
        _execute_web_search_query()/_execute_news_search_query() already
        established.
        """
        params_str = f"query={query!r} kind={kind!r}"

        if session is None:
            return ToolResult(
                tool_name  = "github_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        try:
            text, is_error = await self._call_mcp_tool(
                session, "github_search", {"query": query, "kind": kind}
            )
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for github_search query=%r: %s",
                query, exc,
            )
            return ToolResult(
                tool_name  = "github_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            return ToolResult(
                tool_name  = "github_search",
                parameters = params_str,
                result     = _normalize_mcp_error_text(text),
                success    = False,
            )

        try:
            data = json.loads(text)
        except Exception as exc:
            return ToolResult(
                tool_name  = "github_search",
                parameters = params_str,
                result     = f"ERROR: failed to parse github_search response — {exc}",
                success    = False,
            )

        if data.get("is_miss", False):
            return ToolResult(
                tool_name  = "github_search",
                parameters = params_str,
                result     = "",
                success    = False,
            )

        return ToolResult(
            tool_name  = "github_search",
            parameters = params_str,
            result     = data.get("result_text", ""),
            success    = True,
        )

    async def _run_github_read(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> ToolResult:
        """
        Read a public repo's README/file/directory listing. Unlike
        github_search (reached via Planner's keyword gate, P3-github),
        github_read is only ever reached when a caller supplies
        context["github_repo"] as an "owner/repo" string directly — same
        caller-supplied-pin convention as news_search's
        context["news_article_url"] (see _run_news_search) — Planner's
        keyword gate can detect that a GitHub-shaped request was made, not
        which repo. context["github_path"] is optional (omitted reads the
        repo's README).
        """
        repo_spec = context.get("github_repo") or ""
        path      = context.get("github_path") or None
        params_str = f"repo={repo_spec!r} path={path!r}"

        if "/" not in repo_spec:
            return ToolResult(
                tool_name  = "github_read",
                parameters = params_str,
                result     = 'ERROR: github_repo not provided in context (expected "owner/repo")',
                success    = False,
            )

        owner, _, repo = repo_spec.partition("/")

        if session is None:
            return ToolResult(
                tool_name  = "github_read",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        arguments: dict[str, Any] = {"owner": owner, "repo": repo}
        if path:
            arguments["path"] = path

        try:
            text, is_error = await self._call_mcp_tool(session, "github_read", arguments)
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for github_read repo=%r path=%r: %s",
                repo_spec, path, exc,
            )
            return ToolResult(
                tool_name  = "github_read",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            return ToolResult(
                tool_name  = "github_read",
                parameters = params_str,
                result     = _normalize_mcp_error_text(text),
                success    = False,
            )

        try:
            data = json.loads(text)
        except Exception as exc:
            return ToolResult(
                tool_name  = "github_read",
                parameters = params_str,
                result     = f"ERROR: failed to parse github_read response — {exc}",
                success    = False,
            )

        return ToolResult(
            tool_name  = "github_read",
            parameters = params_str,
            result     = data.get("content", ""),
            success    = True,
        )

    def _derive_github_release_tag(self, instruction: str, context: dict[str, Any]) -> str | None:
        """context["github_tag"] wins if supplied; otherwise the first
        version-shaped token in the instruction (see _GITHUB_VERSION_RE).
        None means "latest release" — github_release()'s own default."""
        explicit = context.get("github_tag")
        if explicit:
            return explicit
        match = _GITHUB_VERSION_RE.search(instruction)
        return match.group(0) if match else None

    def _derive_github_release_query(self, instruction: str) -> str:
        """
        Derive a repo-name search query from a release-shaped instruction
        with no explicit context["github_repo"] pin. Two strategies, see
        _GITHUB_RELEASE_SUBJECT_RE's module comment for both:

          1. "release notes/changelog FOR/OF X" (name trails the marker)
             -> subject_match captures "X" directly, used as-is.
          2. "fetch the X 0.5.3 release notes ..." (name precedes the
             marker) -> truncate before the earliest marker/version match,
             strip filler.
        """
        subject_match = _GITHUB_RELEASE_SUBJECT_RE.search(instruction.strip())
        if subject_match:
            return subject_match.group(1).strip()[:120]

        marker_match  = _GITHUB_RELEASE_MARKER_RE.search(instruction)
        version_match = _GITHUB_VERSION_RE.search(instruction)
        cut_points = [m.start() for m in (marker_match, version_match) if m]
        candidate = instruction[:min(cut_points)] if cut_points else instruction

        # Deliberately lstrip() only, not strip() — a filler prefix like
        # "fetch the " carries its own trailing space, and pre-stripping
        # the candidate's trailing space made that exact (longer, more
        # complete) prefix fail to match, silently falling through to a
        # shorter one ("fetch ") that left a dangling "the" behind as the
        # whole "query" (caught live, 2026-07-29: "fetch the release notes
        # for cli/cli" derived "the" instead of falling back to the full
        # instruction). Trailing whitespace is trimmed at the very end
        # instead, after all stripping is done.
        stripped = candidate.lstrip()
        for filler in _GITHUB_RELEASE_FILLER_PREFIXES:
            if stripped.lower().startswith(filler):
                stripped = stripped[len(filler):]
                break
        # \b\s* (not \s+) so a leftover bare article with nothing after it
        # ("the" alone) is still stripped, not just one followed by more text.
        stripped = re.sub(r"^\s*(the|a|an)\b\s*", "", stripped, flags=re.IGNORECASE)
        stripped = stripped.strip()
        return stripped[:120] or instruction.strip()[:120]

    def _extract_owner_repo_from_search_result(self, result_text: str) -> tuple[str, str] | None:
        """
        Pulls the top hit's owner/repo out of a github_search ToolResult's
        formatted text via the first *bracketed* URL specifically (see
        _GITHUB_SEARCH_BRACKETED_URL_RE — deliberately not _URL_RE, which
        would also match an unrelated URL embedded in a hit's own
        description/summary text if one appears earlier in the string).
        Returns None if no bracketed URL is found or it doesn't have at
        least two path segments.
        """
        match = _GITHUB_SEARCH_BRACKETED_URL_RE.search(result_text)
        if not match:
            return None
        parts = urlparse(match.group(1)).path.strip("/").split("/")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            return None
        return parts[0], parts[1]

    async def _run_github_release(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> list[ToolResult]:
        """
        Fetch one release's notes — the latest release, or a specific
        tagged one when the instruction (or context["github_tag"]) names
        a version.

        Unlike github_search/github_read, this is a two-step chain when
        no explicit context["github_repo"] pin is supplied: a
        github_search call first resolves a bare project name (e.g.
        "oMLX") to a concrete owner/repo from its top hit, then
        github_release fetches that repo's release. Same "search, then
        fetch a specific record off the top hit" shape
        _enrich_top_result() already established for web_search/
        news_search — both ToolResults are returned (not just the
        winner), stamped with a shared workflow_id, mirroring that
        convention. When context["github_repo"] is already known (same
        caller-supplied-pin convention as news_search's
        context["news_article_url"] — no chat UI sets this yet, same as
        github_read), the search step is skipped entirely and only the
        release ToolResult is returned, no workflow_id.
        """
        tag = self._derive_github_release_tag(instruction, context)
        repo_spec = context.get("github_repo") or ""
        search_result: ToolResult | None = None

        if "/" in repo_spec:
            owner, _, repo = repo_spec.partition("/")
        else:
            query = self._derive_github_release_query(instruction)
            search_result = await self._execute_github_search_query(
                session, connect_error, query, "repositories"
            )
            if not search_result.success:
                return [ToolResult(
                    tool_name  = "github_release",
                    parameters = f"query={query!r} tag={tag!r}",
                    result     = search_result.result or f"ERROR: could not resolve a repo for {query!r}",
                    success    = False,
                )]

            resolved = self._extract_owner_repo_from_search_result(search_result.result)
            if resolved is None:
                return [search_result, ToolResult(
                    tool_name  = "github_release",
                    parameters = f"query={query!r} tag={tag!r}",
                    result     = f"ERROR: could not resolve owner/repo from github_search result for {query!r}",
                    success    = False,
                )]
            owner, repo = resolved

        params_str = f"owner={owner!r} repo={repo!r} tag={tag!r}"

        def _finish(release_result: ToolResult) -> list[ToolResult]:
            if search_result is None:
                return [release_result]
            workflow_id = str(uuid.uuid4())
            return [
                replace(search_result, workflow_id=workflow_id),
                replace(release_result, workflow_id=workflow_id),
            ]

        if session is None:
            return _finish(ToolResult(
                tool_name  = "github_release",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            ))

        arguments: dict[str, Any] = {"owner": owner, "repo": repo}
        if tag:
            arguments["tag"] = tag

        try:
            text, is_error = await self._call_mcp_tool(session, "github_release", arguments)
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for github_release "
                "owner=%r repo=%r tag=%r: %s",
                owner, repo, tag, exc,
            )
            return _finish(ToolResult(
                tool_name  = "github_release",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            ))

        if is_error:
            return _finish(ToolResult(
                tool_name  = "github_release",
                parameters = params_str,
                result     = _normalize_mcp_error_text(text),
                success    = False,
            ))

        try:
            data = json.loads(text)
        except Exception as exc:
            return _finish(ToolResult(
                tool_name  = "github_release",
                parameters = params_str,
                result     = f"ERROR: failed to parse github_release response — {exc}",
                success    = False,
            ))

        header = f"{owner}/{repo} {data.get('name') or data.get('tag_name', '')} — {data.get('published_at', '')}"
        result_text = f"{header}\n{data.get('body', '')}\n[{data.get('html_url', '')}]"

        return _finish(ToolResult(
            tool_name  = "github_release",
            parameters = params_str,
            result     = result_text,
            success    = True,
        ))

    # -----------------------------------------------------------------------
    # hacker_news_search — served by localist-mcp
    # -----------------------------------------------------------------------

    async def _run_hacker_news_search(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> ToolResult:
        """
        Execute one hacker_news_search query — Hacker News story/discussion
        search via Algolia's HN Search API. Single query only, same
        "no NewsAPI-style fallback tier" shape as github_search (Algolia's
        HN Search API has no "miss vs. error" distinction worth a second
        provider) — reuses the same query resolution web_search/
        news_search/github_search already use via _derive_initial_query.

        context["hn_story_url"], when supplied (the Live Feed panel's
        "Ask about this" button on a Hacker News story), pins the result to
        that one already-known story — same caller-supplied-pin convention
        as news_search's context["news_article_url"].
        """
        query = self._derive_initial_query(instruction, context)
        story_url = context.get("hn_story_url") or None
        params_str = f"query={query!r}" if not story_url else f"query={query!r}, url={story_url!r}"

        if session is None:
            return ToolResult(
                tool_name  = "hacker_news_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        tool_args: dict[str, Any] = {"query": query}
        if story_url:
            tool_args["url"] = story_url

        try:
            text, is_error = await self._call_mcp_tool(
                session, "hacker_news_search", tool_args
            )
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for hacker_news_search query=%r: %s",
                query, exc,
            )
            return ToolResult(
                tool_name  = "hacker_news_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            return ToolResult(
                tool_name  = "hacker_news_search",
                parameters = params_str,
                result     = _normalize_mcp_error_text(text),
                success    = False,
            )

        try:
            data = json.loads(text)
        except Exception as exc:
            return ToolResult(
                tool_name  = "hacker_news_search",
                parameters = params_str,
                result     = f"ERROR: failed to parse hacker_news_search response — {exc}",
                success    = False,
            )

        if data.get("is_miss", False):
            return ToolResult(
                tool_name  = "hacker_news_search",
                parameters = params_str,
                result     = "",
                success    = False,
            )

        return ToolResult(
            tool_name  = "hacker_news_search",
            parameters = params_str,
            result     = data.get("result_text", ""),
            success    = True,
        )

    # -----------------------------------------------------------------------
    # research — bounded search / evaluate / reformulate / fetch loop
    # -----------------------------------------------------------------------

    async def _run_research_loop(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> list[ToolResult]:
        """
        Loop up to _MAX_RESEARCH_ITERATIONS times: web_search -> evaluate
        (cheap yes/no classifier call, same pattern as
        controller_agent._execute_plan's P5 episodic-relevance check) ->
        if the result text already contains concrete pricing, url_fetch the
        top candidate page and re-run the gate on the full text; if not,
        reformulate the query (one more bounded infer call) and retry.

        Returns every ToolResult produced along the way (all search/fetch
        attempts, not just the winning one) so controller_agent's existing
        logging/fallback logic (Step 3b: corpus fallback when every
        web_search result failed) keeps working unmodified.

        Every ToolResult returned carries the same freshly-generated
        `workflow_id` (see ToolResult.workflow_id) — this method is already
        a single bounded call representing one research "workflow", so the
        id is generated once here and stamped on every constituent result
        rather than needing a correlation key threaded in from outside.
        Read by controller_agent.py to build metadata["workflow_steps"] for
        the Episode Browsing UI's step-chain view
        (episode-browsing-ui-plan.md, Phase 2).

        Two distinct "didn't work" outcomes are handled differently:
          - A search/fetch call itself fails (provider/connectivity error):
            the failing ToolResult already has tool_name="web_search"/
            "url_fetch" and success=False, so it's indistinguishable from a
            plain web_search failure — Step 3b's existing
            `r.tool_name == "web_search" and not r.success` check already
            catches it with no changes needed there.
          - Every iteration's search/fetch call *succeeds* but the pricing
            gate never passes (loop exhausts, or reformulation degenerates
            to a repeat): every individual ToolResult in that case has
            success=True (the searches worked; they just didn't find
            pricing), so nothing in the returned list would trip Step 3b's
            web_search check. A synthetic trailing ToolResult
            (tool_name="research", result starting with "ERROR:",
            success=False) is appended in that case only — same
            "ERROR: ..." shape every other failure path in this file uses,
            so it flows into controller_agent's tool_failures prompt slot
            (letting the model honestly say it couldn't find pricing rather
            than guessing) and, via the added `or r.tool_name == "research"`
            in Step 3b, also triggers the corpus fallback.
        """
        results: list[ToolResult] = []
        tried_queries: list[str] = []
        tried_urls:    set[str]  = set()
        connectivity_failed = False
        workflow_id = str(uuid.uuid4())

        query = self._derive_initial_query(instruction, context)

        for iteration in range(_MAX_RESEARCH_ITERATIONS):
            tried_queries.append(query)
            search_result = await self._execute_web_search_query(
                session, connect_error, query
            )
            search_result.workflow_id = workflow_id
            results.append(search_result)

            if not search_result.success:
                # Provider/connectivity failure, not a "no pricing found"
                # outcome — stop the loop, let controller_agent's Step 3b
                # corpus fallback take over exactly as it does for a plain
                # web_search failure today.
                connectivity_failed = True
                break

            gate_pass = await self._evaluate_pricing_gate(instruction, search_result.result)

            candidate_url = self._extract_first_url(search_result.result, tried_urls)

            if not gate_pass and candidate_url:
                # Search snippet alone was inconclusive but pointed at a
                # page — pull the full page before giving up on this query.
                tried_urls.add(candidate_url)
                fetch_result = await self._run_url_fetch(
                    session, connect_error, instruction,
                    {**context, "fetch_url": candidate_url},
                )
                fetch_result.workflow_id = workflow_id
                results.append(fetch_result)
                if fetch_result.success:
                    gate_pass = await self._evaluate_pricing_gate(instruction, fetch_result.result)

            if gate_pass:
                logger.info(
                    "MCPToolDispatcher: research loop — pricing found after "
                    "%d iteration(s), queries=%s.",
                    iteration + 1, tried_queries,
                )
                return results

            if iteration == _MAX_RESEARCH_ITERATIONS - 1:
                break

            query = await self._reformulate_query(instruction, tried_queries)
            if query in tried_queries:
                # Reformulation degenerated to a repeat — stop rather than
                # spend another round-trip on a query we know fails.
                break

        logger.info(
            "MCPToolDispatcher: research loop — exhausted %d iteration(s) "
            "without concrete pricing, queries=%s.",
            len(tried_queries), tried_queries,
        )
        if not connectivity_failed:
            results.append(ToolResult(
                tool_name   = "research",
                parameters  = f"queries={tried_queries!r}",
                result      = (
                    f"ERROR: research loop exhausted {len(tried_queries)} "
                    f"iteration(s) without finding concrete pricing "
                    f"information (queries tried: {tried_queries})."
                ),
                success     = False,
                workflow_id = workflow_id,
            ))
        return results

    def _derive_initial_query(self, instruction: str, context: dict[str, Any]) -> str:
        # Reuse the exact same resolution order _run_web_search already
        # uses (explicit context["web_search_queries"][0], else derived
        # from the instruction) so "research" and "web_search" behave
        # identically on turn one and only diverge once evaluation kicks in.
        raw_queries: list[str] = context.get("web_search_queries") or []
        if raw_queries:
            return raw_queries[0]
        derived = instruction.strip()
        for filler in _WEB_SEARCH_FILLER_PREFIXES:
            if derived.lower().startswith(filler):
                derived = derived[len(filler):]
                break
        return derived[:120]

    async def _evaluate_pricing_gate(self, instruction: str, text: str) -> bool:
        """Single bounded yes/no inference call. Never raises — a failed
        gate check is treated as "no", same fail-open-to-continue posture
        as every other try/except in this file.

        `instruction` is the original question, passed alongside `text` so
        the classifier can judge relevance (does this text specifically
        answer THIS question) rather than merely detecting that some
        pricing/spec-shaped content is present somewhere in `text` — see
        _RESEARCH_GATE_SYSTEM_PROMPT and
        diagnostics/reports/research_loop_qa_assessment_2026-07-20.md for
        the false-positive pattern this fixes (e.g. gate-passing on a
        different product's price, or on content that never actually
        states the requested number)."""
        try:
            raw = self._runtime.infer(
                system      = _RESEARCH_GATE_SYSTEM_PROMPT,
                prompt      = (
                    f"Original question:\n\n{instruction}\n\n"
                    f"Text:\n\n{text[:3000]}\n\n"
                    f"Does the text directly answer the original question with a "
                    f"specific number (yes/no):"
                ),
                max_tokens  = 10,
                temperature = 0.1,
                timeout     = _RESEARCH_CLASSIFIER_TIMEOUT,
            )
            return raw.strip().lower().startswith("yes")
        except Exception as exc:
            logger.debug("MCPToolDispatcher: research gate check failed (%s).", exc)
            return False

    async def _reformulate_query(self, instruction: str, tried: list[str]) -> str:
        try:
            raw = self._runtime.infer(
                system      = _RESEARCH_REFORMULATE_SYSTEM_PROMPT,
                prompt      = (
                    f"Original request: {instruction}\n"
                    f"Queries already tried: {tried}\n\nNew query:"
                ),
                max_tokens  = 40,
                temperature = 0.3,
                timeout     = _RESEARCH_CLASSIFIER_TIMEOUT,
            )
            return raw.strip().strip('"')[:120]
        except Exception as exc:
            logger.debug("MCPToolDispatcher: query reformulation failed (%s).", exc)
            return tried[-1]  # fall through to the repeat-guard, which stops the loop

    @staticmethod
    def _extract_first_url(text: str, exclude: set[str]) -> str | None:
        for match in _URL_RE.finditer(text):
            # _URL_RE already excludes ]/) from the match itself (2026-07-16
            # fix), but a URL pulled out of running text can still end in
            # trailing sentence punctuation a URL is very unlikely to
            # legitimately end with (e.g. "...pricing." at a sentence
            # boundary) — stripped here as a second, cheap layer of defense
            # against a differently-formatted future source hitting the
            # same class of bug the bracket-wrapping case did.
            url = match.group(0).rstrip(".,;:")
            if url not in exclude:
                return url
        return None

    # -----------------------------------------------------------------------
    # chart — served by localist-mcp (generate_chart)
    # -----------------------------------------------------------------------

    async def _run_chart(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> ToolResult | None:
        """
        Extract generate_chart arguments from `instruction` via a bounded
        few-shot inference call, then dispatch to the generate_chart MCP
        tool. Promotes diag_shadow_chart_toolcall_v4_full.py's measured
        pipeline to production, unchanged — see the module docstring's
        "chart" paragraph and claude/chart-mcp-tool-scoping.md for the
        reliability numbers this is based on (66.7% MATCH on
        chart-expected instructions, 12.1% residual failure accepted by
        design).

        Returns None — not an ERROR-shaped ToolResult — on any failure
        (post-retry still malformed, schema-invalid, the model legitimately
        declining via {"tool_call": null}, an unreachable localist-mcp
        server, or the generate_chart tool call itself failing). See
        _dispatch_async's chart branch: a failed chart never reaches the
        model as a visible tool error, it just means the turn ends up with
        no chart — the accepted residual-failure behavior.
        """
        arguments = await self._extract_chart_arguments(instruction)
        if arguments is None:
            return None

        params_str = f"chart_type={arguments.get('chart_type')!r}"

        if session is None:
            logger.warning(
                "MCPToolDispatcher: chart — localist-mcp unreachable (%s).",
                connect_error,
            )
            return None

        try:
            text, is_error = await self._call_mcp_tool(session, "generate_chart", arguments)
        except Exception as exc:
            logger.warning("MCPToolDispatcher: chart — localist-mcp call failed: %s", exc)
            return None

        if is_error:
            logger.warning(
                "MCPToolDispatcher: chart — generate_chart tool failed: %s",
                _normalize_mcp_error_text(text),
            )
            return None

        try:
            data = json.loads(text)
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: chart — failed to parse generate_chart response: %s", exc
            )
            return None

        logger.info(
            "MCPToolDispatcher: chart complete — %s", params_str,
        )
        return ToolResult(
            tool_name  = "chart",
            parameters = params_str,
            # Only "summary" ever reaches the model (Slot 5's 500-token
            # ceiling) — png_path/chart_config ride in .artifact instead,
            # read directly by controller_agent.py, never rendered into
            # prompt-facing text. See prompt_builder.ToolResult.artifact.
            result     = data.get("summary", ""),
            success    = True,
            artifact   = {
                "png_path":     data.get("png_path"),
                "chart_config": data.get("chart_config"),
            },
        )

    async def _extract_chart_arguments(self, instruction: str) -> dict[str, Any] | None:
        """
        Run the infer -> repair -> validate pipeline at temperature=0.0; on
        a malformed envelope, retry once at _CHART_RETRY_TEMPERATURE. The
        retry's outcome is final — no second retry (matches the "one
        retry" scope diag_shadow_chart_toolcall_v4_full.py measured).

        Returns valid chart arguments, or None on any other outcome
        (schema-invalid, the model declining via null, or still malformed
        after the retry).
        """
        attempt = await self._run_chart_extraction_attempt(instruction, temperature=0.0)
        if attempt["outcome"] == "malformed":
            attempt = await self._run_chart_extraction_attempt(
                instruction, temperature=_CHART_RETRY_TEMPERATURE
            )
        return attempt["arguments"] if attempt["outcome"] == "match" else None

    async def _run_chart_extraction_attempt(
        self, instruction: str, temperature: float
    ) -> dict[str, Any]:
        """
        One infer -> repair -> classify pass. Returns
        {"outcome": ..., "arguments": ...}, where outcome is one of:
          "malformed"      — envelope missing/wrong-shaped/unknown tool
                              name (retry-eligible on the first attempt).
          "no_tool"        — well-formed {"tool_call": null} — the model
                              declined to chart.
          "schema_invalid" — well-formed tool_call, but
                              validate_chart_arguments() found problems.
          "match"          — valid chart arguments ready to dispatch.

        Ported verbatim from diag_shadow_chart_toolcall_v4_full.py's
        _run_one()/_classify_envelope() — same envelope-shape checks,
        same KNOWN_TOOL_NAMES/validate_chart_arguments() calls.
        """
        try:
            raw = self._runtime.infer(
                prompt      = instruction,
                system      = SYSTEM_PROMPT_FEWSHOT,
                max_tokens  = _CHART_MAX_TOKENS,
                temperature = temperature,
            )
        except Exception as exc:
            logger.debug("MCPToolDispatcher: chart — infer() failed (%s).", exc)
            return {"outcome": "malformed", "arguments": None}

        obj, _repair_outcome = repair_envelope(raw)

        if not isinstance(obj, dict) or "tool_call" not in obj:
            return {"outcome": "malformed", "arguments": None}

        call = obj["tool_call"]
        if call is None:
            return {"outcome": "no_tool", "arguments": None}

        if (
            not isinstance(call, dict)
            or "name" not in call
            or "arguments" not in call
            or not isinstance(call["name"], str)
            or not isinstance(call["arguments"], dict)
            or call["name"] not in KNOWN_TOOL_NAMES
        ):
            return {"outcome": "malformed", "arguments": None}

        problems = validate_chart_arguments(call["arguments"])
        if problems:
            return {"outcome": "schema_invalid", "arguments": None}

        return {"outcome": "match", "arguments": call["arguments"]}

    async def _call_mcp_tool(
        self, session: ClientSession, name: str, arguments: dict[str, Any]
    ) -> tuple[str, bool]:
        """
        Call an MCP tool on an already-open session. Returns (result_text,
        is_error).

        session.call_tool() internally issues a "tools/list" request the
        first time it validates a successful result's output schema against
        a tool name it hasn't seen yet in this session's cache (see
        mcp.client.session.ClientSession._validate_tool_result) — this is
        the SDK's own bookkeeping, not something we invoke here. With one
        session reused for a whole dispatch() call, that fires at most once
        per dispatch (on the first successful call) instead of once per
        tool call, and it's no longer cancelled mid-flight by an immediate
        session teardown.
        """
        result = await session.call_tool(name, arguments)
        text = "\n".join(
            block.text for block in result.content if hasattr(block, "text")
        )
        return text, result.isError
