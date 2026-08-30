"""
Phase 6 integration tests — Localist Tool Dispatcher.

Originally covered the legacy ToolDispatcher class directly (6.1
interface, 6.2 web_search, 6.3 file_op) plus 6.4 ControllerAgent wiring.
ToolDispatcher was deleted in Phase 4 (cleanup, 2026-07-03) once file_op,
url_fetch, and web_search were all first-class in MCPToolDispatcher —
6.1/6.2/6.3 were tests of that now-deleted class and were removed with it.
Direct unit coverage of MCPToolDispatcher and the localist-mcp MCP server
now lives in test_mcp_tool_dispatcher.py and test_mcp_server.py
respectively (Phases 1-3). What remains here is 6.4 only:

  6.4 — ControllerAgent wiring:
         [TOOL RESULTS] slot in prebuilt prompt when tool fires
         Slot ordering: [TOOL RESULTS] < [INSTRUCTION]
         Token ceiling enforced (slot 6 ≤ 500 tokens)
         Tool dispatch failure → graceful absence of [TOOL RESULTS]
         Quality filter: ERROR/repr/short strings excluded from slot 6
         Ingest path (P1) produces no tool results even when tool keyword present
           unless compound detection fires
         file_op / web_search real MCP round trips (Phase 1 follow-up, Phase 3)

ControllerAgent tests use a real SQLite DB via tmp_path.
"""

import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from localist.controller_agent import ControllerAgent, TaskStatus, AgentResult
from localist.mcp_tool_dispatcher import MCPToolDispatcher
from localist.memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    MemoryManager(db_path=path)
    return path


@pytest.fixture()
def mm(db_path: Path) -> MemoryManager:
    return MemoryManager(db_path=db_path)


@pytest.fixture()
def localist_mcp_server(tmp_path: Path):
    """
    Start a real localist-mcp (FastAPI + FastMCP) server as a subprocess,
    sandboxed to tmp_path, for the duration of one test. Torn down after.

    Used by test_file_op_results_appear_in_prompt to exercise the real MCP
    round trip instead of assuming synchronous in-process file_op — see
    sessions-log.md, 2026-07-03, Phase 1 follow-up.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    backend_dir = Path(__file__).resolve().parent.parent
    env = {**os.environ, "LOCALIST_MCP_PROJECT_ROOT": str(tmp_path),
           "LOCALIST_LOG_LEVEL": "WARNING"}

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "localist.mcp_server.main:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd     = str(backend_dir),
        env     = env,
        stdout  = subprocess.DEVNULL,
        stderr  = subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"

    try:
        deadline = time.time() + 15
        healthy  = False
        while time.time() < deadline:
            try:
                if requests.get(f"{base_url}/health", timeout=1.0).status_code == 200:
                    healthy = True
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)

        if not healthy:
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail("localist-mcp test server did not become healthy in time.")

        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture()
def localist_mcp_server_no_langsearch_key(tmp_path: Path):
    """
    Same as localist_mcp_server, but LANGSEARCH_API_KEY is forced empty in
    the subprocess's environment — a separate fixture rather than a
    parameter on the existing one so the file_op fixture stays untouched.

    Set to "" rather than popped: mcp_server/main.py calls load_dotenv() on
    import (needed so the real service picks up LANGSEARCH_API_KEY from
    backend/.env when launched normally), and load_dotenv()'s default
    override=False only skips keys already present in the environment —
    an *absent* key would get silently reloaded from backend/.env in this
    subprocess, defeating the point of this fixture. An empty string is
    "present," so it survives, and web_search.py's `if not api_key` check
    treats "" the same as absent.

    SEARCH_PROVIDER is pinned to "langsearch" for the same reason: this
    fixture's whole point is to force web_search to fail, and backend/.env
    may legitimately have SEARCH_PROVIDER=brave with a real, working
    BRAVE_API_KEY configured for local Brave testing — in which case an
    *unpinned* subprocess would happily dispatch to Brave and succeed
    instead of failing, since only the LangSearch key is forced empty here.

    OLLAMA_API_KEY is forced empty for the identical reason, added
    2026-07-31 alongside mcp_tool_dispatcher.py's WEB_SEARCH_PROVIDER
    fallback: without this, backend/.env's real OLLAMA_API_KEY would
    survive into the subprocess (same load_dotenv()-reload gap as
    LANGSEARCH_API_KEY above), and this test's forced Brave/LangSearch
    failure would trigger a live, authenticated call to
    https://ollama.com/api/web_search instead of a clean
    "OLLAMA_API_KEY not configured" failure — confirmed happening in
    practice before this fixture was updated.

    Used by test_web_search_missing_key_triggers_corpus_fallback (Phase 3)
    to prove controller_agent.py's Step 3b corpus fallback — see
    sessions-log.md, 2026-07-03, Phase 3.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    backend_dir = Path(__file__).resolve().parent.parent
    env = {**os.environ, "LOCALIST_MCP_PROJECT_ROOT": str(tmp_path),
           "LOCALIST_LOG_LEVEL": "WARNING"}
    env["LANGSEARCH_API_KEY"] = ""
    env["SEARCH_PROVIDER"]    = "langsearch"
    env["OLLAMA_API_KEY"]     = ""

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "localist.mcp_server.main:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd     = str(backend_dir),
        env     = env,
        stdout  = subprocess.DEVNULL,
        stderr  = subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"

    try:
        deadline = time.time() + 15
        healthy  = False
        while time.time() < deadline:
            try:
                if requests.get(f"{base_url}/health", timeout=1.0).status_code == 200:
                    healthy = True
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)

        if not healthy:
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail("localist-mcp test server did not become healthy in time.")

        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def make_conv_agent(answer="Test answer."):
    received = []
    agent = MagicMock()
    agent.name = "conversational_agent"
    agent.can_handle.return_value = True
    def run(subtask):
        received.append(subtask)
        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = "conversational_agent",
            status     = TaskStatus.COMPLETE,
            output     = {"answer": answer, "sources": [], "grounded": False},
        )
    agent.run.side_effect = run
    agent._received = received
    return agent


# ---------------------------------------------------------------------------
# 6.4 — ControllerAgent integration
# ---------------------------------------------------------------------------

class TestControllerToolIntegration:

    def test_tool_results_slot_present_in_prompt(self, mm):
        """
        web_search success result flows through to [TOOL RESULTS] slot.

        Originally exercised the runtime.infer() hallucination fallback for
        a missing LANGSEARCH_API_KEY — that fallback was removed in Phase 3
        (web_search migration, 2026-07-03; see sessions-log.md), so this now
        patches MCPToolDispatcher._call_mcp_tool to simulate a successful
        MCP round trip instead, same idiom as test_mcp_tool_dispatcher.py.

        Also patches _open_session (the MCP follow-up's per-dispatch
        session-reuse seam, 2026-07-03) so this stays a pure unit test —
        without it, dispatch() would attempt a real SSE connection before
        ever reaching the mocked _call_mcp_tool.

        Instruction changed 2026-07-29 from "What are the latest oMLX
        release notes?" — that phrasing now correctly (and intentionally)
        routes to the new github_release tool instead (§14.11's
        _GITHUB_RELEASE_KEYWORDS gate), a real behavior change, not a
        regression; this test's own point is just "a successful tool
        result reaches [TOOL RESULTS]," which any web_search-routed
        instruction demonstrates equally well.
        """
        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = [
            "no",    # pre-dispatch episodic check
            "NONE",  # implicit extraction
        ]

        async def fake_open_session(stack):
            return object()

        async def fake_call_mcp_tool(session, name, arguments):
            assert name == "web_search"
            return json.dumps({
                "query":        arguments["query"],
                "result_text":  "• oMLX 0.4.2 released.\n  Supports Gemma 4B quantized.\n  [example.com]",
                "result_count": 1,
            }), False

        conv = make_conv_agent("Here is what I found.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        with patch(
            "localist.mcp_tool_dispatcher.MCPToolDispatcher._open_session",
            side_effect=fake_open_session,
        ), patch(
            "localist.mcp_tool_dispatcher.MCPToolDispatcher._call_mcp_tool",
            side_effect=fake_call_mcp_tool,
        ):
            ctrl.handle_task({
                "instruction": "What's the current status of oMLX?",
                "context":     {"project_context": "LORA"},
            })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[TOOL RESULTS]" in prompt
        assert "oMLX" in prompt

    def test_slot_order_user_before_tool_results(self, mm):
        """
        Mocks the MCP transport seam (same idiom as
        test_tool_results_slot_present_in_prompt above) rather than relying
        on nothing being reachable at mcp_tool_dispatcher._MCP_SERVER_URL —
        that assumption is false whenever a real localist-mcp is running
        locally (e.g. via start_localist.sh) or in any CI environment with
        a stray process on that port, and previously left this test's
        assertion inside an `if "[TOOL RESULTS]" in prompt:` guard that
        could pass vacuously without ever checking slot order.
        """
        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = ["no", "NONE"]

        async def fake_open_session(stack):
            return object()

        async def fake_call_mcp_tool(session, name, arguments):
            return json.dumps({
                "query":        arguments["query"],
                "result_text":  "• Search result with enough content to pass quality filter.",
                "result_count": 1,
            }), False

        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        with patch(
            "localist.mcp_tool_dispatcher.MCPToolDispatcher._open_session",
            side_effect=fake_open_session,
        ), patch(
            "localist.mcp_tool_dispatcher.MCPToolDispatcher._call_mcp_tool",
            side_effect=fake_call_mcp_tool,
        ):
            ctrl.handle_task({
                "instruction": "What are the latest changes?",
                "context":     {"project_context": "LORA"},
            })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[TOOL RESULTS]" in prompt
        assert prompt.index("[TOOL RESULTS]") < prompt.index("[INSTRUCTION]")

    def test_tool_slot_ceiling_enforced(self, mm):
        """
        Slot 6 must not exceed 1500 tokens (6000 chars) in the prompt.

        Mocks the MCP transport seam (see test_slot_order_user_before_tool_results
        above) instead of relying on nothing being reachable on
        _MCP_SERVER_URL, for the same reason.
        """
        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = ["no", "NONE"]

        async def fake_open_session(stack):
            return object()

        async def fake_call_mcp_tool(session, name, arguments):
            return json.dumps({
                "query":        arguments["query"],
                "result_text":  "• " + "A" * 9000,  # far exceeds 1500-token slot ceiling
                "result_count": 1,
            }), False

        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        with patch(
            "localist.mcp_tool_dispatcher.MCPToolDispatcher._open_session",
            side_effect=fake_open_session,
        ), patch(
            "localist.mcp_tool_dispatcher.MCPToolDispatcher._call_mcp_tool",
            side_effect=fake_call_mcp_tool,
        ):
            ctrl.handle_task({
                "instruction": "What are the latest changes?",
                "context":     {"project_context": "LORA"},
            })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[TOOL RESULTS]" in prompt
        start = prompt.index("[TOOL RESULTS]")
        # Find the next slot label after TOOL RESULTS (or end of string)
        next_slot = len(prompt)
        for label in ["[INSTRUCTION]", "[WORKING MEMORY]", "[WORKING STATE]",
                      "[EPISODIC MEMORY]", "[CONTEXT]"]:
            pos = prompt.find(label, start + 1)
            if pos != -1:
                next_slot = min(next_slot, pos)
        tool_section = prompt[start:next_slot]
        assert len(tool_section) // 4 <= 1505  # 1500 + small label tolerance

    def test_tool_dispatch_failure_graceful(self, mm):
        """
        If localist-mcp is unreachable, task still completes; [TOOL RESULTS]
        absent. Mocks _open_session to raise (the documented test seam for
        simulating an unreachable MCP server — see its docstring in
        mcp_tool_dispatcher.py) rather than relying on nothing actually
        listening on _MCP_SERVER_URL, which is false whenever a real
        localist-mcp happens to be running locally or in CI.
        """
        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = ["no", "NONE"]

        async def fake_open_session_unreachable(stack):
            raise ConnectionError("mock: localist-mcp unreachable")

        conv = make_conv_agent("Graceful answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        with patch(
            "localist.mcp_tool_dispatcher.MCPToolDispatcher._open_session",
            side_effect=fake_open_session_unreachable,
        ):
            result = ctrl.handle_task({
                "instruction": "What are the latest changes?",
                "context":     {"project_context": "LORA"},
            })

        assert result["status"] == "complete"
        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[TOOL RESULTS]" not in prompt

    def test_quality_filter_excludes_error_results(self, mm):
        """
        Tool results beginning with ERROR: must not appear in slot 6. Mocks
        _open_session to raise (see test_tool_dispatch_failure_graceful
        above) so the ERROR: result is deterministic rather than depending
        on _MCP_SERVER_URL being unreachable.
        """
        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = ["no", "NONE"]

        async def fake_open_session_unreachable(stack):
            raise ConnectionError("mock: localist-mcp unreachable")

        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        with patch(
            "localist.mcp_tool_dispatcher.MCPToolDispatcher._open_session",
            side_effect=fake_open_session_unreachable,
        ):
            ctrl.handle_task({
                "instruction": "What are the latest changes?",
                "context":     {"project_context": "LORA"},
            })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "ERROR:" not in prompt

    def test_ingest_path_no_tool_results(self, mm):
        """P1 (ingest) fires alone — no tool results even with 'latest' keyword
        unless compound detection fires (both ingest + web_search keywords)."""
        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.return_value = "NONE"

        wiki = MagicMock()
        wiki.name = "wiki_agent"
        wiki.can_handle.return_value = True
        wiki.run.return_value = AgentResult(
            subtask_id = "w-0",
            agent_name = "wiki_agent",
            status     = TaskStatus.COMPLETE,
            output     = {"new_pages": [], "applied": False},
        )

        ctrl = ControllerAgent(runtime=rt, agents=[wiki], memory_manager=mm)
        ctrl.handle_task({
            "instruction": "ingest this document",
            "context":     {"raw_path": "/data/notes.md"},
        })

        # wiki_agent was dispatched; routing has no tools
        routing = wiki.run.call_args[0][0].context["_routing"]
        assert routing["tools_to_call"] == []

    def test_file_op_results_appear_in_prompt(self, mm, tmp_path, localist_mcp_server):
        """
        file_op read result flows through to [TOOL RESULTS] slot.

        Since Phase 1 (MCP server + file_op migration, 2026-07-03), file_op is
        served out-of-process by localist-mcp — this is a real MCP round trip
        via the localist_mcp_server fixture, not an in-process assumption.
        """
        notes = tmp_path / "generated_files" / "notes.md"
        notes.parent.mkdir(parents=True, exist_ok=True)
        notes.write_text(
            "LORA memory system uses SQLite for persistence.", encoding="utf-8"
        )

        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = [
            "no",   # pre-dispatch episodic check
            "NONE", # implicit extraction
        ]

        conv = make_conv_agent("Here is the file content.")
        ctrl = ControllerAgent(
            runtime=rt, agents=[conv], memory_manager=mm
        )
        with patch("localist.mcp_tool_dispatcher._MCP_SERVER_URL", localist_mcp_server + "/sse"):
            ctrl.handle_task({
                "instruction": "read the file notes.md",
                "context": {
                    "project_root":   str(tmp_path),
                    "file_op_action": "read",
                    "file_op_path":   "notes.md",
                    "project_context": "LORA",
                },
            })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[TOOL RESULTS]" in prompt
        assert "SQLite" in prompt

    def test_web_search_missing_key_triggers_corpus_fallback(
        self, mm, tmp_path, monkeypatch, localist_mcp_server_no_langsearch_key
    ):
        """
        Proves controller_agent.py's Step 3b corpus fallback actually fires
        when web_search fails due to a missing LANGSEARCH_API_KEY.

        This was flagged as never provably exercised end-to-end before
        Phase 3 (web_search migration, 2026-07-03) — see sessions-log.md.
        rt.infer is given exactly the 2 side_effect values ControllerAgent's
        own internal calls need (pre-dispatch episodic check, implicit
        extraction); if the removed runtime.infer() hallucination fallback
        were still being called anywhere on this path, the mock would raise
        StopIteration instead of the assertions below failing quietly.

        WEB_SEARCH_PROVIDER is pinned to "brave" (mcp_tool_dispatcher.py's
        default, 2026-07-31) so this test's outcome doesn't depend on
        whatever a developer's own shell happens to export — the point
        here is Brave/LangSearch failing and falling through to a
        deliberately-unconfigured Ollama fallback (see
        localist_mcp_server_no_langsearch_key's OLLAMA_API_KEY="" above),
        not Ollama succeeding as primary.
        """
        monkeypatch.setenv("WEB_SEARCH_PROVIDER", "brave")
        doc_path    = tmp_path / "zylophonic-notes.md"
        doc_content = "Zylophonic quarterly earnings update web search"
        doc_path.write_text(doc_content, encoding="utf-8")
        mm.index_document(doc_path, "raw", content=doc_content, embed=False)

        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = [
            "no",   # pre-dispatch episodic check
            "NONE", # implicit extraction
        ]

        conv = make_conv_agent("Here is what I found.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        with patch(
            "localist.mcp_tool_dispatcher._MCP_SERVER_URL",
            localist_mcp_server_no_langsearch_key + "/sse",
        ):
            ctrl.handle_task({
                "instruction": "do a web search for Zylophonic quarterly earnings update",
                "context":     {"project_context": "LORA"},
            })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[CONTEXT]" in prompt
        assert "Zylophonic" in prompt
        assert rt.infer.call_count == 2

    def test_news_search_double_miss_triggers_corpus_fallback(self, mm, tmp_path):
        """
        Proves controller_agent.py's Step 3b corpus fallback still fires
        end-to-end when the news_search chain is fully exhausted: tier-1
        NewsAPI misses, tier-2 Brave fallback also fails.

        This is exactly the case the Step 3b `r.tool_name.startswith(
        "news_search")` clause (added alongside mcp_tool_dispatcher.
        _run_news_search, news-query-routing plan, 2026-07-22) exists for —
        without it, the Brave-fallback ToolResult's renamed tool_name
        ("news_search:brave_fallback") would silently escape Step 3b's
        exact `tool_name == "web_search"` check, the same gap "research"
        hit and was fixed for the same way (see the comment above
        _web_search_failed in controller_agent.py).

        Mocks the MCP transport seam directly (same idiom as
        test_tool_results_slot_present_in_prompt above) rather than
        spinning up a real localist-mcp subprocess — this is a pure
        controller_agent.py/mcp_tool_dispatcher.py wiring proof, not a
        real-network one (that's what live verification is for).
        """
        doc_path    = tmp_path / "zylophonic-news.md"
        doc_content = "Zylophonic quarterly earnings breaking news update"
        doc_path.write_text(doc_content, encoding="utf-8")
        mm.index_document(doc_path, "raw", content=doc_content, embed=False)

        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = [
            "no",   # pre-dispatch episodic check
            "NONE", # implicit extraction
        ]

        async def fake_open_session(stack):
            return object()

        async def fake_call_mcp_tool(session, name, arguments):
            if name == "news_search":
                return json.dumps({
                    "query": arguments["query"], "result_text": "", "result_count": 0,
                    "is_miss": True,
                }), False
            assert name == "web_search"
            return (
                "Error executing tool web_search: ERROR: BRAVE_API_KEY not configured"
            ), True

        conv = make_conv_agent("Here is what I found.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        with patch(
            "localist.mcp_tool_dispatcher.MCPToolDispatcher._open_session",
            side_effect=fake_open_session,
        ), patch(
            "localist.mcp_tool_dispatcher.MCPToolDispatcher._call_mcp_tool",
            side_effect=fake_call_mcp_tool,
        ):
            ctrl.handle_task({
                "instruction": "any breaking news on Zylophonic quarterly earnings?",
                "context":     {"project_context": "LORA"},
            })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[CONTEXT]" in prompt
        assert "Zylophonic" in prompt
        assert rt.infer.call_count == 2


# ---------------------------------------------------------------------------
# MCP follow-up (2026-07-03) — session reuse, live over real SSE
# ---------------------------------------------------------------------------

class TestMCPSessionReuseLive:
    """
    Unlike test_mcp_tool_dispatcher.py's TestSessionReuse (which mocks
    _open_session/_call_mcp_tool), this exercises the real localist-mcp
    subprocess over real SSE, confirming what a live trace actually shows
    on the wire: how many "tools/list" requests get sent, and whether any
    of them get cancelled mid-flight by a session teardown.

    ClientSession.call_tool() internally issues its own "tools/list"
    request the first time it validates a successful result's output
    schema against a tool name it hasn't cached yet (see
    mcp.client.session.ClientSession._validate_tool_result) — this isn't
    something MCPToolDispatcher calls itself, so there's nothing in
    mcp_tool_dispatcher.py to delete. Before the session-reuse refactor, a
    fresh ClientSession was opened per tool call, so this fired — and got
    cancelled by the immediate session teardown — on every single call.
    With one session reused per dispatch(), it fires at most once (on the
    first successful call) and, since the session stays open, completes
    normally instead of being cancelled.
    """

    def test_multi_call_dispatch_sends_at_most_one_tools_list_uncancelled(
        self, tmp_path, localist_mcp_server, caplog
    ):
        (tmp_path / "generated_files").mkdir(parents=True, exist_ok=True)
        (tmp_path / "generated_files" / "notes.md").write_text(
            "hello from notes", encoding="utf-8"
        )

        dispatcher = MCPToolDispatcher(
            runtime=None, mcp_server_url=localist_mcp_server + "/sse"
        )

        with caplog.at_level(logging.DEBUG, logger="mcp.client.sse"):
            results = dispatcher.dispatch(
                tools_to_call = ["file_op", "file_op", "file_op"],
                instruction   = "read notes.md three times",
                context       = {"file_op_action": "read", "file_op_path": "notes.md"},
            )

        assert len(results) == 3
        assert all(r.success is True for r in results)
        assert all(r.result == "hello from notes" for r in results)

        tools_list_count = caplog.text.count("method='tools/list'")
        assert tools_list_count <= 1, (
            f"expected at most one tools/list request for the whole dispatch, "
            f"saw {tools_list_count}"
        )
        assert "CancelledError" not in caplog.text
        assert "cancel" not in caplog.text.lower()
