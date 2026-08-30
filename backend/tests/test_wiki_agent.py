"""
Tests for wiki_agent — _validate_links() and run() wiring.

Covers:
  1. Link to an existing page resolves silently.
  2. Link to a self-proposed page resolves silently.
  3. Link to neither is flagged; content is provably unchanged.
  4. Case-mismatch resolves after normalization.
  5. Word-count mismatch does NOT resolve (normalization is narrow).
  6. Section scoping — links outside Mapped/Related Pages are ignored.
  7. Missing Mapped/Related Pages sections don't error.
  8. End-to-end run() wiring — unresolved_links in output + warning logged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from unittest.mock import patch

from localist import wiki_maintenance_log
from localist.wiki_agent import (
    Actions,
    ApplyDiff,
    CreatePage,
    WikiAgent,
    _validate_links,
    apply_unified_diff,
    build_diff_prompt,
    build_user_prompt,
    build_slim_prompt,
    parse_model_xml,
    sweep_expired_snapshots,
)
from localist.controller_agent import SubTask, TaskStatus


@pytest.fixture(autouse=True)
def _isolate_wiki_maintenance_log(tmp_path, monkeypatch):
    """
    _prune_page_snapshots() now writes to wiki_maintenance_log on every
    prune (see §17.8 audit-log parity), so any test that ages a snapshot
    past the TTL and triggers a prune would otherwise append to the real
    backend/logs/wiki_maintenance.log. Point it at a tmp_path file for
    every test in this module — matches the _isolate_log fixture already
    used for the same reason in test_memory_phase1.py.
    """
    monkeypatch.setattr(wiki_maintenance_log, "_LOG_PATH", tmp_path / "wiki_maintenance.log")


# ---------------------------------------------------------------------------
# Runtime fake — established convention for this file
#
# Use _FakeRuntime (not MagicMock) for any test that exercises WikiAgent.run().
# MagicMock auto-creates infer_with_file as an attribute, causing hasattr()
# in run() to return True and silently routing the test down the
# infer_with_file / build_slim_prompt path instead of the infer() string-prompt
# path. _FakeRuntime has exactly the two methods in the RuntimeClient Protocol
# and deliberately omits infer_with_file, so the routing is deterministic.
# ---------------------------------------------------------------------------

class _FakeRuntime:
    """Protocol-shaped fake — has only infer() and embed(), matching
    RuntimeClient exactly. Deliberately has no infer_with_file, so
    run()'s hasattr() check correctly routes to the infer() string-prompt
    path, the same way a real OMLXRuntimeClient/FoundryRuntimeClient
    without infer_with_file support would."""

    def __init__(self, response: str) -> None:
        self._response = response

    def infer(self, *args, **kwargs) -> str:
        return self._response

    def embed(self, text: str) -> list[float]:
        return [0.0] * 768


class _CapturingFakeRuntime(_FakeRuntime):
    """Same contract as _FakeRuntime, but records the prompt kwarg passed to
    infer() so tests can assert on the assembled diff prompt's contents."""

    def __init__(self, response: str) -> None:
        super().__init__(response)
        self.captured_prompt: str | None = None

    def infer(self, *args, **kwargs) -> str:
        self.captured_prompt = kwargs.get("prompt")
        return super().infer(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_actions(*pages: tuple[str, str]) -> Actions:
    """Create an Actions object with new_pages from (page_name, content) pairs."""
    return Actions(
        new_pages=[
            CreatePage(page_name=name, page_type="RESEARCH_NOTE", content=content)
            for name, content in pages
        ],
        diffs=[],
    )


def _page_with_links(mapped: str = "", related: str = "", concepts: str = "") -> str:
    """Build a page body with optional links in each section."""
    mapped_body = f"- [[{mapped}]] — a mapped page.\n" if mapped else "- null\n"
    related_body = f"- [[{related}]]\n" if related else ""
    concepts_body = f"- Some concept with [[{concepts}]].\n" if concepts else "- A concept.\n"
    return (
        "## Summary\n\nA research note.\n\n"
        "## Details\n\n"
        "### Extracted Concepts\n\n"
        f"{concepts_body}\n"
        "### Mapped Pages\n\n"
        f"{mapped_body}\n"
        "### Proposed New Pages\n\n"
        "- null\n\n"
        "## Related Pages\n\n"
        f"{related_body}\n"
        "## Revision History\n\n"
        "- 2026-06-19 — Created.\n"
    )


# ---------------------------------------------------------------------------
# Test 1 — Link to an existing page resolves silently
# ---------------------------------------------------------------------------

def test_existing_page_resolves():
    actions = _make_actions(
        ("my-page", _page_with_links(mapped="existing-page", related="existing-page"))
    )
    wiki_pages = {"existing-page": "Some content."}
    result = _validate_links(actions, wiki_pages)
    assert result == {}


# ---------------------------------------------------------------------------
# Test 2 — Link to a self-proposed page resolves silently
# ---------------------------------------------------------------------------

def test_self_proposed_page_resolves():
    content_a = _page_with_links(related="new-page-b")
    actions = _make_actions(
        ("new-page-a", content_a),
        ("new-page-b", _page_with_links()),
    )
    wiki_pages = {}  # nothing exists yet
    result = _validate_links(actions, wiki_pages)
    assert result == {}


# ---------------------------------------------------------------------------
# Test 3 — Nonexistent link flagged; content byte-for-byte unchanged
# ---------------------------------------------------------------------------

def test_nonexistent_link_flagged_content_unchanged():
    content_before = _page_with_links(related="totally-nonexistent-page")
    actions = _make_actions(("my-page", content_before))
    wiki_pages = {}

    result = _validate_links(actions, wiki_pages)

    assert result == {"my-page": ["totally-nonexistent-page"]}
    # Content must be character-for-character identical after the call
    assert actions.new_pages[0].content == content_before


# ---------------------------------------------------------------------------
# Test 4 — Case-mismatch resolves post-normalization
# ---------------------------------------------------------------------------

def test_case_mismatch_resolves_after_normalization():
    # Real corpus link: [[Localist Master Project Outline]]
    # Real page stem:   localist-master-project-outline
    content = _page_with_links(related="Localist Master Project Outline")
    actions = _make_actions(("my-page", content))
    wiki_pages = {"localist-master-project-outline": "Some content."}

    result = _validate_links(actions, wiki_pages)

    assert result == {}  # resolves — no unresolved entries


# ---------------------------------------------------------------------------
# Test 5 — Word-count mismatch does NOT resolve (normalization is narrow)
# ---------------------------------------------------------------------------

def test_word_count_mismatch_not_resolved():
    # Real corpus defect: [[Localist Software Stack Overview]]
    # vs page stem:       localist-software-stack  (one word missing)
    content = _page_with_links(related="Localist Software Stack Overview")
    actions = _make_actions(("my-page", content))
    wiki_pages = {"localist-software-stack": "Some content."}

    result = _validate_links(actions, wiki_pages)

    assert "my-page" in result
    assert "localist-software-stack-overview" in result["my-page"]


# ---------------------------------------------------------------------------
# Test 6 — Links outside Mapped/Related Pages sections are ignored
# ---------------------------------------------------------------------------

def test_links_outside_sections_ignored():
    # Link appears ONLY in Extracted Concepts, not in Mapped Pages or Related Pages
    content = _page_with_links(concepts="out-of-scope-link")
    actions = _make_actions(("my-page", content))
    wiki_pages = {}  # out-of-scope-link doesn't exist

    result = _validate_links(actions, wiki_pages)

    assert result == {}  # not scanned → not flagged


# ---------------------------------------------------------------------------
# Test 7 — Missing Mapped/Related Pages sections don't error
# ---------------------------------------------------------------------------

def test_missing_sections_no_error():
    # Minimal page with no Mapped Pages or Related Pages headings at all
    content = "## Summary\n\nJust a summary, no section structure.\n"
    actions = _make_actions(("bare-page", content))
    wiki_pages = {}

    result = _validate_links(actions, wiki_pages)

    assert result == {}


# ---------------------------------------------------------------------------
# Test 8 — End-to-end run() wiring
# ---------------------------------------------------------------------------

_RUN_XML = """\
<actions>
  <action name="create_page">
    <page_name>run-test-page</page_name>
    <page_type>RESEARCH_NOTE</page_type>
    <content>
## Summary

A test page.

## Details

### Extracted Concepts

- A concept.

### Mapped Pages

- [[existing-wiki-page]] — Exists.

### Proposed New Pages

- null

## Related Pages

- [[existing-wiki-page]]
- [[totally-missing-page]]

## Revision History

- 2026-06-19 — Created.
    </content>
  </action>
</actions>
"""


# ---------------------------------------------------------------------------
# Stub fixtures for prompt-content tests (Rules 1–7)
# ---------------------------------------------------------------------------

_PROMPT_STUBS = dict(
    schema_text  = "# Schema\n",
    templates    = {},
    wiki_context = "(no existing wiki pages)",
    raw_filename = "test.md",
)


# ---------------------------------------------------------------------------
# Test — Rule 7 present in build_user_prompt() output
# ---------------------------------------------------------------------------

def test_build_user_prompt_contains_rule7():
    """build_user_prompt() must contain Rule 7 (verbatim-link-target constraint)."""
    out = build_user_prompt(**_PROMPT_STUBS, raw_content="Some raw content.")
    assert "Every [[...]] link target" in out
    assert "MUST exactly match an existing page name" in out
    assert "propose it as a new page" in out


# ---------------------------------------------------------------------------
# Test — link-target rule (rule 8, since the OKF front-matter rule became
# rule 2) present in build_slim_prompt() output, identical wording
# ---------------------------------------------------------------------------

def test_build_slim_prompt_contains_link_rule_identical_to_user_prompt():
    """build_slim_prompt() must contain the link-target rule, byte-identical
    to build_user_prompt()."""
    user_out = build_user_prompt(**_PROMPT_STUBS, raw_content="Some raw content.")
    slim_out = build_slim_prompt(**_PROMPT_STUBS)

    assert "Every [[...]] link target" in slim_out
    assert "MUST exactly match an existing page name" in slim_out
    assert "propose it as a new page" in slim_out

    # Extract just the link-rule text from each and confirm they match exactly.
    import re
    def _extract_link_rule(text: str) -> str:
        m = re.search(r"8\. Every \[\[\.\.\..*?instead of linking to a guessed name\.", text, re.DOTALL)
        assert m, "Link-target rule (8.) not found in prompt output"
        return m.group(0)

    assert _extract_link_rule(user_out) == _extract_link_rule(slim_out), (
        "Link-target rule wording differs between build_user_prompt() and build_slim_prompt()"
    )


# ---------------------------------------------------------------------------
# Test — Rules 1–7 text unchanged in both prompt functions (rule 2 is the
# OKF front-matter fields rule added alongside index.md/logs.md support)
# ---------------------------------------------------------------------------

_RULE_SUBSTRINGS = [
    "1. Create exactly one RESEARCH_NOTE",
    "2. Front matter MUST also include: title",
    "description",
    "resource",
    "tags",
    "timestamp",
    "3. Details MUST contain three H3 subsections",
    "4. Optionally propose new CONCEPT or SYSTEM pages",
    "5. For existing wiki pages, propose minimal unified diffs",
    "6. Page names MUST be kebab-case",
    "7. Use ",
    "as the date in all Revision History entries",
]


def test_prompt_rules_1_through_7_unchanged():
    """Rules 1–7 must be present and unchanged in both build_user_prompt() and build_slim_prompt()."""
    user_out = build_user_prompt(**_PROMPT_STUBS, raw_content="Some raw content.")
    slim_out = build_slim_prompt(**_PROMPT_STUBS)
    for expected in _RULE_SUBSTRINGS:
        assert expected in user_out, f"build_user_prompt() missing: {expected!r}"
        assert expected in slim_out, f"build_slim_prompt() missing: {expected!r}"


# ---------------------------------------------------------------------------
# Test 8 — End-to-end run() wiring
# ---------------------------------------------------------------------------

def test_run_unresolved_links_in_output_and_logged(tmp_path: Path, caplog):
    # Set up a minimal wiki directory structure
    wiki_dir      = tmp_path / "wiki"
    wiki_dir.mkdir()
    schema_path   = tmp_path / "SCHEMA.md"
    schema_path.write_text("# Schema\n", encoding="utf-8")
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    raw_path      = tmp_path / "raw-input.md"
    raw_path.write_text("Some raw content.\n", encoding="utf-8")

    # Pre-create the existing page so it appears in wiki_pages
    (wiki_dir / "existing-wiki-page.md").write_text("Existing page content.\n", encoding="utf-8")

    rt = _FakeRuntime(_RUN_XML)
    # _FakeRuntime has no infer_with_file → hasattr() returns False → infer() path taken
    assert not hasattr(rt, "infer_with_file")

    agent = WikiAgent(runtime=rt, project_root=tmp_path)

    subtask = MagicMock()
    subtask.subtask_id  = "run-test-0"
    subtask.instruction = "ingest raw-input.md"
    subtask.context = {
        "raw_path":     str(raw_path),
        "wiki_dir":     str(wiki_dir),
        "schema_path":  str(schema_path),
        "templates_dir": str(templates_dir),
        "auto_apply":   False,
    }

    with caplog.at_level(logging.WARNING, logger="localist.wiki_agent"):
        result = agent.run(subtask)

    assert result.status == TaskStatus.COMPLETE
    assert "unresolved_links" in result.output

    unresolved = result.output["unresolved_links"]
    assert "run-test-page" in unresolved
    assert "totally-missing-page" in unresolved["run-test-page"]
    assert "existing-wiki-page" not in unresolved.get("run-test-page", [])

    # Confirm the warning was logged
    assert any(
        "totally-missing-page" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


# ---------------------------------------------------------------------------
# Tests — parse_model_xml() strips leading/trailing whitespace from content
# (Open Item 6 write-time fix)
# ---------------------------------------------------------------------------

# Standard model output: Gemma places a newline immediately after <content>.
# This exercises the __CONTENT_N__ placeholder path (all content blocks go
# through _shield_content_blocks, so raw_content is always a placeholder for
# non-empty content; the if-branch in parse_model_xml resolves it back to
# contents[idx]).  The strip() is applied to raw_content after that resolution.
_XML_LEADING_NEWLINE = """\
<actions>
  <action name="create_page">
    <page_name>my-research-note</page_name>
    <page_type>RESEARCH_NOTE</page_type>
    <content>
---
title: My Research Note
type: research-note
---

## Summary

Body text.
    </content>
  </action>
</actions>"""


def test_parse_model_xml_strips_leading_newline_from_content():
    """parse_model_xml must strip the leading newline Gemma places after <content>."""
    actions = parse_model_xml(_XML_LEADING_NEWLINE)
    assert len(actions.new_pages) == 1
    content = actions.new_pages[0].content
    # After stripping, content must start with the frontmatter fence, not a blank line
    assert not content.startswith("\n"), "Leading newline was not stripped"
    assert content.startswith("---"), "Frontmatter fence must be first character"


def test_parse_model_xml_strips_trailing_whitespace_from_content():
    """parse_model_xml must strip trailing whitespace/newlines from content."""
    actions = parse_model_xml(_XML_LEADING_NEWLINE)
    content = actions.new_pages[0].content
    # The XML above has trailing spaces + newline before </content>
    assert not content.endswith("    "), "Trailing indent was not stripped"
    assert content.endswith("Body text."), "Content should end at last real text line"


# XML where content has only trailing whitespace (no leading newline) —
# confirms strip() is applied regardless of which side has the whitespace.
_XML_TRAILING_ONLY = """\
<actions>
  <action name="create_page">
    <page_name>concept-page</page_name>
    <page_type>CONCEPT</page_type>
    <content>## A Concept

Some concept text.
  </content>
  </action>
</actions>"""


def test_parse_model_xml_strips_trailing_only_whitespace():
    """parse_model_xml must strip trailing whitespace even when there is no leading newline."""
    actions = parse_model_xml(_XML_TRAILING_ONLY)
    assert len(actions.new_pages) == 1
    content = actions.new_pages[0].content
    assert not content.endswith("  "), "Trailing spaces were not stripped"
    assert content.startswith("## A Concept")


# ---------------------------------------------------------------------------
# parse_model_xml() — <diff> blocks must be shielded like <content> blocks
#
# Live finding (gemma4:31b-cloud, 2026-07-09, diagnostics/diag_wiki_agent_
# diff_only.py): a real diff against wiki/localist-software-stack.md carried
# an unchanged context line "Local Tools & Libraries" — the bare "&" is not
# valid inside XML character data and broke ET.fromstring() before
# _shield_content_blocks() covered <diff> blocks too.
# ---------------------------------------------------------------------------

_XML_DIFF_WITH_AMPERSAND = """\
<actions>
  <action name="apply_diff">
    <page_name>localist-software-stack</page_name>
    <diff>
@@ -1,3 +1,3 @@
 ### Extracted Concepts

-- **Local Tools:** SQLite, pydantic.
+- **Local Tools & Libraries:** SQLite, pydantic, uvicorn.
</diff>
  </action>
</actions>
"""


def test_parse_model_xml_shields_ampersand_in_diff_block():
    """A bare '&' in an unchanged diff context line must not break XML
    parsing — <diff> blocks are shielded the same way <content> is."""
    actions = parse_model_xml(_XML_DIFF_WITH_AMPERSAND)
    assert len(actions.diffs) == 1
    assert actions.diffs[0].page_name == "localist-software-stack"
    assert "Local Tools & Libraries" in actions.diffs[0].diff


# ---------------------------------------------------------------------------
# Diff-only path — WikiAgent._run_diff_only() / build_diff_prompt()
# ---------------------------------------------------------------------------

def _diff_only_env(tmp_path: Path, page_content: str) -> dict[str, Path]:
    """Set up a minimal wiki/schema/templates directory tree for diff-only tests."""
    wiki_dir      = tmp_path / "wiki"
    wiki_dir.mkdir()
    schema_path   = tmp_path / "SCHEMA.md"
    schema_path.write_text("# Schema\n", encoding="utf-8")
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (wiki_dir / "existing-page.md").write_text(page_content, encoding="utf-8")
    return {
        "wiki_dir":      wiki_dir,
        "schema_path":   schema_path,
        "templates_dir": templates_dir,
    }


_PAGE_CONTENT = "## Summary\n\nOld summary.\n"

_DIFF_ONLY_XML = """\
<actions>
  <action name="apply_diff">
    <page_name>existing-page</page_name>
    <diff>
@@ -3,1 +3,1 @@
-Old summary.
+New summary.
</diff>
  </action>
</actions>
"""

_DIFF_ONLY_XML_WITH_CREATE_PAGE = """\
<actions>
  <action name="create_page">
    <page_name>should-not-exist</page_name>
    <page_type>CONCEPT</page_type>
    <content>## Should not be created</content>
  </action>
  <action name="apply_diff">
    <page_name>existing-page</page_name>
    <diff>
@@ -3,1 +3,1 @@
-Old summary.
+New summary.
</diff>
  </action>
</actions>
"""


def test_diff_prompt_omits_raw_file_section():
    """build_diff_prompt() must not contain the RAW FILE TO INGEST section
    build_user_prompt()/build_slim_prompt() carry, and must scope the model
    to a single named target page."""
    out = build_diff_prompt(
        schema_text  = "# Schema\n",
        templates    = {},
        page_name    = "existing-page",
        page_content = _PAGE_CONTENT,
        instruction  = "Update it to reflect the new backend.",
    )
    assert "RAW FILE TO INGEST" not in out
    assert "TARGET WIKI PAGE: existing-page" in out
    assert "Old summary." in out
    assert "create_page" in out  # only mentioned to disallow it
    assert "NOT propose a create_page action" in out


def test_diff_prompt_with_tool_context_adds_fetched_context_section():
    """Priority 1c compound plans (diff_target + a dispatched tool) pass
    tool_context through — it must appear as a distinct section between
    the target page and the task instructions."""
    out = build_diff_prompt(
        schema_text  = "# Schema\n",
        templates    = {},
        page_name    = "existing-page",
        page_content = _PAGE_CONTENT,
        instruction  = "Update it to reflect the changelog.",
        tool_context = "Title: Changelog\nSource: https://example.com\n\nAdded X.",
    )
    assert "# FETCHED CONTEXT" in out
    assert "Added X." in out
    # Ordering: target page content, then fetched context, then the task.
    assert out.index("Old summary.") < out.index("# FETCHED CONTEXT") < out.index("# YOUR TASK")


def test_diff_prompt_without_tool_context_is_unchanged():
    """tool_context omitted (default None) reproduces prior output exactly
    — no '# FETCHED CONTEXT' section at all. Regression guard."""
    out = build_diff_prompt(
        schema_text  = "# Schema\n",
        templates    = {},
        page_name    = "existing-page",
        page_content = _PAGE_CONTENT,
        instruction  = "Update it to reflect the new backend.",
    )
    assert "# FETCHED CONTEXT" not in out


def test_run_dispatches_to_diff_only_path_without_calling_resolve_raw_path(tmp_path: Path):
    """diff_target present, raw_path absent → _run_diff_only() runs;
    _resolve_raw_path() (the ingest-only path guard) must never be called."""
    paths = _diff_only_env(tmp_path, _PAGE_CONTENT)
    rt = _FakeRuntime(_DIFF_ONLY_XML)
    agent = WikiAgent(runtime=rt, project_root=tmp_path)

    subtask = MagicMock()
    subtask.subtask_id  = "diff-test-0"
    subtask.instruction = "Update existing-page to reflect the new backend."
    subtask.context = {
        "diff_target":   "existing-page",
        "wiki_dir":      str(paths["wiki_dir"]),
        "schema_path":   str(paths["schema_path"]),
        "templates_dir": str(paths["templates_dir"]),
        "auto_apply":    False,
    }

    with patch.object(WikiAgent, "_resolve_raw_path") as mock_resolve:
        result = agent.run(subtask)

    mock_resolve.assert_not_called()
    assert result.status == TaskStatus.COMPLETE
    assert result.output["raw_filename"] is None
    assert result.output["diff_target"]  == "existing-page"
    assert len(result.output["diffs"])   == 1
    assert result.output["new_pages"]    == []


def test_run_diff_only_threads_tool_context_into_prompt(tmp_path: Path):
    """context["tool_context"] (set by ControllerAgent when Priority 1c's
    compound plan dispatched a tool alongside the pinned diff target) must
    reach the model's actual prompt."""
    paths = _diff_only_env(tmp_path, _PAGE_CONTENT)
    rt = _CapturingFakeRuntime(_DIFF_ONLY_XML)
    agent = WikiAgent(runtime=rt, project_root=tmp_path)

    subtask = MagicMock()
    subtask.subtask_id  = "diff-test-tool-context"
    subtask.instruction = "Update existing-page to reflect the changelog."
    subtask.context = {
        "diff_target":   "existing-page",
        "wiki_dir":      str(paths["wiki_dir"]),
        "schema_path":   str(paths["schema_path"]),
        "templates_dir": str(paths["templates_dir"]),
        "auto_apply":    False,
        "tool_context":  "Title: Changelog\nSource: https://example.com\n\nAdded X.",
    }

    result = agent.run(subtask)

    assert result.status == TaskStatus.COMPLETE
    assert rt.captured_prompt is not None
    assert "# FETCHED CONTEXT" in rt.captured_prompt
    assert "Added X." in rt.captured_prompt


def test_diff_only_run_applies_diff_to_disk_when_auto_apply(tmp_path: Path):
    paths = _diff_only_env(tmp_path, _PAGE_CONTENT)
    rt = _FakeRuntime(_DIFF_ONLY_XML)
    agent = WikiAgent(runtime=rt, project_root=tmp_path)

    subtask = MagicMock()
    subtask.subtask_id  = "diff-test-1"
    subtask.instruction = "Update existing-page to reflect the new backend."
    subtask.context = {
        "diff_target":   "existing-page",
        "wiki_dir":      str(paths["wiki_dir"]),
        "schema_path":   str(paths["schema_path"]),
        "templates_dir": str(paths["templates_dir"]),
        "auto_apply":    True,
    }

    result = agent.run(subtask)

    assert result.status == TaskStatus.COMPLETE
    assert result.output["applied"] is True
    assert "existing-page" in result.output["written"]
    updated = (paths["wiki_dir"] / "existing-page.md").read_text(encoding="utf-8")
    assert "New summary." in updated
    assert "Old summary." not in updated


def test_diff_only_run_discards_create_page_actions(tmp_path: Path, caplog):
    """A model that emits create_page on the diff-only path must have it
    discarded — only apply_diff is a legal action on this path."""
    paths = _diff_only_env(tmp_path, _PAGE_CONTENT)
    rt = _FakeRuntime(_DIFF_ONLY_XML_WITH_CREATE_PAGE)
    agent = WikiAgent(runtime=rt, project_root=tmp_path)

    subtask = MagicMock()
    subtask.subtask_id  = "diff-test-2"
    subtask.instruction = "Update existing-page to reflect the new backend."
    subtask.context = {
        "diff_target":   "existing-page",
        "wiki_dir":      str(paths["wiki_dir"]),
        "schema_path":   str(paths["schema_path"]),
        "templates_dir": str(paths["templates_dir"]),
        "auto_apply":    True,
    }

    with caplog.at_level(logging.WARNING, logger="localist.wiki_agent"):
        result = agent.run(subtask)

    assert result.status == TaskStatus.COMPLETE
    assert result.output["new_pages"] == []
    assert len(result.output["diffs"]) == 1
    assert not (paths["wiki_dir"] / "should-not-exist.md").exists()
    assert any(
        "discarding" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_diff_only_run_fails_when_target_not_found(tmp_path: Path):
    paths = _diff_only_env(tmp_path, _PAGE_CONTENT)
    rt = _FakeRuntime(_DIFF_ONLY_XML)
    agent = WikiAgent(runtime=rt, project_root=tmp_path)

    subtask = MagicMock()
    subtask.subtask_id  = "diff-test-3"
    subtask.instruction = "Update nonexistent-page to reflect the new backend."
    subtask.context = {
        "diff_target":   "nonexistent-page",
        "wiki_dir":      str(paths["wiki_dir"]),
        "schema_path":   str(paths["schema_path"]),
        "templates_dir": str(paths["templates_dir"]),
        "auto_apply":    False,
    }

    result = agent.run(subtask)

    assert result.status == TaskStatus.FAILED
    assert "diff_target page not found" in result.error


def test_apply_pending_diff_success_writes_to_disk(tmp_path: Path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    page_path = wiki_dir / "existing-page.md"
    page_path.write_text(_PAGE_CONTENT, encoding="utf-8")

    agent = WikiAgent(runtime=_FakeRuntime(""), project_root=tmp_path)

    diff_text = (
        "@@ -3,1 +3,1 @@\n"
        "-Old summary.\n"
        "+New summary.\n"
    )
    result = agent.apply_pending_diff("existing-page", diff_text, wiki_dir)

    assert result.status == TaskStatus.COMPLETE
    assert result.output == {"page_name": "existing-page", "written": True}
    updated = page_path.read_text(encoding="utf-8")
    assert "New summary." in updated
    assert "Old summary." not in updated


def test_apply_pending_diff_reindexes_via_memory_manager(tmp_path: Path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    (wiki_dir / "existing-page.md").write_text(_PAGE_CONTENT, encoding="utf-8")

    mm = MagicMock()
    agent = WikiAgent(runtime=_FakeRuntime(""), project_root=tmp_path, memory_manager=mm)

    diff_text = "@@ -3,1 +3,1 @@\n-Old summary.\n+New summary.\n"
    result = agent.apply_pending_diff("existing-page", diff_text, wiki_dir)

    assert result.status == TaskStatus.COMPLETE
    mm.index_document.assert_called_once()
    _, kwargs = mm.index_document.call_args
    assert kwargs["path"] == wiki_dir / "existing-page.md"
    assert kwargs["doc_type"] == "wiki"


def test_apply_pending_diff_stale_content_fails_without_writing(tmp_path: Path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    page_path = wiki_dir / "existing-page.md"
    page_path.write_text(_PAGE_CONTENT, encoding="utf-8")

    agent = WikiAgent(runtime=_FakeRuntime(""), project_root=tmp_path)

    # Diff references content that no longer matches the file.
    diff_text = (
        "@@ -3,1 +3,1 @@\n"
        "-This text does not exist in the page.\n"
        "+New summary.\n"
    )
    result = agent.apply_pending_diff("existing-page", diff_text, wiki_dir)

    assert result.status == TaskStatus.FAILED
    assert result.output["error_kind"] == "stale"
    assert result.error is not None
    # File must be byte-for-byte unchanged.
    assert page_path.read_text(encoding="utf-8") == _PAGE_CONTENT


def test_apply_pending_diff_missing_page_fails(tmp_path: Path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()

    agent = WikiAgent(runtime=_FakeRuntime(""), project_root=tmp_path)
    result = agent.apply_pending_diff("nonexistent-page", "@@ -1,1 +1,1 @@\n-a\n+b\n", wiki_dir)

    assert result.status == TaskStatus.FAILED
    assert result.output["error_kind"] == "not_found"


def test_raw_path_wins_when_both_raw_path_and_diff_target_present(tmp_path: Path):
    """Both keys present → ingest path (raw_path) wins; diff_target is
    ignored rather than triggering a lookup that could fail."""
    paths = _diff_only_env(tmp_path, _PAGE_CONTENT)
    raw_path = tmp_path / "raw-input.md"
    raw_path.write_text("Some raw content.\n", encoding="utf-8")

    rt = _FakeRuntime(_RUN_XML)  # ingest-shaped XML from Test 8, above
    agent = WikiAgent(runtime=rt, project_root=tmp_path)

    subtask = MagicMock()
    subtask.subtask_id  = "diff-test-4"
    subtask.instruction = "ingest raw-input.md"
    subtask.context = {
        "raw_path":      str(raw_path),
        "diff_target":   "nonexistent-page",  # would fail if this path were taken
        "wiki_dir":      str(paths["wiki_dir"]),
        "schema_path":   str(paths["schema_path"]),
        "templates_dir": str(paths["templates_dir"]),
        "auto_apply":    False,
    }

    result = agent.run(subtask)

    assert result.status == TaskStatus.COMPLETE
    assert result.output["raw_filename"] == "raw-input.md"
    assert result.output["diff_target"]  is None


# ---------------------------------------------------------------------------
# apply_unified_diff() — content-based hunk matching
#
# Live finding (gemma4:31b-cloud, 2026-07-09,
# diagnostics/diag_wiki_agent_diff_only.py, auto_apply=True): the model's
# "@@ -21,..." header claimed line 21, but the actual -/+ line pair it
# emitted belonged at a different position in the file entirely — the model
# is unreliable at counting exact line numbers in a prompt-sized file, but
# reliably reproduces the real text being replaced. apply_unified_diff() now
# locates each hunk by content (_locate_hunk()), using orig_start only as a
# disambiguation hint.
# ---------------------------------------------------------------------------

def test_apply_unified_diff_ignores_wrong_line_number_in_header():
    """Regression test for the exact live failure shape: header claims the
    wrong line number; the real before-content is unique elsewhere in the
    file. Must apply at the correct (content-matched) position, not the
    hinted one."""
    original = (
        "# localist-software-stack\n"
        "\n"
        "## Summary\n"
        "\n"
        "**Core Software Stack:** Includes oMLX (Local inference server), "
        "MLX (Apple Silicon ML framework).\n"
        "- filler line 6\n"
        "- filler line 7\n"
        "- filler line 8\n"
        "- filler line 9\n"
        "- filler line 10\n"
        "- filler line 11\n"
        "- filler line 12\n"
        "- filler line 13\n"
        "- filler line 14\n"
        "- filler line 15\n"
        "- filler line 16\n"
        "- filler line 17\n"
        "- filler line 18\n"
        "- filler line 19\n"
        "- filler line 20\n"
        "**Cloud Models:** Azure AI Foundry is listed as an optional "
        "backend, accessible via FoundryRuntimeClient.\n"
    )
    assert original.splitlines(keepends=True)[20].startswith("**Cloud Models:**")

    # Header claims line 21 (the "Cloud Models" line) — wrong; the real
    # target ("Core Software Stack", line 5) is elsewhere in the file.
    diff_text = (
        "@@ -21,1 +21,1 @@\n"
        "-**Core Software Stack:** Includes oMLX (Local inference server), "
        "MLX (Apple Silicon ML framework).\n"
        "+**Core Software Stack:** Includes oMLX (Local inference server), "
        "Ollama (Local/Cloud runtime), MLX (Apple Silicon ML framework).\n"
    )

    updated = apply_unified_diff(original, diff_text)
    updated_lines = updated.splitlines(keepends=True)

    # Correct line (5, 0-indexed 4) was replaced...
    assert "Ollama (Local/Cloud runtime)" in updated_lines[4]
    # ...and the unrelated line the model's wrong header pointed at is untouched.
    assert updated_lines[20].startswith("**Cloud Models:**")
    assert len(updated_lines) == len(original.splitlines(keepends=True))


def test_apply_unified_diff_ambiguous_match_prefers_hinted_position():
    """Two identical lines in the file; orig_start disambiguates which one
    is meant — the match closest to the hinted line wins, not the first
    occurrence in file order."""
    original = (
        "unique line 1\n"
        "duplicate line\n"
        "unique line 3\n"
        "unique line 4\n"
        "unique line 5\n"
        "unique line 6\n"
        "unique line 7\n"
        "unique line 8\n"
        "duplicate line\n"
        "unique line 10\n"
    )
    diff_text = (
        "@@ -9,1 +9,1 @@\n"
        "-duplicate line\n"
        "+REPLACED line\n"
    )

    updated = apply_unified_diff(original, diff_text)
    updated_lines = updated.splitlines(keepends=True)

    # The line 2 occurrence (index 1) is untouched...
    assert updated_lines[1] == "duplicate line\n"
    # ...the line 9 occurrence (index 8, closest to the orig_start=9 hint) was replaced.
    assert updated_lines[8] == "REPLACED line\n"


def test_apply_unified_diff_recovers_from_bullet_marker_collision():
    """Regression test for the exact second live failure shape
    (diagnostics/diag_wiki_agent_diff_only.py, gemma4:31b-cloud, auto_apply=
    True, 2026-07-09): the model collapsed a removed bulleted line's "- "
    prefix into the diff's own "-" marker, e.g. "-**Core Software
    Stack:**..." instead of "-- **Core Software Stack:**...". Must recover
    and apply at the correct position rather than raising."""
    original = (
        "- **Core Software Stack:** Includes oMLX (Local inference server), "
        "MLX (Apple Silicon ML framework).\n"
        "- **Local Tools & Libraries:** Essential dependencies like SQLite.\n"
    )
    # Model dropped the bullet's "- " on both the removed and added line —
    # a single leading "-"/"+" instead of "--"/"+-".
    diff_text = (
        "@@ -1,1 +1,1 @@\n"
        "-**Core Software Stack:** Includes oMLX (Local inference server), "
        "MLX (Apple Silicon ML framework).\n"
        "+**Core Software Stack:** Includes oMLX (Local inference server), "
        "Ollama (Local/Cloud runtime), MLX (Apple Silicon ML framework).\n"
    )

    updated = apply_unified_diff(original, diff_text)
    updated_lines = updated.splitlines(keepends=True)

    assert updated_lines[0] == (
        "- **Core Software Stack:** Includes oMLX (Local inference server), "
        "Ollama (Local/Cloud runtime), MLX (Apple Silicon ML framework).\n"
    )
    # Unrelated second line untouched.
    assert updated_lines[1].startswith("- **Local Tools & Libraries:**")


def test_apply_unified_diff_recovers_from_context_bullet_marker_collision():
    """Regression test generalizing the 2026-07-09 bullet/diff-marker fix to
    the mirror-image collision confirmed live but left unfixed at the time
    (docs/architecture/17-wiki-agent-diff-target.md §17.7 point 3): an
    unchanged CONTEXT line that itself starts with a markdown bullet can
    have its leading " " marker dropped, reproducing the bullet's own "- "
    dash verbatim — which then reads exactly like a removed/added line
    whose marker char happens to be that same dash. Must recognize the
    dropped-context-marker shape and apply at the correct position rather
    than raising (previously a safe 409, not a corruption, but a block)."""
    original = (
        "- **A:** foo\n"
        "- **B:** bar\n"
        "- **C:** baz\n"
    )
    # Line 1 is unchanged CONTEXT but dropped its leading " " marker instead
    # of reading " - **A:** foo". Lines 2/3 are correctly-formatted
    # removed/added bulleted lines (proper double marker+bullet dash).
    diff_text = (
        "@@ -1,3 +1,3 @@\n"
        "- **A:** foo\n"
        "-- **B:** bar\n"
        "+- **B:** bar UPDATED\n"
        " - **C:** baz\n"
    )

    updated = apply_unified_diff(original, diff_text)
    updated_lines = updated.splitlines(keepends=True)

    assert updated_lines[0] == "- **A:** foo\n"
    assert updated_lines[1] == "- **B:** bar UPDATED\n"
    assert updated_lines[2] == "- **C:** baz\n"


def test_apply_unified_diff_recovers_from_both_bullet_collisions_in_one_hunk():
    """Both collision directions can occur in the same hunk: a genuinely
    removed/added bulleted line with its own "- " dropped (recover_bullet_
    marker), and a separate unchanged context bulleted line with its
    leading " " marker dropped (recover_context_bullet_marker). Neither
    single-hypothesis recovery resolves this alone; the combined-mode
    fallback must."""
    original = (
        "- **A:** foo\n"
        "- **B:** bar\n"
        "- **C:** baz\n"
    )
    diff_text = (
        "@@ -1,3 +1,3 @@\n"
        "- **A:** foo\n"          # context, dropped its " " marker
        "-**B:** bar\n"           # removed, dropped its own "- " bullet
        "+**B:** bar UPDATED\n"   # added, dropped its own "- " bullet
        " - **C:** baz\n"         # correctly-formatted context
    )

    updated = apply_unified_diff(original, diff_text)
    updated_lines = updated.splitlines(keepends=True)

    assert updated_lines[0] == "- **A:** foo\n"
    assert updated_lines[1] == "- **B:** bar UPDATED\n"
    assert updated_lines[2] == "- **C:** baz\n"


def test_apply_unified_diff_missing_trailing_newline_mid_file_does_not_merge_lines():
    """Regression test for the third live failure shape
    (POST /wiki/apply-diff, gemma4:31b-cloud, 2026-07-09): the model's
    last "+" line lacked a trailing newline even though it replaces a
    line nowhere near the end of the file. Live symptom: the replacement
    line got glued directly onto the following heading with no line
    break — "...usage-based.### Mapped Pages". Must insert the missing
    newline rather than merging the two lines."""
    original = (
        "- **Cost Structure:** All software is Free.\n"
        "\n"
        "### Mapped Pages\n"
        "\n"
        "- null\n"
    )
    # Note: the "+" line deliberately has NO trailing newline, mirroring the
    # model's real (buggy) output — it was the last line of its diff block.
    diff_text = (
        "@@ -1,1 +1,1 @@\n"
        "-- **Cost Structure:** All software is Free.\n"
        "+- **Cost Structure:** All software is Free, Ollama Cloud is usage-based."
    )

    updated = apply_unified_diff(original, diff_text)
    updated_lines = updated.splitlines(keepends=True)

    assert updated_lines[0] == "- **Cost Structure:** All software is Free, Ollama Cloud is usage-based.\n"
    # The blank line + heading must survive as their own lines, not merged.
    assert updated_lines[1] == "\n"
    assert updated_lines[2] == "### Mapped Pages\n"
    assert "### Mapped Pages" not in updated_lines[0]


def test_apply_unified_diff_genuine_mismatch_raises():
    """before-content that doesn't appear anywhere in the file is a real
    content mismatch, not a position problem — must still raise ValueError."""
    original = "line1\nline2\nline3\n"
    diff_text = (
        "@@ -2,1 +2,1 @@\n"
        "-this text does not exist in the file\n"
        "+new line\n"
    )

    with pytest.raises(ValueError, match="not found anywhere in the file"):
        apply_unified_diff(original, diff_text)


# ---------------------------------------------------------------------------
# §17.8 — pre-write snapshot safety net
# ---------------------------------------------------------------------------

import os
import time


def test_snapshot_page_creates_versioned_file_with_correct_content(tmp_path: Path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    (wiki_dir / "existing-page.md").write_text(_PAGE_CONTENT, encoding="utf-8")

    WikiAgent._snapshot_page("existing-page", wiki_dir)

    snapshots = list((wiki_dir / ".snapshots").glob("existing-page.v*.md"))
    assert len(snapshots) == 1
    assert snapshots[0].name.startswith("existing-page.v1.")
    assert snapshots[0].name.endswith(".md")
    assert snapshots[0].read_text(encoding="utf-8") == _PAGE_CONTENT


def test_snapshot_page_increments_version_number(tmp_path: Path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    page_path = wiki_dir / "existing-page.md"
    page_path.write_text("v1 content", encoding="utf-8")

    WikiAgent._snapshot_page("existing-page", wiki_dir)
    page_path.write_text("v2 content", encoding="utf-8")
    WikiAgent._snapshot_page("existing-page", wiki_dir)

    snapshots = sorted((wiki_dir / ".snapshots").glob("existing-page.v*.md"))
    assert len(snapshots) == 2
    assert snapshots[0].name.startswith("existing-page.v1.")
    assert snapshots[1].name.startswith("existing-page.v2.")
    assert snapshots[0].read_text(encoding="utf-8") == "v1 content"
    assert snapshots[1].read_text(encoding="utf-8") == "v2 content"


def test_snapshot_page_versioning_is_per_page_name(tmp_path: Path):
    """A second, differently-named page's snapshots must not affect this
    page's version counter."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    (wiki_dir / "page-a.md").write_text("a", encoding="utf-8")
    (wiki_dir / "page-b.md").write_text("b", encoding="utf-8")

    WikiAgent._snapshot_page("page-a", wiki_dir)
    WikiAgent._snapshot_page("page-b", wiki_dir)

    snapshots_dir = wiki_dir / ".snapshots"
    assert list(snapshots_dir.glob("page-a.v*.md"))[0].name.startswith("page-a.v1.")
    assert list(snapshots_dir.glob("page-b.v*.md"))[0].name.startswith("page-b.v1.")


def test_snapshot_page_non_fatal_when_source_page_missing(tmp_path: Path, caplog):
    """_snapshot_page() must not raise even if the page it's snapshotting
    doesn't exist on disk — matches the non-fatal pattern used elsewhere in
    this module (e.g. _write_journal())."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()

    with caplog.at_level(logging.WARNING, logger="localist.wiki_agent"):
        WikiAgent._snapshot_page("nonexistent-page", wiki_dir)

    assert "Snapshot failed" in caplog.text
    assert not (wiki_dir / ".snapshots").exists()


def test_apply_changes_snapshots_before_patching_existing_page(tmp_path: Path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    page_path = wiki_dir / "existing-page.md"
    page_path.write_text(_PAGE_CONTENT, encoding="utf-8")

    actions = Actions(
        new_pages=[],
        diffs=[ApplyDiff(
            page_name="existing-page",
            diff="@@ -3,1 +3,1 @@\n-Old summary.\n+New summary.\n",
        )],
    )

    applied, written, skipped, diff_errors = WikiAgent._apply_changes(actions, wiki_dir)

    assert applied is True
    assert written == ["existing-page"]
    snapshots = list((wiki_dir / ".snapshots").glob("existing-page.v*.md"))
    assert len(snapshots) == 1
    # The snapshot holds the pre-patch content, not the patched result.
    assert snapshots[0].read_text(encoding="utf-8") == _PAGE_CONTENT
    assert page_path.read_text(encoding="utf-8") != _PAGE_CONTENT


def test_apply_pending_diff_snapshots_before_patching(tmp_path: Path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    page_path = wiki_dir / "existing-page.md"
    page_path.write_text(_PAGE_CONTENT, encoding="utf-8")

    agent = WikiAgent(runtime=_FakeRuntime(""), project_root=tmp_path)
    diff_text = "@@ -3,1 +3,1 @@\n-Old summary.\n+New summary.\n"

    result = agent.apply_pending_diff("existing-page", diff_text, wiki_dir)

    assert result.status == TaskStatus.COMPLETE
    snapshots = list((wiki_dir / ".snapshots").glob("existing-page.v*.md"))
    assert len(snapshots) == 1
    assert snapshots[0].read_text(encoding="utf-8") == _PAGE_CONTENT


def test_prune_page_snapshots_removes_only_expired_files(tmp_path: Path):
    wiki_dir = tmp_path / "wiki"
    snapshots_dir = wiki_dir / ".snapshots"
    snapshots_dir.mkdir(parents=True)

    old = snapshots_dir / "existing-page.v1.20200101T000000.md"
    old.write_text("old", encoding="utf-8")
    recent = snapshots_dir / "existing-page.v2.20260101T000000.md"
    recent.write_text("recent", encoding="utf-8")

    thirty_one_days_ago = time.time() - (31 * 24 * 60 * 60)
    os.utime(old, (thirty_one_days_ago, thirty_one_days_ago))

    WikiAgent._prune_page_snapshots("existing-page", wiki_dir)

    remaining = list(snapshots_dir.glob("existing-page.v*.md"))
    assert remaining == [recent]


def test_prune_page_snapshots_writes_wiki_maintenance_log_entry(tmp_path: Path):
    """_prune_page_snapshots() (the prune-on-write path) must land in the
    same audit trail as sweep_expired_snapshots() (the startup path) — see
    §17.8 audit-log parity. Format matches
    test_log_snapshot_pruned_writes_expected_line in test_memory_phase1.py."""
    wiki_dir = tmp_path / "wiki"
    snapshots_dir = wiki_dir / ".snapshots"
    snapshots_dir.mkdir(parents=True)

    old = snapshots_dir / "existing-page.v1.20200101T000000.md"
    old.write_text("old", encoding="utf-8")
    old_path = str(old)

    thirty_one_days_ago = time.time() - (31 * 24 * 60 * 60)
    os.utime(old, (thirty_one_days_ago, thirty_one_days_ago))

    WikiAgent._prune_page_snapshots("existing-page", wiki_dir)

    log_path = wiki_maintenance_log._LOG_PATH
    assert log_path.exists()
    log_line = log_path.read_text(encoding="utf-8").strip()
    assert "snapshot_pruned" in log_line
    assert "name=existing-page.v1.20200101T000000.md" in log_line
    assert f"path={old_path}" in log_line


def test_prune_page_snapshots_boundary_kept_at_exactly_30_days(tmp_path: Path):
    """A snapshot just inside the 30-day window is kept — the prune
    condition is strictly-older-than, not older-than-or-equal."""
    wiki_dir = tmp_path / "wiki"
    snapshots_dir = wiki_dir / ".snapshots"
    snapshots_dir.mkdir(parents=True)

    boundary = snapshots_dir / "existing-page.v1.20260101T000000.md"
    boundary.write_text("boundary", encoding="utf-8")

    just_inside_window = time.time() - (30 * 24 * 60 * 60) + 5
    os.utime(boundary, (just_inside_window, just_inside_window))

    WikiAgent._prune_page_snapshots("existing-page", wiki_dir)

    assert boundary.exists()


def test_prune_page_snapshots_only_touches_its_own_page(tmp_path: Path):
    wiki_dir = tmp_path / "wiki"
    snapshots_dir = wiki_dir / ".snapshots"
    snapshots_dir.mkdir(parents=True)

    other_old = snapshots_dir / "other-page.v1.20200101T000000.md"
    other_old.write_text("other", encoding="utf-8")
    thirty_one_days_ago = time.time() - (31 * 24 * 60 * 60)
    os.utime(other_old, (thirty_one_days_ago, thirty_one_days_ago))

    WikiAgent._prune_page_snapshots("existing-page", wiki_dir)

    assert other_old.exists()


def test_sweep_expired_snapshots_removes_across_all_pages(tmp_path: Path):
    wiki_dir = tmp_path / "wiki"
    snapshots_dir = wiki_dir / ".snapshots"
    snapshots_dir.mkdir(parents=True)

    old_a = snapshots_dir / "page-a.v1.20200101T000000.md"
    old_a.write_text("a", encoding="utf-8")
    old_b = snapshots_dir / "page-b.v1.20200101T000000.md"
    old_b.write_text("b", encoding="utf-8")
    recent = snapshots_dir / "page-a.v2.20260101T000000.md"
    recent.write_text("recent", encoding="utf-8")

    thirty_one_days_ago = time.time() - (31 * 24 * 60 * 60)
    os.utime(old_a, (thirty_one_days_ago, thirty_one_days_ago))
    os.utime(old_b, (thirty_one_days_ago, thirty_one_days_ago))

    pruned = sweep_expired_snapshots(wiki_dir)

    assert sorted(p.name for p in pruned) == [
        "page-a.v1.20200101T000000.md",
        "page-b.v1.20200101T000000.md",
    ]
    assert not old_a.exists()
    assert not old_b.exists()
    assert recent.exists()


def test_sweep_expired_snapshots_no_snapshots_dir_returns_empty(tmp_path: Path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()

    assert sweep_expired_snapshots(wiki_dir) == []


# ---------------------------------------------------------------------------
# §17.8 follow-up — success logging + env-var-overridable TTL
# ---------------------------------------------------------------------------

def test_snapshot_page_logs_success_at_info_level(tmp_path: Path, caplog):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    (wiki_dir / "existing-page.md").write_text(_PAGE_CONTENT, encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="localist.wiki_agent"):
        WikiAgent._snapshot_page("existing-page", wiki_dir)

    assert "Snapshotted 'existing-page'" in caplog.text
    snapshot_path = next((wiki_dir / ".snapshots").glob("existing-page.v*.md"))
    assert str(snapshot_path) in caplog.text


def test_prune_page_snapshots_respects_env_var_ttl_override(tmp_path: Path, monkeypatch):
    """With the TTL overridden down to a few seconds, a snapshot older than
    that (but nowhere near the real 30-day default) must be pruned."""
    monkeypatch.setenv("LOCALIST_WIKI_SNAPSHOT_TTL_SECONDS", "5")

    wiki_dir = tmp_path / "wiki"
    snapshots_dir = wiki_dir / ".snapshots"
    snapshots_dir.mkdir(parents=True)

    old = snapshots_dir / "existing-page.v1.20260101T000000.md"
    old.write_text("old", encoding="utf-8")
    recent = snapshots_dir / "existing-page.v2.20260101T000001.md"
    recent.write_text("recent", encoding="utf-8")

    ten_seconds_ago = time.time() - 10
    os.utime(old, (ten_seconds_ago, ten_seconds_ago))

    WikiAgent._prune_page_snapshots("existing-page", wiki_dir)

    remaining = list(snapshots_dir.glob("existing-page.v*.md"))
    assert remaining == [recent]


def test_sweep_expired_snapshots_respects_env_var_ttl_override(tmp_path: Path, monkeypatch):
    """Same override, exercised through the startup-sweep entry point."""
    monkeypatch.setenv("LOCALIST_WIKI_SNAPSHOT_TTL_SECONDS", "5")

    wiki_dir = tmp_path / "wiki"
    snapshots_dir = wiki_dir / ".snapshots"
    snapshots_dir.mkdir(parents=True)

    old = snapshots_dir / "page-a.v1.20260101T000000.md"
    old.write_text("old", encoding="utf-8")
    recent = snapshots_dir / "page-b.v1.20260101T000001.md"
    recent.write_text("recent", encoding="utf-8")

    ten_seconds_ago = time.time() - 10
    os.utime(old, (ten_seconds_ago, ten_seconds_ago))

    pruned = sweep_expired_snapshots(wiki_dir)

    assert [p.name for p in pruned] == ["page-a.v1.20260101T000000.md"]
    assert not old.exists()
    assert recent.exists()


# ---------------------------------------------------------------------------
# OKF alignment (§18) — META_WIKI_FILENAMES exclusion, index.md/logs.md
# generation, and end-to-end wiring into _finalize()
# ---------------------------------------------------------------------------

_REAL_PAGE = """\
---
type: RESEARCH_NOTE
title: Real Page
description: A real content page.
---

## Summary

Content.
"""


def test_load_wiki_pages_excludes_meta_filenames(tmp_path: Path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    (wiki_dir / "real-page.md").write_text(_REAL_PAGE, encoding="utf-8")
    (wiki_dir / "index.md").write_text("# Wiki Index\n", encoding="utf-8")
    (wiki_dir / "logs.md").write_text("# Wiki Changelog\n", encoding="utf-8")
    (wiki_dir / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")

    pages = WikiAgent._load_wiki_pages(wiki_dir)

    assert list(pages.keys()) == ["real-page"]


class TestRegenerateIndexMd:

    def test_groups_by_type_and_includes_title_description(self, tmp_path: Path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        pages = {
            "concept-a": "---\ntype: CONCEPT\ntitle: Concept A\ndescription: About A.\n---\n\nbody",
            "note-b":    "---\ntype: RESEARCH_NOTE\ntitle: Note B\n---\n\nbody",
        }

        WikiAgent._regenerate_index_md(pages, wiki_dir)

        out = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "## CONCEPT" in out
        assert "## RESEARCH_NOTE" in out
        assert "[[concept-a]] — Concept A: About A." in out
        assert "[[note-b]] — Note B" in out
        assert out.index("## CONCEPT") < out.index("## RESEARCH_NOTE")  # alphabetical grouping

    def test_missing_frontmatter_falls_back_to_stem_no_crash(self, tmp_path: Path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        pages = {"no-frontmatter-page": "Just a plain body, no front matter at all.\n"}

        WikiAgent._regenerate_index_md(pages, wiki_dir)

        out = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "## UNSPECIFIED" in out
        assert "[[no-frontmatter-page]] — no-frontmatter-page" in out

    def test_regeneration_overwrites_previous_index(self, tmp_path: Path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        WikiAgent._regenerate_index_md({"page-a": _REAL_PAGE}, wiki_dir)
        WikiAgent._regenerate_index_md({"page-b": _REAL_PAGE}, wiki_dir)

        out = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "page-a" not in out
        assert "page-b" in out


class TestAppendLogsMd:

    def test_creates_file_with_header_on_first_write(self, tmp_path: Path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()

        WikiAgent._append_logs_md([("Creation", "new-page")], wiki_dir)

        out = (wiki_dir / "logs.md").read_text(encoding="utf-8")
        assert out.startswith("# Wiki Changelog")
        assert "- Creation: [[new-page]]" in out

    def test_second_call_appends_not_overwrites(self, tmp_path: Path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()

        WikiAgent._append_logs_md([("Creation", "page-one")], wiki_dir)
        WikiAgent._append_logs_md([("Update", "page-two")], wiki_dir)

        out = (wiki_dir / "logs.md").read_text(encoding="utf-8")
        assert "- Creation: [[page-one]]" in out
        assert "- Update: [[page-two]]" in out

    def test_empty_entries_is_a_no_op(self, tmp_path: Path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()

        WikiAgent._append_logs_md([], wiki_dir)

        assert not (wiki_dir / "logs.md").exists()


def test_end_to_end_diff_only_run_regenerates_index_and_logs(tmp_path: Path):
    """auto_apply=True on the diff-only path must regenerate index.md and
    append logs.md, in addition to the existing disk-write/reindex/graph
    behavior already covered by test_diff_only_run_applies_diff_to_disk_when_auto_apply."""
    paths = _diff_only_env(tmp_path, _PAGE_CONTENT)
    (paths["wiki_dir"] / "existing-page.md").write_text(
        "---\ntype: RESEARCH_NOTE\ntitle: Existing Page\n---\n\n" + _PAGE_CONTENT,
        encoding="utf-8",
    )
    rt = _FakeRuntime(_DIFF_ONLY_XML)
    agent = WikiAgent(runtime=rt, project_root=tmp_path)

    subtask = MagicMock()
    subtask.subtask_id  = "diff-test-okf"
    subtask.instruction = "Update existing-page to reflect the new backend."
    subtask.context = {
        "diff_target":   "existing-page",
        "wiki_dir":      str(paths["wiki_dir"]),
        "schema_path":   str(paths["schema_path"]),
        "templates_dir": str(paths["templates_dir"]),
        "auto_apply":    True,
    }

    result = agent.run(subtask)

    assert result.status == TaskStatus.COMPLETE
    index_out = (paths["wiki_dir"] / "index.md").read_text(encoding="utf-8")
    assert "[[existing-page]]" in index_out
    logs_out = (paths["wiki_dir"] / "logs.md").read_text(encoding="utf-8")
    assert "- Update: [[existing-page]]" in logs_out


# ---------------------------------------------------------------------------
# OCR raw_path ingest routing (docs/architecture/22-local-ocr-service.md
# §22.8 follow-up: OCR-eligible image/PDF raw_path files are now extracted
# via extract_raw_content_via_ocr() instead of read_text_file(), and always
# take the infer() string-prompt path — never infer_with_file()/MarkItDown —
# so ingestion behaves identically across oMLX/Ollama/Foundry.)
# ---------------------------------------------------------------------------

from localist import wiki_agent as wiki_agent_module
from localist.prompt_builder import ToolResult


class _RuntimeWithInferWithFile(_CapturingFakeRuntime):
    """Fake runtime that DOES expose infer_with_file, unlike _FakeRuntime —
    proves OCR routing wins over hasattr(runtime, 'infer_with_file') even
    when the runtime looks like oMLX."""

    def infer_with_file(self, *args, **kwargs) -> str:
        raise AssertionError(
            "infer_with_file must not be called for an OCR-eligible raw_path"
        )


def _make_ocr_ingest_subtask(tmp_path: Path, raw_path: Path) -> MagicMock:
    wiki_dir      = tmp_path / "wiki"
    wiki_dir.mkdir(exist_ok=True)
    schema_path   = tmp_path / "SCHEMA.md"
    schema_path.write_text("# Schema\n", encoding="utf-8")
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir(exist_ok=True)

    subtask = MagicMock()
    subtask.subtask_id  = "ocr-test-0"
    subtask.instruction = "ingest this file"
    subtask.context = {
        "raw_path":      str(raw_path),
        "wiki_dir":      str(wiki_dir),
        "schema_path":   str(schema_path),
        "templates_dir": str(templates_dir),
        "auto_apply":    False,
    }
    return subtask


_OCR_RUN_XML = """\
<actions>
  <action name="create_page">
    <page_name>ocr-ingested-page</page_name>
    <page_type>RESEARCH_NOTE</page_type>
    <content>
## Summary

OCR ingested.

## Details

### Extracted Concepts

- A concept.

### Mapped Pages

- null

### Proposed New Pages

- null

## Related Pages

## Revision History

- 2026-08-03 — Created.
    </content>
  </action>
</actions>
"""


class TestOcrRawPathIngest:
    """WikiAgent.run() routes OCR-eligible (image/PDF) raw_path files through
    extract_raw_content_via_ocr() instead of read_text_file()/infer_with_file()."""

    def test_ocr_eligible_raw_path_routes_through_ocr_no_infer_with_file(
        self, tmp_path: Path, monkeypatch,
    ):
        # Real, non-UTF-8 bytes — if this regressed to read_text_file(), it
        # would raise UnicodeDecodeError and fail the test before OCR ever ran.
        raw_path = tmp_path / "scan.png"
        raw_path.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xd8\xff\x00not-utf8\xfe")

        subtask = _make_ocr_ingest_subtask(tmp_path, raw_path)

        rt = _CapturingFakeRuntime(_OCR_RUN_XML)
        assert not hasattr(rt, "infer_with_file")

        extracted_calls = []

        def _fake_extract(runtime, path, mime_type):
            extracted_calls.append((runtime, path, mime_type))
            return "OCR'd text: hello world"

        monkeypatch.setattr(wiki_agent_module, "extract_raw_content_via_ocr", _fake_extract)

        agent = WikiAgent(runtime=rt, project_root=tmp_path)
        result = agent.run(subtask)

        assert result.status == TaskStatus.COMPLETE
        assert extracted_calls == [(rt, raw_path, "image/png")]
        assert "OCR'd text: hello world" in rt.captured_prompt

    def test_ocr_routing_wins_even_when_runtime_has_infer_with_file(
        self, tmp_path: Path, monkeypatch,
    ):
        raw_path = tmp_path / "scan.pdf"
        raw_path.write_bytes(b"%PDF-1.4\xff\xfe not real pdf bytes")

        subtask = _make_ocr_ingest_subtask(tmp_path, raw_path)

        rt = _RuntimeWithInferWithFile(_OCR_RUN_XML)
        assert hasattr(rt, "infer_with_file")

        monkeypatch.setattr(
            wiki_agent_module, "extract_raw_content_via_ocr",
            lambda runtime, path, mime_type: "extracted pdf text",
        )

        agent = WikiAgent(runtime=rt, project_root=tmp_path)
        result = agent.run(subtask)

        assert result.status == TaskStatus.COMPLETE
        assert "extracted pdf text" in rt.captured_prompt

    def test_ocr_extraction_failure_returns_fail_result(
        self, tmp_path: Path, monkeypatch,
    ):
        raw_path = tmp_path / "blank-scan.png"
        raw_path.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")

        subtask = _make_ocr_ingest_subtask(tmp_path, raw_path)

        rt = _FakeRuntime(_OCR_RUN_XML)

        def _fake_extract(runtime, path, mime_type):
            raise RuntimeError("no readable text detected")

        monkeypatch.setattr(wiki_agent_module, "extract_raw_content_via_ocr", _fake_extract)

        agent = WikiAgent(runtime=rt, project_root=tmp_path)
        result = agent.run(subtask)

        assert result.status == TaskStatus.FAILED
        assert "OCR extraction error" in result.error
        assert "no readable text detected" in result.error


class TestExtractRawContentViaOcr:
    """Unit tests for extract_raw_content_via_ocr() itself — temp-file
    lifecycle and MCPToolDispatcher wiring, independent of WikiAgent.run()."""

    def test_success_writes_temp_file_dispatches_and_cleans_up(
        self, tmp_path: Path, monkeypatch,
    ):
        upload_root = tmp_path / "chat_uploads"
        upload_root.mkdir()
        monkeypatch.setattr(wiki_agent_module, "_ocr_get_upload_root", lambda: upload_root)

        dispatch_calls = []

        class _FakeDispatcher:
            def __init__(self, runtime=None, **kwargs):
                self._runtime = runtime

            def dispatch(self, tools_to_call, instruction, context=None):
                dispatch_calls.append((tools_to_call, instruction, context))
                tmp_file = upload_root / context["ocr_file_path"]
                assert tmp_file.exists(), "temp file must exist at dispatch time"
                return [ToolResult(
                    tool_name="ocr_extract", parameters="", result="extracted!",
                    success=True,
                )]

        monkeypatch.setattr(wiki_agent_module, "MCPToolDispatcher", _FakeDispatcher)

        raw_path = tmp_path / "photo.jpg"
        raw_path.write_bytes(b"\xff\xd8\xff not real jpeg bytes")

        rt = object()
        out = wiki_agent_module.extract_raw_content_via_ocr(rt, raw_path, "image/jpeg")

        assert out == "extracted!"
        assert len(dispatch_calls) == 1
        tools_to_call, instruction, context = dispatch_calls[0]
        assert tools_to_call == ["ocr_extract"]
        assert context["ocr_mime_type"] == "image/jpeg"
        # Temp file removed after dispatch, regardless of outcome.
        assert list(upload_root.iterdir()) == []

    def test_failure_still_cleans_up_temp_file_and_raises(
        self, tmp_path: Path, monkeypatch,
    ):
        upload_root = tmp_path / "chat_uploads"
        upload_root.mkdir()
        monkeypatch.setattr(wiki_agent_module, "_ocr_get_upload_root", lambda: upload_root)

        class _FakeDispatcher:
            def __init__(self, runtime=None, **kwargs):
                pass

            def dispatch(self, tools_to_call, instruction, context=None):
                return [ToolResult(
                    tool_name="ocr_extract", parameters="", success=False,
                    result="ERROR: no readable text detected",
                )]

        monkeypatch.setattr(wiki_agent_module, "MCPToolDispatcher", _FakeDispatcher)

        raw_path = tmp_path / "blank.png"
        raw_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        with pytest.raises(RuntimeError, match="no readable text detected"):
            wiki_agent_module.extract_raw_content_via_ocr(object(), raw_path, "image/png")

        assert list(upload_root.iterdir()) == []


class TestResolveRawPathOcrExtensions:
    """_resolve_raw_path() accepts OCR-eligible extensions without requiring
    valid UTF-8 text, while non-OCR extensions keep the existing gate."""

    def test_accepts_binary_png(self, tmp_path: Path):
        p = tmp_path / "scan.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe not utf-8 at all")
        resolved = WikiAgent._resolve_raw_path({"raw_path": str(p)})
        assert resolved == p

    def test_accepts_binary_pdf(self, tmp_path: Path):
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4\xff\xfe binary")
        resolved = WikiAgent._resolve_raw_path({"raw_path": str(p)})
        assert resolved == p

    def test_still_rejects_unsupported_extension(self, tmp_path: Path):
        p = tmp_path / "archive.zip"
        p.write_bytes(b"PK\x03\x04")
        with pytest.raises(ValueError, match=r"raw_path must be a \.md, \.txt, or OCR-eligible"):
            WikiAgent._resolve_raw_path({"raw_path": str(p)})

    def test_still_rejects_non_utf8_txt(self, tmp_path: Path):
        p = tmp_path / "notes.txt"
        p.write_bytes(b"\xff\xfe not valid utf-8")
        with pytest.raises(ValueError, match="does not appear to be a UTF-8 text file"):
            WikiAgent._resolve_raw_path({"raw_path": str(p)})
