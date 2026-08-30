"""
backend/json_envelope_repair.py
==================================
Promoted from diagnostics/json_envelope_repair.py — this is the production
copy MCPToolDispatcher._run_chart() imports (see mcp_tool_dispatcher.py).
The diagnostics/ copy is left in place for reproducibility of the measured
numbers but is not imported from here or from any other production path.

Bracket-balanced repair pass for the specific {"tool_call": ...} JSON
envelope corruption pattern observed in diag_shadow_chart_toolcall.py's
first run against gemma-4-e4b-it-4bit (4-bit quantized, oMLX): an
otherwise well-formed JSON object with a short run of stray tokens
(observed: "だろう", "的比") inserted mid-structure — after a nested
array/object closes but before the enclosing braces resume — which
breaks json.loads() even though the JSON is fully "there."

This is NOT a general JSON repair library. It targets exactly one
corruption shape: a run of non-structural characters sitting where the
parser expects a structural delimiter ('}', ']', ','). The repair
strategy uses json.JSONDecodeError.pos (the exact character offset where
the standard library's parser gave up) to search a short window ahead
for a delimiter whose excision makes the WHOLE string parse successfully.

Safety property: a candidate excision is only ever accepted if the
resulting string parses as valid JSON in full. The naive version of this
("cut at the first delimiter-looking character after the error") is not
safe on its own — a stray token can itself contain a delimiter-looking
character (e.g. a decoy '}') that isn't the one that actually closes the
right scope, and cutting there silently produces a *different*, wrong
but still-parseable structure (e.g. one bracket level short). That would
be worse than not repairing at all, since it would corrupt the
downstream MATCH count with an object that doesn't reflect anything the
model actually said. So every candidate delimiter position in the window
is tried, in order, and the first one that yields a full successful
json.loads() is accepted — never the first one that merely looks
plausible.

Deliberately NOT repaired (by design, not by omission):
  - Genuine truncation (output cut off mid-array/mid-string before any
    closing delimiter appears at all) — reported as "truncated" without
    fabricating a fix.
  - Plain prose with no JSON structure at all (e.g. the model answered
    the question directly instead of emitting a tool_call envelope) —
    reported as "not_json". This failure mode belongs to the prompt
    (see chart_tool_schema.py's SYSTEM_PROMPT_FEWSHOT), not the parser.
  - Corruption where no candidate excision in the window yields a fully
    valid parse — reported as "unrepairable" rather than guessed at.

repair_envelope() never raises — any failure to recover a valid object
is reported via the outcome string, never an exception.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

# How far past a JSONDecodeError's failure position to search for a
# delimiter whose excision fixes the parse. Observed corruption is a
# handful of characters; a genuinely different kind of malformed output
# should not be silently "fixed" by scanning arbitrarily far ahead.
_MAX_EXCISION_WINDOW = 40
_MAX_REPAIR_ATTEMPTS = 3

_CANDIDATE_DELIMS = ("}", "]", ",")


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _try_single_excision(working: str, pos: int) -> str | None:
    """
    Search working[pos : pos + _MAX_EXCISION_WINDOW] for a delimiter
    whose excision (removing everything from pos up to, but not
    including, the delimiter) makes the whole string parse successfully.
    Returns the repaired string, or None if no candidate in the window
    works.
    """
    window = working[pos:pos + _MAX_EXCISION_WINDOW]
    for i, ch in enumerate(window):
        if ch not in _CANDIDATE_DELIMS:
            continue
        candidate = working[:pos] + working[pos + i:]
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return candidate
    return None


def repair_envelope(raw: str) -> tuple[object | None, str]:
    """
    Attempt to parse `raw` as the {"tool_call": ...} envelope, repairing
    the specific "stray token mid-structure" corruption if a direct parse
    fails.

    Returns (parsed_value_or_None, outcome), where outcome is one of:
      "no_repair_needed"        — direct parse succeeded, nothing to do.
      "trailing_garbage_removed" — direct parse failed; excising a short
                                   run of characters at (or spanning) the
                                   decoder's failure point let the WHOLE
                                   string parse successfully.
      "truncated"                — parse failed, the text looks like an
                                   in-progress JSON object (contains an
                                   unmatched '{'), but no excision in the
                                   window produced a valid parse —
                                   consistent with output cut off rather
                                   than corrupted.
      "not_json"                 — parse failed and the text contains no
                                   '{' at all — the model didn't attempt
                                   a JSON envelope (e.g. answered in
                                   prose).
      "unrepairable"              — contains '{' and isn't a truncation
                                   pattern, but no excision in the window
                                   recovered a valid parse either.

    parsed_value_or_None is whatever json.loads() returns — typically a
    dict, but callers should still validate shape before trusting it
    (same contract as raw json.loads()).
    """
    cleaned = _strip_code_fences(raw)

    working = cleaned
    for attempt in range(_MAX_REPAIR_ATTEMPTS + 1):
        try:
            obj = json.loads(working)
            outcome = "no_repair_needed" if attempt == 0 else "trailing_garbage_removed"
            return obj, outcome
        except json.JSONDecodeError as exc:
            if attempt == _MAX_REPAIR_ATTEMPTS:
                break
            repaired = _try_single_excision(working, exc.pos)
            if repaired is None:
                break
            working = repaired

    if "{" in cleaned:
        # Distinguish "cut off before any recoverable delimiter" from
        # "has braces but excision genuinely couldn't fix it" — both are
        # left unrepaired, but the former is the truncation case the
        # module docstring commits to never fabricating a fix for.
        try:
            json.loads(cleaned)
        except json.JSONDecodeError as exc:
            tail = cleaned[exc.pos:exc.pos + _MAX_EXCISION_WINDOW]
            if not any(ch in _CANDIDATE_DELIMS for ch in tail):
                return None, "truncated"
        return None, "unrepairable"

    return None, "not_json"
