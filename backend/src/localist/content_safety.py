"""
LORA — Content safety scanner for episodic memory writes
===========================================================
Lightweight pattern-based scanner for episodic memory writes.

Not a full prompt-injection classifier — a bounded set of regex/string
checks against known-bad patterns, mirroring the approach documented at
https://hermes-agent.nousresearch.com/docs/user-guide/features/memory
(threat-pattern matching + invisible-Unicode detection, not an LLM call).
Runs synchronously in the write path — must stay fast (no I/O, no inference).

Why this exists
----------------
model_extracted episodes are written from a single, unreviewed inference
call over raw user/agent turn text, then replayed back into every future
[EPISODIC MEMORY] prompt block via by_recency()/by_similarity() and
rendered into MEMORY.md. Nothing upstream of EpisodicMemoryWriter.insert()
currently checks what that extraction call's output actually contains
before it starts being replayed forward — this module is that check.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Pattern tables — module-level constants so the threat list can be extended
# without touching scan_content() or any caller.
# ---------------------------------------------------------------------------

# Phrases that look like an attempt to break out of the `content` field into
# a structural prompt position (role markers, instruction overrides).
# Matched case-insensitively against the whole text.
_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"new\s+instructions\s*:", re.IGNORECASE),
    re.compile(r"^\s*system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*assistant\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"#{1,3}\s*system\b", re.IGNORECASE),
)

# Credential/secret-looking patterns. Prefix checks are intentionally
# specific (real provider token formats); the high-entropy run is a simple
# heuristic, not real entropy scoring — see module docstring.
_CREDENTIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),                 # OpenAI-style secret key
    re.compile(r"\bAKIA[A-Z0-9]{12,}\b"),                    # AWS access key id
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),                 # GitHub personal access token
    re.compile(r"\bxox[bp]-[A-Za-z0-9-]{10,}\b"),            # Slack bot/user token
    re.compile(r"-----BEGIN[ A-Z]*PRIVATE KEY-----"),        # PEM private key header
    re.compile(r"\bssh-(rsa|ed25519)\s+[A-Za-z0-9+/]{20,}"), # SSH public key material
    # Long unbroken run of base64/hex-alphabet characters — a coarse
    # heuristic for embedded secrets/tokens, not real entropy scoring.
    re.compile(r"[A-Za-z0-9+/]{32,}"),
)

def _scan_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PROMPT_INJECTION_PATTERNS)


def _scan_credential_exfil(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CREDENTIAL_PATTERNS)


def _scan_invisible_unicode(text: str) -> bool:
    # Ordinary whitespace (space, tab, newline) is category Zs/Cc, not Cf,
    # so it's never affected by this check — only zero-width/format/
    # bidi-override characters (category "Cf") trip it.
    return any(unicodedata.category(ch) == "Cf" for ch in text)


def scan_content(text: str) -> str | None:
    """
    Scan `text` for known threat patterns.

    Checked in order (first match wins): prompt injection, credential
    exfiltration, invisible/control Unicode. Order reflects severity for
    logging purposes only — a text could in principle match more than one
    category, but only the first matched category is reported.

    Parameters
    ----------
    text :
        Free-text content to scan (an episode's `subject` or `content`).

    Returns
    -------
    str | None
        One of "prompt_injection", "credential_exfil",
        "invisible_unicode" if `text` matches a known threat pattern,
        else None if clean.
    """
    if not text:
        return None
    if _scan_prompt_injection(text):
        return "prompt_injection"
    if _scan_credential_exfil(text):
        return "credential_exfil"
    if _scan_invisible_unicode(text):
        return "invisible_unicode"
    return None
