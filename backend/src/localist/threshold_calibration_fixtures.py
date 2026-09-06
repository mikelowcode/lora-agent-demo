"""
Shared test-battery fixtures for semantic-gating threshold calibration.

Single source of truth for the utterance batteries used to measure
planner.py's four semantic-gating thresholds (explicit_search_action,
lookup_request, research_intent, episodic_relevance). Read by both:
  - diagnostics/nomic_embed_text_threshold_probe.py (offline, human-reviewed)
  - threshold_calibration.py (live, in-app calibration)

These must never drift apart, since a discrepancy would make live-calibrated
thresholds not comparable to the hand-validated ones. Data copied verbatim
from the original tuning diagnostics (score_lookup_request_templates.py,
score_research_intent_templates.py, score_episodic_relevance_templates.py) —
do not re-invent or rephrase entries here; add new ones instead if coverage
needs to grow.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# lookup_request / explicit_search_action — Categories A-D
# ---------------------------------------------------------------------------

LR_CAT_A: list[tuple[str, str]] = [
    ("A", "Tell me how Localist works?"),
    ("A", "Can you read my wiki files?"),
    ("A", "List the files in my vault?"),
]

LR_CAT_B: list[tuple[str, str]] = [
    ("B-identity", "who are you"),
    ("B-identity", "what are you"),
    ("B-identity", "what can you do"),
    ("B-identity", "what can you help with"),
    ("B-identity", "what do you do"),
    ("B-greeting", "hey lora"),
    ("B-greeting", "hi there"),
    ("B-greeting", "hey there"),
    ("B-greeting", "what's up"),
]

LR_CAT_C: list[tuple[str, str]] = [
    ("C", "Can you look up Apple's price hike for the MacBook Neo and iPad?"),
    ("C", "Can you look up their next-generation in-house Microsoft AI models?"),
    ("C", "Can you look up Microsoft's next-generation in-house AI models?"),
]

LR_CAT_D: list[tuple[str, str, str]] = [
    ("D-verb-swap",   "short/generic",        "Can you help me with this?"),
    ("D-verb-swap",   "short/generic",        "Could you check this for me?"),
    ("D-verb-swap",   "short/generic",        "Would you look at this?"),
    ("D-verb-swap",   "short/generic",        "Can you tell me about this?"),
    ("D-modal-swap",  "short/generic",        "Will you look into this?"),
    ("D-modal-swap",  "short/generic",        "Do you mind looking at this?"),
    ("D-length",      "bare-minimum",         "Can you help?"),
    ("D-length",      "fuller-sentence",      "Can you help me understand this particular concept in more depth?"),
    ("D-domain",      "project-referential",  "Can you help me understand how Localist works?"),
    ("D-domain",      "project-referential",  "Could you explain what this system does?"),
    ("D-domain",      "file-referencing",     "Can you look at my notes and help me organize them?"),
    ("D-domain",      "file-referencing",     "Could you read through this document for me?"),
    ("D-domain",      "generic/unrelated",    "Would you help me plan a trip to Japan?"),
    ("D-domain",      "generic/unrelated",    "Can you tell me a joke?"),
]

# ---------------------------------------------------------------------------
# research_intent — Categories T/L/K/F/E/G
# ---------------------------------------------------------------------------

RI_CAT_T: list[str] = [
    "How much does the Tesla Model 3 cost?",
    "What's the price of the new iPhone?",
    "Can you find out how much AWS charges for S3 storage?",
    "What are the pricing tiers for Notion?",
    "Look up the specs on the RTX 4090",
    "Find me the cost of a one-bedroom apartment in Austin",
    "What does ChatGPT Plus cost per month?",
    "Track down pricing for Salesforce Enterprise",
    "Can you check what a Peloton subscription costs?",
    "What's the going rate for a plumber in this area?",
]
RI_CAT_L: list[str] = [
    "Can you look up the release date for this?",
    "Could you look up what year this happened?",
    "Can you look up information about the latest Apple products?",
    "Can you look up Apple's price hike for the MacBook Neo and iPad?",
    "Can you look up their next-generation in-house Microsoft AI models?",
]
RI_CAT_L_PRICE_ADJACENT: str = "Could you find out the current stock price for me?"
RI_CAT_K: list[str] = [
    "What is this?",
    "Tell me about this company",
    "Explain how blockchain works",
    "What do you know about this?",
]
RI_CAT_F: list[str] = [
    "What's the latest on this?",
    "Is there anything new about this?",
    "What's the current status of this project?",
]
RI_CAT_E: list[str] = [
    "Is this too expensive for me?",
    "Do you think this is worth the price?",
    "I can't believe how expensive rent is these days",
    "That seems like a lot of money for what you get",
]
RI_CAT_G: list[str] = [
    "Can you help me with this?",
    "What's up?",
    "How are you doing today?",
    "Thanks, that's helpful",
]

# ---------------------------------------------------------------------------
# episodic_relevance — POSITIVES/NEGATIVES
# ---------------------------------------------------------------------------

EP_POSITIVES: list[str] = [
    "Help me prepare for the upcoming Claude Impact Lab on August 6th.",
    "What do I need to bring to my dentist appointment next week?",
    "Can you help me get ready for my presentation on Friday?",
    "What did we decide about the migration plan?",
    "What do I have going on this week?",
    "Catch me up on my project status.",
    "What was the plan we settled on last time?",
    "Help me get ready for my trip to Japan.",
    "Prep me for my meeting with the investors tomorrow.",
    "What did I say I would do about the server migration?",
]
EP_NEGATIVES: list[str] = [
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "Search the web for the latest news on SpaceX.",
    "What is 2+2?",
    "Summarize this document for me.",
    "Fetch this URL and tell me what it says.",
    "Remind me how photosynthesis works.",
    "Help me write a cover letter for a job application.",
    "Prepare a report on quarterly earnings.",
    "What is on the front page of the New York Times today?",
    "Explain how neural networks work.",
    "Get ready to receive a large file upload.",
    "I have a headache, what should I do?",
    "My favorite color is blue, what goes well with it?",
    "Can you help me plan a birthday party for my friend?",
    "What is my IP address?",
    "Prepare a Python script that reverses a string.",
    "I need help understanding this error message.",
    "What time is it in Tokyo right now?",
    "Draft an email to my landlord about the leak.",
]

# A real live-reported failure (Xbox/gaming episodic-memory miss against
# nomic-embed-text), added on top of the original battery.
EP_POSITIVES_LIVE: list[str] = [
    "What games do I play?",
    "What console do I own?",
    "Tell me about my Xbox.",
]
