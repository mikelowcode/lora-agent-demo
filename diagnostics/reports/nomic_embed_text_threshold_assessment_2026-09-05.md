# Semantic-Gating Threshold Assessment — `nomic-embed-text:latest`

**Date:** 2026-09-05
**Script:** `diagnostics/nomic_embed_text_threshold_probe.py`
**Model under test:** `nomic-embed-text:latest` — real Ollama `/api/embed`, no stubs
**Tuned-model baseline:** `mlx-community/embeddinggemma-300m-4bit` (current production thresholds: ESA=0.72, LR=0.6, RI=0.65, episodic=0.7)
**Status:** READ-ONLY. `planner.py` unmodified. All test batteries copied verbatim from the original tuning diagnostics for direct comparability.

**Trigger:** docs/architecture/16-runtime-backend-layer.md §16.15 — a live report that an Xbox-related episodic memory never reached the prompt after switching the desktop build's embedding source to nomic-embed-text, because `_semantic_gating_disabled` unconditionally disables the episodic-relevance and search-intent semantic gates for any non-tuned embedding model.

## `lookup_request` / `explicit_search_action`

Categories: A = live false positives (must NOT fire), B = `_ALL_SEARCH_NEGATIVE_FILTERS` phrases (filtered pre-gate regardless of threshold, shown for reference only), C = confirmed true positives (MUST fire), D = adversarial negatives (must NOT fire).

| Cat | Utterance | LR score | ESA score | Filtered |
|-----|-----------|---------:|----------:|:--------:|
| A | Tell me how Localist works? | 0.4278 | 0.4077 |  |
| A | Can you read my wiki files? | 0.4930 | 0.4760 |  |
| A | List the files in my vault? | 0.4649 | 0.4380 |  |
| B-identity | who are you | 0.4108 | 0.3814 | Y |
| B-identity | what are you | 0.4643 | 0.4285 | Y |
| B-identity | what can you do | 0.4667 | 0.4926 | Y |
| B-identity | what can you help with | 0.4936 | 0.5049 | Y |
| B-identity | what do you do | 0.4428 | 0.4805 | Y |
| B-greeting | hey lora | 0.3952 | 0.3924 | Y |
| B-greeting | hi there | 0.4735 | 0.4366 | Y |
| B-greeting | hey there | 0.5013 | 0.4626 | Y |
| B-greeting | what's up | 0.6084 | 0.4616 | Y |
| C | Can you look up Apple's price hike for the MacBook Neo… | 0.6908 | 0.4803 |  |
| C | Can you look up their next-generation in-house Microso… | 0.6395 | 0.5350 |  |
| C | Can you look up Microsoft's next-generation in-house A… | 0.6462 | 0.5693 |  |
| D-verb-swap | Can you help me with this? | 0.5819 | 0.5776 |  |
| D-verb-swap | Could you check this for me? | 0.6071 | 0.5839 |  |
| D-verb-swap | Would you look at this? | 0.7039 | 0.6476 |  |
| D-verb-swap | Can you tell me about this? | 0.7365 | 0.5301 |  |
| D-modal-swap | Will you look into this? | 0.6873 | 0.6630 |  |
| D-modal-swap | Do you mind looking at this? | 0.6832 | 0.6185 |  |
| D-length | Can you help? | 0.5131 | 0.5047 |  |
| D-length | Can you help me understand this particular concept in … | 0.4506 | 0.4862 |  |
| D-domain | Can you help me understand how Localist works? | 0.4030 | 0.3937 |  |
| D-domain | Could you explain what this system does? | 0.4779 | 0.4721 |  |
| D-domain | Can you look at my notes and help me organize them? | 0.4804 | 0.4622 |  |
| D-domain | Could you read through this document for me? | 0.5573 | 0.5269 |  |
| D-domain | Would you help me plan a trip to Japan? | 0.4453 | 0.4459 |  |
| D-domain | Can you tell me a joke? | 0.4940 | 0.3978 |  |

### `lookup_request` threshold trade-off (Cat C must fire, Cat A/D must not)

| Threshold | C survivors | A/D false positives |
|:---------:|:-----------:|:--------------------:|
| 0.30 | 3/3 | 17/17 |
| 0.32 | 3/3 | 17/17 |
| 0.34 | 3/3 | 17/17 |
| 0.36 | 3/3 | 17/17 |
| 0.38 | 3/3 | 17/17 |
| 0.40 | 3/3 | 17/17 |
| 0.42 | 3/3 | 16/17 |
| 0.44 | 3/3 | 15/17 |
| 0.46 | 3/3 | 13/17 |
| 0.48 | 3/3 | 11/17 |
| 0.50 | 3/3 | 8/17 |
| 0.52 | 3/3 | 7/17 |
| 0.54 | 3/3 | 7/17 |
| 0.56 | 3/3 | 6/17 |
| 0.58 | 3/3 | 6/17 |
| 0.60 | 3/3 | 5/17 |
| 0.62 | 3/3 | 4/17 |
| 0.64 | 2/3 | 4/17 |
| 0.66 | 1/3 | 4/17 |
| 0.68 | 1/3 | 4/17 |
| 0.70 | 0/3 | 2/17 |
| 0.72 | 0/3 | 1/17 |
| 0.74 | 0/3 | 0/17 |
| 0.76 | 0/3 | 0/17 |
| 0.78 | 0/3 | 0/17 |
| 0.80 | 0/3 | 0/17 |
| 0.82 | 0/3 | 0/17 |
| 0.84 | 0/3 | 0/17 |
| 0.86 | 0/3 | 0/17 |
| 0.88 | 0/3 | 0/17 |
| 0.90 | 0/3 | 0/17 |

Min Cat-C LR score: **0.6395**. Max Cat-A/D LR score: **0.7365**. **No clean separation** in this battery — see trade-off table for the actual cost at each candidate.

### `explicit_search_action` — same battery, no dedicated Cat-C-equivalent true positives exist in this battery (ESA has no positive templates tested here beyond what LR's Cat C also scores on); reported for completeness/regression only.

| Threshold | A/D false positives (of 17) |
|:---------:|:--------------------------------------------------------:|
| 0.30 | 17/17 |
| 0.32 | 17/17 |
| 0.34 | 17/17 |
| 0.36 | 17/17 |
| 0.38 | 17/17 |
| 0.40 | 15/17 |
| 0.42 | 14/17 |
| 0.44 | 13/17 |
| 0.46 | 12/17 |
| 0.48 | 9/17 |
| 0.50 | 8/17 |
| 0.52 | 7/17 |
| 0.54 | 5/17 |
| 0.56 | 5/17 |
| 0.58 | 4/17 |
| 0.60 | 3/17 |
| 0.62 | 2/17 |
| 0.64 | 2/17 |
| 0.66 | 1/17 |
| 0.68 | 0/17 |
| 0.70 | 0/17 |
| 0.72 | 0/17 |
| 0.74 | 0/17 |
| 0.76 | 0/17 |
| 0.78 | 0/17 |
| 0.80 | 0/17 |
| 0.82 | 0/17 |
| 0.84 | 0/17 |
| 0.86 | 0/17 |
| 0.88 | 0/17 |
| 0.90 | 0/17 |

## `research_intent`

| Cat | Utterance | RI score | Filtered |
|-----|-----------|---------:|:--------:|
| T | How much does the Tesla Model 3 cost? | 0.5530 |  |
| T | What's the price of the new iPhone? | 0.6184 |  |
| T | Can you find out how much AWS charges for S3 storage? | 0.6165 |  |
| T | What are the pricing tiers for Notion? | 0.7321 |  |
| T | Look up the specs on the RTX 4090 | 0.6422 |  |
| T | Find me the cost of a one-bedroom apartment in Austin | 0.5894 |  |
| T | What does ChatGPT Plus cost per month? | 0.6874 |  |
| T | Track down pricing for Salesforce Enterprise | 0.6506 |  |
| T | Can you check what a Peloton subscription costs? | 0.7946 |  |
| T | What's the going rate for a plumber in this area? | 0.5281 |  |
| L | Can you look up the release date for this? | 0.6107 |  |
| L | Could you look up what year this happened? | 0.5097 |  |
| L | Can you look up information about the latest Apple pro… | 0.5603 |  |
| L | Can you look up Apple's price hike for the MacBook Neo… | 0.6364 |  |
| L | Can you look up their next-generation in-house Microso… | 0.4822 |  |
| L-price-adj | Could you find out the current stock price for me? | 0.6939 |  |
| K | What is this? | 0.4871 |  |
| K | Tell me about this company | 0.4494 |  |
| K | Explain how blockchain works | 0.3385 |  |
| K | What do you know about this? | 0.4841 |  |
| F | What's the latest on this? | 0.5132 |  |
| F | Is there anything new about this? | 0.4069 |  |
| F | What's the current status of this project? | 0.5514 |  |
| E | Is this too expensive for me? | 0.5527 | Y |
| E | Do you think this is worth the price? | 0.5526 | Y |
| E | I can't believe how expensive rent is these days | 0.5075 | Y |
| E | That seems like a lot of money for what you get | 0.5310 | Y |
| G | Can you help me with this? | 0.5485 |  |
| G | What's up? | 0.4299 |  |
| G | How are you doing today? | 0.3643 |  |
| G | Thanks, that's helpful | 0.4269 |  |

### `research_intent` threshold trade-off

| Threshold | T survivors | FP pool fires |
|:---------:|:-----------:|:--------------:|
| 0.30 | 10/10 | 16/16 |
| 0.32 | 10/10 | 16/16 |
| 0.34 | 10/10 | 15/16 |
| 0.36 | 10/10 | 15/16 |
| 0.38 | 10/10 | 14/16 |
| 0.40 | 10/10 | 14/16 |
| 0.42 | 10/10 | 13/16 |
| 0.44 | 10/10 | 11/16 |
| 0.46 | 10/10 | 10/16 |
| 0.48 | 10/10 | 10/16 |
| 0.50 | 10/10 | 7/16 |
| 0.52 | 10/10 | 5/16 |
| 0.54 | 9/10 | 5/16 |
| 0.56 | 8/10 | 3/16 |
| 0.58 | 8/10 | 2/16 |
| 0.60 | 7/10 | 2/16 |
| 0.62 | 5/10 | 1/16 |
| 0.64 | 5/10 | 0/16 |
| 0.66 | 3/10 | 0/16 |
| 0.68 | 3/10 | 0/16 |
| 0.70 | 2/10 | 0/16 |
| 0.72 | 2/10 | 0/16 |
| 0.74 | 1/10 | 0/16 |
| 0.76 | 1/10 | 0/16 |
| 0.78 | 1/10 | 0/16 |
| 0.80 | 0/10 | 0/16 |
| 0.82 | 0/10 | 0/16 |
| 0.84 | 0/10 | 0/16 |
| 0.86 | 0/10 | 0/16 |
| 0.88 | 0/10 | 0/16 |
| 0.90 | 0/10 | 0/16 |
| 0.92 | 0/10 | 0/16 |
| 0.94 | 0/10 | 0/16 |

Min Cat-T RI score: **0.5281**. Max FP-pool RI score: **0.6364**. **No clean separation** in this battery.

## `episodic_relevance`

`positive` includes the original 10-utterance battery plus 3 live-motivated additions (the actual Xbox/gaming phrasing this investigation started from).

| Cat | Utterance | Score |
|-----|-----------|------:|
| positive | Help me prepare for the upcoming Claude Impact Lab on … | 0.7411 |
| positive | What do I need to bring to my dentist appointment next… | 0.6644 |
| positive | Can you help me get ready for my presentation on Frida… | 0.7370 |
| positive | What did we decide about the migration plan? | 0.7353 |
| positive | What do I have going on this week? | 0.8375 |
| positive | Catch me up on my project status. | 0.6835 |
| positive | What was the plan we settled on last time? | 0.7309 |
| positive | Help me get ready for my trip to Japan. | 0.7154 |
| positive | Prep me for my meeting with the investors tomorrow. | 0.6542 |
| positive | What did I say I would do about the server migration? | 0.5023 |
| positive | What games do I play? | 0.6179 |
| positive | What console do I own? | 0.5092 |
| positive | Tell me about my Xbox. | 0.5112 |
| negative | What is the capital of France? | 0.3777 |
| negative | Write a haiku about the ocean. | 0.4600 |
| negative | Search the web for the latest news on SpaceX. | 0.4761 |
| negative | What is 2+2? | 0.3978 |
| negative | Summarize this document for me. | 0.5541 |
| negative | Fetch this URL and tell me what it says. | 0.4417 |
| negative | Remind me how photosynthesis works. | 0.4635 |
| negative | Help me write a cover letter for a job application. | 0.5506 |
| negative | Prepare a report on quarterly earnings. | 0.4576 |
| negative | What is on the front page of the New York Times today? | 0.5161 |
| negative | Explain how neural networks work. | 0.3830 |
| negative | Get ready to receive a large file upload. | 0.5112 |
| negative | I have a headache, what should I do? | 0.5767 |
| negative | My favorite color is blue, what goes well with it? | 0.4663 |
| negative | Can you help me plan a birthday party for my friend? | 0.6163 |
| negative | What is my IP address? | 0.4623 |
| negative | Prepare a Python script that reverses a string. | 0.4061 |
| negative | I need help understanding this error message. | 0.4338 |
| negative | What time is it in Tokyo right now? | 0.5128 |
| negative | Draft an email to my landlord about the leak. | 0.4469 |

### `episodic_relevance` threshold trade-off

| Threshold | Positive survivors | Negative false positives |
|:---------:|:-------------------:|:-------------------------:|
| 0.30 | 13/13 | 20/20 |
| 0.32 | 13/13 | 20/20 |
| 0.34 | 13/13 | 20/20 |
| 0.36 | 13/13 | 20/20 |
| 0.38 | 13/13 | 19/20 |
| 0.40 | 13/13 | 17/20 |
| 0.42 | 13/13 | 16/20 |
| 0.44 | 13/13 | 15/20 |
| 0.46 | 13/13 | 12/20 |
| 0.48 | 13/13 | 7/20 |
| 0.50 | 13/13 | 7/20 |
| 0.52 | 10/13 | 4/20 |
| 0.54 | 10/13 | 4/20 |
| 0.56 | 10/13 | 2/20 |
| 0.58 | 10/13 | 1/20 |
| 0.60 | 10/13 | 1/20 |
| 0.62 | 9/13 | 0/20 |
| 0.64 | 9/13 | 0/20 |
| 0.66 | 8/13 | 0/20 |
| 0.68 | 7/13 | 0/20 |
| 0.70 | 6/13 | 0/20 |
| 0.72 | 5/13 | 0/20 |
| 0.74 | 2/13 | 0/20 |
| 0.76 | 1/13 | 0/20 |
| 0.78 | 1/13 | 0/20 |
| 0.80 | 1/13 | 0/20 |
| 0.82 | 1/13 | 0/20 |
| 0.84 | 0/13 | 0/20 |
| 0.86 | 0/13 | 0/20 |
| 0.88 | 0/13 | 0/20 |
| 0.90 | 0/13 | 0/20 |
| 0.92 | 0/13 | 0/20 |
| 0.94 | 0/13 | 0/20 |

Min positive score: **0.5023**. Max negative score: **0.6163**. **No clean separation** in this battery — see trade-off table.

## Summary

No threshold is auto-selected here beyond what's stated per section above — same discipline as the original tuning diagnostics (data informs the choice; the choice itself is recorded in planner.py's per-model threshold table with its rationale).

---

*Generated by `diagnostics/nomic_embed_text_threshold_probe.py` against the real, running Ollama daemon.*
