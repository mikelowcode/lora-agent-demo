"""
bm25 — Hand-rolled Okapi BM25 scoring for the corpus keyword prefilter.

Pure scoring module: no I/O, no SQLite, no embeddings, no knowledge of any
other project module. Replaces the Jaccard set-overlap scoring
(memory_manager._keyword_score()) previously used as stage 1 of
MemoryManager.query_corpus()'s two-stage (keyword prefilter -> embedding
re-rank) pipeline, and the equivalent per-row fallback in
EpisodicMemoryReader._score_all_active().

Why hand-rolled rather than a library (rank_bm25, bm25s, etc.): Localist's
corpus is small (single-user, local-first, a personal wiki, not thousands
of documents), query_corpus() already fetches the full candidate set fresh
from SQLite on every call rather than maintaining a persistent index, and
there is no sustained query throughput to speak of. A vectorized library's
selling point (fast scoring over large corpora / high QPS) doesn't apply
here, and an index-then-retrieve library API would be a structural
mismatch against the existing fetch-then-score-in-memory pattern. A small,
dependency-free function fits the existing shape exactly: same input/output
contract as the _keyword_score() it replaces, no new object to invalidate,
no new package in requirements.txt.

Scores are raw and unbounded — deliberately not normalized onto a [0, 1]
scale. A ceiling-based normalization (score / the score's own supremum as
term frequency -> infinity) was tried and rejected: it mathematically caps
a document that mentions every query term exactly once — a perfectly good
single-mention match, the common case for short wiki pages — at roughly
40% of scale, making a fixed absolute threshold like the RAG-source 0.55
gate in controller_agent.py nearly unreachable for real matches. Callers
that need an absolute relevance floor comparable to cosine similarity
should keep using cosine (or, for the keyword-only case, drop the floor
entirely and trust BM25's relative ranking instead — see
memory_manager.DocumentResult.scored_by_embedding and
controller_agent.py's RAG-source filters for how that's applied).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Hashable

DEFAULT_K1 = 1.5
DEFAULT_B  = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    Term-frequency-preserving tokenization: a list, not a set.

    BM25's core input signal — how many times a term appears in a
    document — is lost the moment tokens are deduplicated. This is why
    memory_manager.document_index.token_set (a deduplicated set, built for
    Jaccard overlap) isn't reused here; BM25 tokenizes the raw `content`
    column fresh at query time instead. Same lowercase-alphanumeric-run
    rule as memory_manager._tokenize(), so results stay comparable to the
    keyword-fallback path elsewhere in the codebase.
    """
    return _TOKEN_RE.findall(text.lower())


def score_documents(
    query:     str,
    documents: list[tuple[Hashable, str]],
    k1:        float = DEFAULT_K1,
    b:         float = DEFAULT_B,
) -> dict[Hashable, float]:
    """
    Score every document in `documents` against `query` via Okapi BM25.

    Parameters
    ----------
    query :
        Free-text query string.
    documents :
        (key, content) pairs. `key` is any hashable identifier the caller
        uses to map scores back onto its own rows (e.g. a document_index
        row's `id`); `content` is the raw, untokenized document text.
    k1, b :
        Standard BM25 free parameters. Defaults (1.5, 0.75) per Robertson/
        Sparck Jones — override only with a concrete, measured reason.

    Returns
    -------
    dict[key, float]
        Raw, unbounded BM25 score per document key — comparable *within*
        one call (same query, same candidate set) for ranking purposes,
        but not on any fixed scale across different queries or corpora,
        and not comparable to cosine similarity (see module docstring).
        0.0 for a document sharing no terms with the query, and for every
        document when the query itself tokenizes to nothing.

        IDF and average-document-length are computed over exactly this
        `documents` set, per call — no persisted corpus-wide statistics.
        This mirrors _keyword_score()'s existing per-call scoping (it only
        ever sees the current call's row set, never a separately
        maintained global corpus stat), just batched across the whole set
        instead of evaluated one row at a time, since BM25's IDF and
        average-document-length terms are corpus-wide rather than
        per-row.
    """
    query_terms = tokenize(query)
    if not query_terms:
        return {key: 0.0 for key, _ in documents}

    doc_term_counts: dict[Hashable, Counter] = {}
    doc_lengths:     dict[Hashable, int]     = {}
    doc_freq:        Counter                 = Counter()

    for key, content in documents:
        terms = tokenize(content)
        counts = Counter(terms)
        doc_term_counts[key] = counts
        doc_lengths[key]     = len(terms)
        for term in counts:
            doc_freq[term] += 1

    num_docs    = len(documents)
    total_len   = sum(doc_lengths.values())
    avg_doc_len = (total_len / num_docs) if num_docs else 0.0

    # Lucene-style non-negative IDF: log(1 + (N - df + 0.5) / (df + 0.5)).
    # The original Robertson/Sparck Jones formula (without the "+1") can go
    # negative for terms present in over half the corpus, which would let
    # a document's score decrease by containing a common query term — an
    # odd outcome for a single-user corpus with a handful of documents
    # where that "over half" case is easy to hit by chance.
    idf: dict[str, float] = {
        term: math.log(1 + (num_docs - doc_freq.get(term, 0) + 0.5) / (doc_freq.get(term, 0) + 0.5))
        for term in set(query_terms)
    }

    scores: dict[Hashable, float] = {}
    for key, _ in documents:
        counts  = doc_term_counts[key]
        doc_len = doc_lengths[key]
        score   = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            length_norm = (
                (1 - b + b * doc_len / avg_doc_len) if avg_doc_len else 1.0
            )
            score += idf[term] * (tf * (k1 + 1)) / (tf + k1 * length_norm)
        scores[key] = score

    return scores
