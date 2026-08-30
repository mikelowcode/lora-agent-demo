"""
Tests for bm25.py — the hand-rolled Okapi BM25 scorer that replaces Jaccard
keyword-overlap scoring as stage 1 of MemoryManager.query_corpus().
"""

from localist import bm25


class TestTokenize:

    def test_lowercases_and_splits_on_non_alnum(self):
        assert bm25.tokenize("Hello, World! v0.5.0") == ["hello", "world", "v0", "5", "0"]

    def test_preserves_repeats_unlike_a_set(self):
        # This is the whole reason bm25 doesn't reuse document_index.token_set
        # (a deduplicated set) — BM25 needs term *frequency*.
        tokens = bm25.tokenize("the cat sat on the mat with the hat")
        assert tokens.count("the") == 3

    def test_empty_string_returns_empty_list(self):
        assert bm25.tokenize("") == []


class TestScoreDocuments:

    def test_empty_query_scores_everything_zero(self):
        docs = [("a", "some content"), ("b", "other content")]
        scores = bm25.score_documents("", docs)
        assert scores == {"a": 0.0, "b": 0.0}

    def test_non_matching_document_scores_zero(self):
        docs = [("a", "apple pie recipe"), ("b", "completely unrelated text")]
        scores = bm25.score_documents("apple pie", docs)
        assert scores["b"] == 0.0
        assert scores["a"] > 0.0

    def test_no_documents_returns_empty_dict(self):
        assert bm25.score_documents("anything", []) == {}

    def test_every_document_scored_and_keyed_correctly(self):
        docs = [("first", "sqlite memory backend"), ("second", "oMLX runtime version")]
        scores = bm25.score_documents("sqlite backend", docs)
        assert set(scores.keys()) == {"first", "second"}

    def test_rare_term_outranks_common_term_repeated_many_times(self):
        # The actual point of the BM25 upgrade over plain Jaccard overlap:
        # a document containing a rare term (high IDF, since almost no
        # other document contains it) should outrank a document that only
        # repeats a common term (low IDF, since most documents contain it)
        # many times — raw term-frequency alone should not win.
        docs = [
            ("common_repeated", ("common " * 20).strip()),
            ("rare_once", "rare fact about localist framework"),
            ("filler1", "common stuff here"),
            ("filler2", "common stuff there"),
            ("filler3", "common stuff everywhere"),
            ("filler4", "common stuff again"),
        ]
        scores = bm25.score_documents("rare common", docs)
        assert scores["rare_once"] > scores["common_repeated"]

    def test_higher_term_frequency_scores_higher_all_else_equal(self):
        docs = [
            ("low_tf", "sqlite is used once here for storage"),
            ("high_tf", "sqlite sqlite sqlite is used for storage"),
        ]
        scores = bm25.score_documents("sqlite", docs)
        assert scores["high_tf"] > scores["low_tf"]

    def test_shorter_document_scores_higher_than_longer_for_same_term_count(self):
        # Document-length normalization (the `b` parameter): a document that
        # pads out its length with unrelated tokens should score lower for
        # the same single occurrence of the query term.
        docs = [
            ("short", "sqlite committed"),
            ("long", "sqlite committed " + ("padding word here to increase length " * 10)),
        ]
        scores = bm25.score_documents("sqlite committed", docs)
        assert scores["short"] > scores["long"]

    def test_custom_k1_b_accepted(self):
        docs = [("a", "sqlite memory backend")]
        scores = bm25.score_documents("sqlite", docs, k1=1.2, b=0.5)
        assert scores["a"] > 0.0
