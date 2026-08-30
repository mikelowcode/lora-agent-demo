"""
Phase 1 tests — localist-mcp server (mcp_server/).
Phase 2 adds fetch_url coverage (mcp_server/url_fetch.py) — ports the
retired standalone Fetcher microservice's /extract path in-process.
Phase 3 adds web_search coverage (mcp_server/web_search.py) — ports the
LangSearch integration in-process, no runtime.infer() fallback. A second
provider (Brave) was later added behind the SEARCH_PROVIDER env var switch.
Also covers github.github_search / github.github_read / github.github_release
(mcp_server/github.py) — public-repo GitHub REST reads, GITHUB_TOKEN optional
(used opportunistically, never required — see the module docstring).

Covers:
  - file_ops.read_file / write_file / append_file: sandboxing, truncation,
    error raising (ported behaviour from ToolDispatcher._file_read/_write/_append)
  - url_fetch.fetch_url: success, timeout clamping, connection error, HTTP
    4xx/5xx, and extraction_failed (paywall/empty content) — error taxonomy
    ported from fetcher/models.py's ErrorResponse.error_code
  - web_search.web_search: results found (bullet formatting matches the
    legacy shape exactly), empty results, missing API key (clean error, no
    inference call), network/timeout error — for both the LangSearch
    provider (default / SEARCH_PROVIDER=langsearch) and the Brave provider
    (SEARCH_PROVIDER=brave), plus the unknown-provider error
  - chart.generate_chart: valid bar/line/pie cases, each argument-validation
    rejection case (bad chart_type, empty labels, empty/mismatched datasets,
    multi-dataset pie), and confirmation that the PNG actually lands on disk
    at the returned png_path.
  - All tools as registered on the FastMCP instance, exercised through
    an in-process MCP client session (mcp.shared.memory) — no network server
    required.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import fitz  # PyMuPDF
import httpx
import pytest

from mcp.shared.memory import create_connected_server_and_client_session

from localist.mcp_server import chart, file_ops, github, hacker_news, news_search, ocr, search_format, url_fetch, web_search
from localist.mcp_server.main import mcp as mcp_app


# ---------------------------------------------------------------------------
# file_ops — direct unit tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_project_root():
    """Every test sets its own root explicitly; avoid state leaking between tests."""
    yield
    file_ops.set_project_root(Path(__file__).resolve().parent.parent)


@pytest.fixture(autouse=True)
def _reset_ocr_upload_root():
    """Same convention as _reset_project_root above, for ocr.py's sandbox root."""
    yield
    ocr.set_upload_root(Path(__file__).resolve().parent.parent)


class TestFileOpsRead:
    def test_read_returns_file_content(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        (tmp_path / "notes.md").write_text("hello world", encoding="utf-8")
        assert file_ops.read_file("notes.md") == "hello world"

    def test_read_missing_file_raises(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError, match="file not found"):
            file_ops.read_file("ghost.md")

    def test_read_truncates_long_content(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        big = "x" * (file_ops._MAX_FILE_READ_CHARS + 500)
        (tmp_path / "big.txt").write_text(big, encoding="utf-8")
        result = file_ops.read_file("big.txt")
        assert result.endswith("\n… [truncated]")
        assert len(result) == file_ops._MAX_FILE_READ_CHARS + len("\n… [truncated]")

    def test_read_path_traversal_blocked(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError, match="path traversal"):
            file_ops.read_file("../../etc/passwd")


class TestFileOpsWrite:
    def test_write_creates_file(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        result = file_ops.write_file("out/result.md", "# Result\nContent here.")
        assert result.startswith("OK: wrote")
        assert (tmp_path / "out" / "result.md").read_text(encoding="utf-8") == "# Result\nContent here."

    def test_write_path_traversal_blocked(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError, match="path traversal"):
            file_ops.write_file("../escape.md", "nope")

    def test_write_empty_content_refused(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError, match="refusing empty file write"):
            file_ops.write_file("empty.md", "")
        assert not (tmp_path / "empty.md").exists()

    def test_write_whitespace_only_content_refused(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError, match="refusing empty file write"):
            file_ops.write_file("whitespace.md", "   \n\t  ")
        assert not (tmp_path / "whitespace.md").exists()

    def test_write_versions_on_existing_file(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        file_ops.write_file("dup.md", "first")
        result = file_ops.write_file("dup.md", "second")
        assert result == "OK: wrote 6 characters to dup_2.md"
        assert (tmp_path / "dup.md").read_text(encoding="utf-8") == "first"
        assert (tmp_path / "dup_2.md").read_text(encoding="utf-8") == "second"


class TestFileOpsAppend:
    def test_append_to_existing_file(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        (tmp_path / "log.txt").write_text("Line 1.\n", encoding="utf-8")
        result = file_ops.append_file("log.txt", "Line 2.\n")
        assert result.startswith("OK: appended")
        assert (tmp_path / "log.txt").read_text(encoding="utf-8") == "Line 1.\nLine 2.\n"

    def test_append_creates_parent_dirs(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        file_ops.append_file("a/b/c/deep.txt", "content")
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").read_text(encoding="utf-8") == "content"


# ---------------------------------------------------------------------------
# url_fetch.fetch_url — direct unit tests
# ---------------------------------------------------------------------------

_SAMPLE_ARTICLE_HTML = b"""
<html><head><title>Test Article Title</title>
<meta name="author" content="Jane Doe">
<meta property="article:published_time" content="2026-01-01">
</head>
<body>
<article>
<h1>Test Article Title</h1>
<p>This is the first paragraph of a reasonably long test article used to
verify that the readability extraction pipeline correctly identifies the
main content block and strips away any surrounding navigation or
boilerplate markup that a typical web page would include around the
actual article body text.</p>
<p>This is a second paragraph adding more substantive content so that the
extractor has enough signal to treat this block as the primary article
content rather than discarding it as noise or a login wall placeholder.</p>
</article>
</body></html>
"""

_EMPTY_HTML = b"<html><head><title>Login</title></head><body></body></html>"


def _raw_response(content: bytes, url: str = "https://example.com/article") -> url_fetch.RawResponse:
    return url_fetch.RawResponse(
        url               = url,
        status_code       = 200,
        content_type      = "text/html",
        content           = content,
        headers           = {},
        fetch_duration_ms = 12.3,
    )


class TestFetchUrlSuccess:
    def test_success_returns_expected_fields(self):
        with patch.object(url_fetch, "_fetch", AsyncMock(return_value=_raw_response(_SAMPLE_ARTICLE_HTML))):
            result = asyncio.run(url_fetch.fetch_url("https://example.com/article"))

        assert result["title"] == "Test Article Title"
        assert result["author"] == "Jane Doe"
        assert result["date_published"] == "2026-01-01"
        assert "reasonably long test article" in result["cleaned_text"]
        assert result["word_count"] > 0
        assert result["url"] == "https://example.com/article"
        assert result["fetch_duration_ms"] == 12.3

    def test_timeout_is_clamped_before_reaching_fetch(self):
        fake_fetch = AsyncMock(return_value=_raw_response(_SAMPLE_ARTICLE_HTML))
        with patch.object(url_fetch, "_fetch", fake_fetch):
            asyncio.run(url_fetch.fetch_url("https://example.com/article", timeout=100.0))
        fake_fetch.assert_called_once_with("https://example.com/article", 30.0)

        fake_fetch.reset_mock()
        with patch.object(url_fetch, "_fetch", fake_fetch):
            asyncio.run(url_fetch.fetch_url("https://example.com/article", timeout=0.1))
        fake_fetch.assert_called_once_with("https://example.com/article", 1.0)


class TestFetchUrlErrors:
    def test_timeout_maps_to_timeout_code(self):
        with patch.object(url_fetch, "_fetch", AsyncMock(side_effect=httpx.TimeoutException("timed out"))):
            with pytest.raises(url_fetch.FetchUrlError) as exc_info:
                asyncio.run(url_fetch.fetch_url("https://example.com/slow"))
        assert exc_info.value.error_code == "timeout"
        assert str(exc_info.value).startswith("ERROR: timeout —")

    def test_connect_error_maps_to_connection_error_code(self):
        with patch.object(url_fetch, "_fetch", AsyncMock(side_effect=httpx.ConnectError("refused"))):
            with pytest.raises(url_fetch.FetchUrlError) as exc_info:
                asyncio.run(url_fetch.fetch_url("https://unreachable.example"))
        assert exc_info.value.error_code == "connection_error"
        assert str(exc_info.value).startswith("ERROR: connection_error —")

    def test_http_404_maps_to_http_client_error_code(self):
        request  = httpx.Request("GET", "https://example.com/missing")
        response = httpx.Response(404, request=request)
        error    = httpx.HTTPStatusError("404", request=request, response=response)
        with patch.object(url_fetch, "_fetch", AsyncMock(side_effect=error)):
            with pytest.raises(url_fetch.FetchUrlError) as exc_info:
                asyncio.run(url_fetch.fetch_url("https://example.com/missing"))
        assert exc_info.value.error_code == "http_client_error"
        assert "404" in str(exc_info.value)

    def test_http_500_maps_to_http_server_error_code(self):
        request  = httpx.Request("GET", "https://example.com/broken")
        response = httpx.Response(500, request=request)
        error    = httpx.HTTPStatusError("500", request=request, response=response)
        with patch.object(url_fetch, "_fetch", AsyncMock(side_effect=error)):
            with pytest.raises(url_fetch.FetchUrlError) as exc_info:
                asyncio.run(url_fetch.fetch_url("https://example.com/broken"))
        assert exc_info.value.error_code == "http_server_error"

    def test_empty_extraction_maps_to_extraction_failed_code(self):
        """Paywall/login-wall page — readability produces no usable content."""
        with patch.object(url_fetch, "_fetch", AsyncMock(return_value=_raw_response(_EMPTY_HTML))):
            with pytest.raises(url_fetch.FetchUrlError) as exc_info:
                asyncio.run(url_fetch.fetch_url("https://example.com/paywalled"))
        assert exc_info.value.error_code == "extraction_failed"
        assert str(exc_info.value).startswith("ERROR: extraction_failed —")


# ---------------------------------------------------------------------------
# web_search.web_search — direct unit tests
# ---------------------------------------------------------------------------

def _langsearch_response(pages: list[dict], status_code: int = 200) -> httpx.Response:
    request = httpx.Request("POST", web_search._LANGSEARCH_ENDPOINT)
    return httpx.Response(
        status_code,
        json    = {"data": {"webPages": {"value": pages}}},
        request = request,
    )


class TestWebSearchSuccess:
    def test_results_formatted_matching_legacy_bullet_shape(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        pages = [
            {
                "name":        "oMLX Release Notes",
                "snippet":     "fallback snippet",
                "summary":     "x" * 750,  # forces truncation
                "displayUrl":  "example.com/omlx",
                "url":         "https://example.com/omlx",
            }
        ]
        response = _langsearch_response(pages)
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
            result = asyncio.run(web_search.web_search("oMLX release notes"))

        assert result["query"] == "oMLX release notes"
        assert result["result_count"] == 1
        text = result["result_text"]
        assert text.startswith("• oMLX Release Notes — example.com\n  ")
        assert text.endswith("[example.com/omlx]")
        # body truncated to <=700 chars on a word boundary, with a trailing
        # ellipsis marking the cut
        body_line = text.splitlines()[1].strip()
        assert len(body_line) <= 701
        assert body_line.endswith("…")

    def test_prefers_summary_over_snippet(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        pages = [{
            "name": "Title", "snippet": "snippet text", "summary": "summary text",
            "url": "https://example.com",
        }]
        response = _langsearch_response(pages)
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
            result = asyncio.run(web_search.web_search("q"))
        assert "summary text" in result["result_text"]
        assert "snippet text" not in result["result_text"]

    def test_empty_results_returns_success_not_error(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        response = _langsearch_response([])
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
            result = asyncio.run(web_search.web_search("nothing found query"))
        assert result["result_text"] == "No results found."
        assert result["result_count"] == 0


class TestWebSearchErrors:
    def test_missing_api_key_raises_clean_error_without_network_call(self, monkeypatch):
        monkeypatch.delenv("LANGSEARCH_API_KEY", raising=False)
        fake_post = AsyncMock()
        with patch.object(httpx.AsyncClient, "post", fake_post):
            with pytest.raises(ValueError, match="LANGSEARCH_API_KEY not configured"):
                asyncio.run(web_search.web_search("anything"))
        fake_post.assert_not_called()

    def test_empty_string_api_key_treated_as_missing(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "")
        with pytest.raises(ValueError, match="LANGSEARCH_API_KEY not configured"):
            asyncio.run(web_search.web_search("anything"))

    def test_connection_error_wraps_as_clean_error(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        with patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=httpx.ConnectError("refused"))):
            with pytest.raises(ValueError, match="ERROR: web_search failed —"):
                asyncio.run(web_search.web_search("q"))

    def test_timeout_wraps_as_clean_error(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        with patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=httpx.TimeoutException("timed out"))):
            with pytest.raises(ValueError, match="ERROR: web_search failed —"):
                asyncio.run(web_search.web_search("q"))

    def test_http_error_status_wraps_as_clean_error(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        response = _langsearch_response([], status_code=500)
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
            with pytest.raises(ValueError, match="ERROR: web_search failed —"):
                asyncio.run(web_search.web_search("q"))


# ---------------------------------------------------------------------------
# web_search.web_search — SEARCH_PROVIDER dispatch (langsearch default,
# explicit langsearch, brave, unknown provider)
# ---------------------------------------------------------------------------

class TestWebSearchProviderDispatch:
    def test_provider_unset_defaults_to_langsearch(self, monkeypatch):
        monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        response = _langsearch_response([{"name": "T", "snippet": "s", "url": "https://e.com"}])
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)) as fake_post:
            result = asyncio.run(web_search.web_search("q"))
        fake_post.assert_called_once()
        assert result["result_count"] == 1

    def test_provider_explicit_langsearch_hits_langsearch(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "langsearch")
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        response = _langsearch_response([{"name": "T", "snippet": "s", "url": "https://e.com"}])
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)) as fake_post:
            result = asyncio.run(web_search.web_search("q"))
        fake_post.assert_called_once()
        assert result["result_count"] == 1

    def test_provider_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "LangSearch")
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        response = _langsearch_response([])
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
            result = asyncio.run(web_search.web_search("q"))
        assert result["result_text"] == "No results found."

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "bing")
        with pytest.raises(ValueError, match="ERROR: unknown SEARCH_PROVIDER 'bing'"):
            asyncio.run(web_search.web_search("q"))


# ---------------------------------------------------------------------------
# web_search._web_search_brave — direct unit tests
# ---------------------------------------------------------------------------

def _brave_response(results: list[dict], status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", web_search._BRAVE_ENDPOINT)
    return httpx.Response(
        status_code,
        json    = {"web": {"results": results}},
        request = request,
    )


class TestWebSearchBraveSuccess:
    def test_results_formatted_matching_bullet_shape(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "brave")
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        results = [
            {
                "title":       "oMLX Release Notes",
                "description": "x" * 750,  # forces truncation
                "url":         "https://example.com/omlx",
            }
        ]
        response = _brave_response(results)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(web_search.web_search("oMLX release notes"))

        assert result["query"] == "oMLX release notes"
        assert result["result_count"] == 1
        text = result["result_text"]
        assert text.startswith("• oMLX Release Notes — example.com\n  ")
        assert text.endswith("[https://example.com/omlx]")
        body_line = text.splitlines()[1].strip()
        assert len(body_line) <= 701
        assert body_line.endswith("…")

    def test_empty_results_returns_success_not_error(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "brave")
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        response = _brave_response([])
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(web_search.web_search("nothing found query"))
        assert result["result_text"] == "No results found."
        assert result["result_count"] == 0


class TestWebSearchBraveErrors:
    def test_missing_api_key_raises_clean_error_without_network_call(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "brave")
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        fake_get = AsyncMock()
        with patch.object(httpx.AsyncClient, "get", fake_get):
            with pytest.raises(ValueError, match="BRAVE_API_KEY not configured"):
                asyncio.run(web_search.web_search("anything"))
        fake_get.assert_not_called()

    def test_empty_string_api_key_treated_as_missing(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "brave")
        monkeypatch.setenv("BRAVE_API_KEY", "")
        with pytest.raises(ValueError, match="BRAVE_API_KEY not configured"):
            asyncio.run(web_search.web_search("anything"))

    def test_connection_error_wraps_as_clean_error(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "brave")
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectError("refused"))):
            with pytest.raises(ValueError, match="ERROR: web_search failed —"):
                asyncio.run(web_search.web_search("q"))

    def test_http_error_status_wraps_as_clean_error(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "brave")
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        response = _brave_response([], status_code=500)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            with pytest.raises(ValueError, match="ERROR: web_search failed —"):
                asyncio.run(web_search.web_search("q"))


# ---------------------------------------------------------------------------
# search_format — direct unit tests
# ---------------------------------------------------------------------------

class TestDeriveSourceFromUrl:
    def test_full_url_with_scheme(self):
        assert search_format.derive_source_from_url("https://www.ign.com/articles/x") == "ign.com"

    def test_bare_domain_no_scheme(self):
        assert search_format.derive_source_from_url("example.com/omlx") == "example.com"

    def test_strips_leading_www(self):
        assert search_format.derive_source_from_url("https://www.example.com/page") == "example.com"

    def test_no_www_left_untouched(self):
        assert search_format.derive_source_from_url("https://example.com") == "example.com"


class TestTruncateSummary:
    def test_short_summary_unchanged_no_ellipsis(self):
        assert search_format.truncate_summary("short text", 700) == "short text"

    def test_long_summary_cut_with_ellipsis(self):
        result = search_format.truncate_summary("word " * 200, 50)
        assert len(result) <= 51
        assert result.endswith("…")

    def test_exact_boundary_no_ellipsis(self):
        text = "x" * 700
        assert search_format.truncate_summary(text, 700) == text


class TestFormatResults:
    def test_empty_list_returns_empty_string(self):
        assert search_format.format_results([], per_result_budget=700) == ""

    def test_single_result_derives_source_from_url(self):
        result = search_format.SearchResult(
            title="Title", summary="Summary text.", url="https://example.com/page",
        )
        text = search_format.format_results([result], per_result_budget=700)
        assert text == "• Title — example.com\n  Summary text.\n  [https://example.com/page]"

    def test_explicit_source_and_published_at_in_header(self):
        result = search_format.SearchResult(
            title="Headline", summary="Body.", url="https://ign.com/x",
            source="IGN", published_at="2026-07-21",
        )
        text = search_format.format_results([result], per_result_budget=700)
        assert text.startswith("• Headline — IGN — 2026-07-21\n")

    def test_extra_field_appended_after_url_line(self):
        result = search_format.SearchResult(
            title="Headline", summary="Body.", url="https://ign.com/x",
            source="IGN", extra="Full article body excerpt.",
        )
        text = search_format.format_results([result], per_result_budget=700)
        assert text.endswith("[https://ign.com/x]\n  Full article body excerpt.")

    def test_multiple_results_joined_with_blank_line(self):
        results = [
            search_format.SearchResult(title="A", summary="a", url="https://a.com"),
            search_format.SearchResult(title="B", summary="b", url="https://b.com"),
        ]
        text = search_format.format_results(results, per_result_budget=700)
        assert "\n\n" in text
        assert text.count("•") == 2


# ---------------------------------------------------------------------------
# news_search.news_search — direct unit tests
# ---------------------------------------------------------------------------

def _newsapi_response(articles: list[dict], total_results: int | None = None, status: str = "ok") -> httpx.Response:
    request = httpx.Request("GET", news_search._NEWSAPI_ENDPOINT)
    payload = {
        "status":       status,
        "totalResults": total_results if total_results is not None else len(articles),
        "articles":     articles,
    }
    return httpx.Response(200, json=payload, request=request)


class TestNewsSearchFormat:
    def test_header_includes_source_and_date(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        articles = [{
            "title":       "Apple Vision Pro Gets a Major Update",
            "description": "A short description.",
            "source":      {"name": "IGN"},
            "publishedAt": "2026-07-21T13:45:00Z",
            "url":         "https://ign.com/articles/vision-pro-update",
        }]
        response = _newsapi_response(articles)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(news_search.news_search("vision pro"))

        assert result["is_miss"] is False
        text = result["result_text"]
        assert text.startswith("• Apple Vision Pro Gets a Major Update — IGN — 2026-07-21\n")
        assert text.endswith("[https://ign.com/articles/vision-pro-update]")

    def test_summary_truncated_past_600_chars_with_ellipsis(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        articles = [{
            "title":       "Long Article",
            "description": "x" * 650,
            "source":      {"name": "IGN"},
            "publishedAt": "2026-07-21T13:45:00Z",
            "url":         "https://ign.com/x",
        }]
        response = _newsapi_response(articles)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(news_search.news_search("long"))

        body_line = result["result_text"].splitlines()[1].strip()
        assert len(body_line) <= 601
        assert body_line.endswith("…")

    def test_pinned_article_appends_content_line(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        articles = [{
            "title":       "Pinned Story",
            "description": "Short description.",
            "source":      {"name": "IGN"},
            "publishedAt": "2026-07-21T13:45:00Z",
            "url":         "https://ign.com/pinned",
            "content":     "Full excerpt of the pinned article body.",
        }]
        response = _newsapi_response(articles)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(news_search.news_search("pinned story", url="https://ign.com/pinned"))

        text = result["result_text"]
        assert text.endswith("Full excerpt of the pinned article body.")

    def test_non_pinned_multi_result_has_no_content_line(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        articles = [{
            "title":       "Story One",
            "description": "Description one.",
            "source":      {"name": "IGN"},
            "publishedAt": "2026-07-21T13:45:00Z",
            "url":         "https://ign.com/one",
            "content":     "This should not appear — not pinned.",
        }]
        response = _newsapi_response(articles)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(news_search.news_search("story one"))

        assert "This should not appear" not in result["result_text"]

    def test_miss_returns_empty_result_text(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        response = _newsapi_response([], total_results=0)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(news_search.news_search("nothing"))

        assert result["is_miss"] is True
        assert result["result_text"] == ""


# ---------------------------------------------------------------------------
# github.github_search / github.github_read — direct unit tests
# ---------------------------------------------------------------------------

def _github_search_response(items: list[dict], status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", f"{github._GITHUB_API_BASE}/search/repositories")
    return httpx.Response(status_code, json={"items": items}, request=request)


def _github_raw_response(text: str, content_type: str = "text/plain; charset=utf-8", status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", f"{github._GITHUB_API_BASE}/repos/o/r/readme")
    return httpx.Response(
        status_code, text=text, headers={"content-type": content_type}, request=request,
    )


def _github_dir_response(entries: list[dict], status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", f"{github._GITHUB_API_BASE}/repos/o/r/contents/src")
    return httpx.Response(status_code, json=entries, request=request)


class TestGithubSearchSuccess:
    def test_repositories_formatted_with_stars_and_url(self):
        items = [{
            "full_name":         "anthropics/claude-code",
            "description":       "Claude Code CLI",
            "html_url":          "https://github.com/anthropics/claude-code",
            "stargazers_count":  42,
        }]
        response = _github_search_response(items)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(github.github_search("claude code"))

        assert result["is_miss"] is False
        assert result["result_count"] == 1
        text = result["result_text"]
        assert text.startswith("• anthropics/claude-code — GitHub\n")
        assert "[https://github.com/anthropics/claude-code]" in text
        assert text.endswith("⭐ 42")

    def test_code_kind_formats_path_and_repo(self):
        items = [{
            "path":        "src/main.py",
            "html_url":    "https://github.com/o/r/blob/main/src/main.py",
            "repository":  {"full_name": "o/r"},
        }]
        response = _github_search_response(items)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(github.github_search("main", kind="code"))

        assert result["result_count"] == 1
        assert "in o/r" in result["result_text"]

    def test_no_items_is_miss(self):
        response = _github_search_response([])
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(github.github_search("nothing found query"))
        assert result["is_miss"] is True
        assert result["result_text"] == ""

    def test_unknown_kind_raises_without_network_call(self):
        fake_get = AsyncMock()
        with patch.object(httpx.AsyncClient, "get", fake_get):
            with pytest.raises(ValueError, match="ERROR: unknown github_search kind"):
                asyncio.run(github.github_search("q", kind="issues"))
        fake_get.assert_not_called()


class TestGithubSearchErrors:
    def test_transport_error_raises_clean_message(self):
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectError("refused"))):
            with pytest.raises(ValueError, match="ERROR: github_search failed —"):
                asyncio.run(github.github_search("q"))

    def test_rate_limited_raises_clean_message(self):
        response = httpx.Response(
            403, json={"message": "API rate limit exceeded"},
            request=httpx.Request("GET", f"{github._GITHUB_API_BASE}/search/repositories"),
        )
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            with pytest.raises(ValueError, match="ERROR: github_search failed —"):
                asyncio.run(github.github_search("q"))


class TestGithubReadSuccess:
    def test_readme_returns_raw_text(self):
        response = _github_raw_response("# Hello\n\nThis is a README.")
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(github.github_read("o", "r"))
        assert result["kind"] == "file"
        assert result["content"] == "# Hello\n\nThis is a README."
        assert result["truncated"] is False

    def test_file_content_truncated_past_budget(self):
        long_text = "x" * (github._GITHUB_CONTENT_CHARS + 500)
        response = _github_raw_response(long_text)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(github.github_read("o", "r", path="big.txt"))
        assert result["truncated"] is True
        assert len(result["content"]) == github._GITHUB_CONTENT_CHARS + 1  # + ellipsis
        assert result["content"].endswith("…")

    def test_directory_listing_formats_names_by_type(self):
        entries = [
            {"name": "src", "type": "dir"},
            {"name": "README.md", "type": "file"},
        ]
        response = _github_dir_response(entries)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(github.github_read("o", "r", path="."))
        assert result["kind"] == "directory"
        assert "📁 src" in result["content"]
        assert "📄 README.md" in result["content"]
        assert result["truncated"] is False


class TestGithubReadErrors:
    def test_not_found_raises_clean_message(self):
        response = httpx.Response(
            404, json={"message": "Not Found"},
            request=httpx.Request("GET", f"{github._GITHUB_API_BASE}/repos/o/r/contents/missing.py"),
        )
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            with pytest.raises(ValueError, match="ERROR: github_read — o/r/missing.py not found"):
                asyncio.run(github.github_read("o", "r", path="missing.py"))

    def test_transport_error_raises_clean_message(self):
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectError("refused"))):
            with pytest.raises(ValueError, match="ERROR: github_read failed —"):
                asyncio.run(github.github_read("o", "r"))


# ---------------------------------------------------------------------------
# github.github_release — direct unit tests
# ---------------------------------------------------------------------------

def _github_release_response(data: dict, status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", f"{github._GITHUB_API_BASE}/repos/o/r/releases/latest")
    return httpx.Response(status_code, json=data, request=request)


class TestGithubReleaseSuccess:
    def test_latest_release_returns_notes(self):
        data = {
            "tag_name": "v1.2.0", "name": "v1.2.0", "body": "Bug fixes and improvements.",
            "published_at": "2026-07-22T00:00:00Z",
            "html_url": "https://github.com/o/r/releases/tag/v1.2.0",
        }
        response = _github_release_response(data)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(github.github_release("o", "r"))

        assert result["tag_name"] == "v1.2.0"
        assert result["body"] == "Bug fixes and improvements."
        assert result["truncated"] is False
        assert result["html_url"] == "https://github.com/o/r/releases/tag/v1.2.0"

    def test_exact_tag_match_hits_tags_endpoint_directly(self):
        data = {"tag_name": "v0.5.3", "name": "0.5.3", "body": "notes", "published_at": "x", "html_url": "y"}
        response = _github_release_response(data)
        fake_get = AsyncMock(return_value=response)
        with patch.object(httpx.AsyncClient, "get", fake_get):
            result = asyncio.run(github.github_release("o", "r", tag="v0.5.3"))

        assert result["tag_name"] == "v0.5.3"
        assert fake_get.await_count == 1
        called_url = fake_get.await_args.args[0]
        assert called_url.endswith("/repos/o/r/releases/tags/v0.5.3")

    def test_bare_version_retries_with_v_prefix_on_404(self):
        not_found = httpx.Response(
            404, json={"message": "Not Found"},
            request=httpx.Request("GET", f"{github._GITHUB_API_BASE}/repos/o/r/releases/tags/0.5.3"),
        )
        found = _github_release_response({
            "tag_name": "v0.5.3", "name": "0.5.3", "body": "notes",
            "published_at": "2026-07-22T00:00:00Z", "html_url": "y",
        })
        fake_get = AsyncMock(side_effect=[not_found, found])
        with patch.object(httpx.AsyncClient, "get", fake_get):
            result = asyncio.run(github.github_release("o", "r", tag="0.5.3"))

        assert result["tag_name"] == "v0.5.3"
        assert fake_get.await_count == 2
        second_url = fake_get.await_args_list[1].args[0]
        assert second_url.endswith("/repos/o/r/releases/tags/v0.5.3")

    def test_v_prefixed_tag_retries_without_v_on_404(self):
        not_found = httpx.Response(
            404, json={"message": "Not Found"},
            request=httpx.Request("GET", f"{github._GITHUB_API_BASE}/repos/o/r/releases/tags/vfoo"),
        )
        found = _github_release_response({
            "tag_name": "foo", "name": "foo", "body": "notes", "published_at": "x", "html_url": "y",
        })
        fake_get = AsyncMock(side_effect=[not_found, found])
        with patch.object(httpx.AsyncClient, "get", fake_get):
            result = asyncio.run(github.github_release("o", "r", tag="vfoo"))

        assert result["tag_name"] == "foo"
        second_url = fake_get.await_args_list[1].args[0]
        assert second_url.endswith("/repos/o/r/releases/tags/foo")

    def test_body_truncated_past_budget(self):
        long_body = "x" * (github._GITHUB_RELEASE_BODY_CHARS + 500)
        data = {"tag_name": "v1", "name": "v1", "body": long_body, "published_at": "x", "html_url": "y"}
        response = _github_release_response(data)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(github.github_release("o", "r"))

        assert result["truncated"] is True
        assert len(result["body"]) == github._GITHUB_RELEASE_BODY_CHARS + 1  # + ellipsis
        assert result["body"].endswith("…")


class TestGithubReleaseErrors:
    def test_no_releases_raises_clean_message(self):
        response = httpx.Response(
            404, json={"message": "Not Found"},
            request=httpx.Request("GET", f"{github._GITHUB_API_BASE}/repos/o/r/releases/latest"),
        )
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            with pytest.raises(ValueError, match="ERROR: github_release — o/r has no releases"):
                asyncio.run(github.github_release("o", "r"))

    def test_tag_not_found_after_both_attempts_raises_clean_message(self):
        not_found = httpx.Response(
            404, json={"message": "Not Found"},
            request=httpx.Request("GET", f"{github._GITHUB_API_BASE}/repos/o/r/releases/tags/9.9.9"),
        )
        fake_get = AsyncMock(return_value=not_found)
        with patch.object(httpx.AsyncClient, "get", fake_get):
            with pytest.raises(ValueError, match="ERROR: github_release — o/r tag '9.9.9' not found"):
                asyncio.run(github.github_release("o", "r", tag="9.9.9"))
        assert fake_get.await_count == 2

    def test_transport_error_raises_clean_message(self):
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectError("refused"))):
            with pytest.raises(ValueError, match="ERROR: github_release failed —"):
                asyncio.run(github.github_release("o", "r"))

    def test_rate_limited_raises_clean_message(self):
        response = httpx.Response(
            403, json={"message": "API rate limit exceeded"},
            request=httpx.Request("GET", f"{github._GITHUB_API_BASE}/repos/o/r/releases/latest"),
        )
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            with pytest.raises(ValueError, match="ERROR: github_release failed —"):
                asyncio.run(github.github_release("o", "r"))


# ---------------------------------------------------------------------------
# hacker_news.hacker_news_search — direct unit tests
# ---------------------------------------------------------------------------

def _hn_search_response(hits: list[dict], status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", hacker_news._HN_ALGOLIA_SEARCH_ENDPOINT)
    return httpx.Response(status_code, json={"hits": hits}, request=request)


def _hn_item_response(children: list[dict], object_id: str = "1", status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", hacker_news._HN_ALGOLIA_ITEM_ENDPOINT.format(object_id))
    return httpx.Response(status_code, json={"children": children}, request=request)


class TestHackerNewsSearchSuccess:
    def test_story_with_url_formatted_with_points_and_comments(self):
        hits = [{
            "objectID": "12345", "title": "A great story",
            "url": "https://example.com/article", "points": 200, "num_comments": 42,
        }]
        response = _hn_search_response(hits)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(hacker_news.hacker_news_search("great story"))

        assert result["is_miss"] is False
        assert result["result_count"] == 1
        text = result["result_text"]
        assert text.startswith("• A great story — Hacker News\n")
        assert "[https://example.com/article]" in text
        assert text.endswith("200 points · 42 comments")

    def test_self_post_falls_back_to_hn_discussion_url(self):
        hits = [{"objectID": "999", "title": "Ask HN: something", "url": None, "points": 5, "num_comments": 1}]
        response = _hn_search_response(hits)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(hacker_news.hacker_news_search("ask hn"))

        assert "[https://news.ycombinator.com/item?id=999]" in result["result_text"]

    def test_no_hits_is_miss(self):
        response = _hn_search_response([])
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(hacker_news.hacker_news_search("nothing found query"))
        assert result["is_miss"] is True
        assert result["result_text"] == ""

    def test_pinned_url_filters_to_matching_hit_only(self):
        hits = [
            {"objectID": "1", "title": "Story one", "url": "https://example.com/one", "points": 10, "num_comments": 1},
            {"objectID": "2", "title": "Story two", "url": "https://example.com/two", "points": 20, "num_comments": 2},
        ]
        response = _hn_search_response(hits)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(hacker_news.hacker_news_search("story", url="https://example.com/two"))

        assert result["result_count"] == 1
        assert "Story two" in result["result_text"]
        assert "Story one" not in result["result_text"]

    def test_pinned_url_with_no_match_falls_back_to_unfiltered(self):
        hits = [{"objectID": "1", "title": "Story one", "url": "https://example.com/one", "points": 10, "num_comments": 1}]
        response = _hn_search_response(hits)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(hacker_news.hacker_news_search("story", url="https://example.com/nonexistent"))

        assert result["result_count"] == 1
        assert "Story one" in result["result_text"]


class TestHackerNewsSearchErrors:
    def test_transport_error_raises_clean_message(self):
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectError("refused"))):
            with pytest.raises(ValueError, match="ERROR: hacker_news_search failed —"):
                asyncio.run(hacker_news.hacker_news_search("q"))

    def test_http_error_raises_clean_message(self):
        response = httpx.Response(
            500, json={"error": "boom"},
            request=httpx.Request("GET", hacker_news._HN_ALGOLIA_SEARCH_ENDPOINT),
        )
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            with pytest.raises(ValueError, match="ERROR: hacker_news_search failed —"):
                asyncio.run(hacker_news.hacker_news_search("q"))


class TestCleanCommentText:
    def test_strips_tags_and_unescapes_entities(self):
        raw = "&gt; quoted bit<p>Actual reply &amp; more &quot;text&quot;.</p>"
        assert hacker_news._clean_comment_text(raw) == '> quoted bit Actual reply & more "text".'

    def test_collapses_whitespace(self):
        raw = "line one\n\n  line   two"
        assert hacker_news._clean_comment_text(raw) == "line one line two"


class TestFetchTopComments:
    def test_returns_author_and_cleaned_text(self):
        children = [
            {"author": "alice", "text": "<p>First point.</p>"},
            {"author": "bob", "text": "Second point &amp; more."},
        ]
        response = _hn_item_response(children)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(hacker_news.fetch_top_comments("1"))

        assert result == [
            {"author": "alice", "text": "First point."},
            {"author": "bob", "text": "Second point & more."},
        ]

    def test_deleted_comments_skipped(self):
        children = [
            {"author": None, "text": None},
            {"author": "bob", "text": "Still here."},
        ]
        response = _hn_item_response(children)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(hacker_news.fetch_top_comments("1"))
        assert result == [{"author": "bob", "text": "Still here."}]

    def test_respects_count_argument(self):
        children = [{"author": f"user{i}", "text": f"comment {i}"} for i in range(10)]
        response = _hn_item_response(children)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(hacker_news.fetch_top_comments("1", count=2))
        assert len(result) == 2

    def test_transport_error_raises(self):
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectError("refused"))):
            with pytest.raises(httpx.ConnectError):
                asyncio.run(hacker_news.fetch_top_comments("1"))


class TestHackerNewsSearchComments:
    def test_pinned_story_includes_real_comment_text(self):
        hits = [{"objectID": "1", "title": "Story", "url": "https://example.com/one", "points": 10, "num_comments": 2}]
        search_response = _hn_search_response(hits)
        item_response = _hn_item_response([
            {"author": "alice", "text": "<p>This is a great point.</p>"},
            {"author": "bob", "text": "I disagree &amp; here is why."},
        ])
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=[search_response, item_response])):
            result = asyncio.run(hacker_news.hacker_news_search("story", url="https://example.com/one"))

        text = result["result_text"]
        assert "Top comments:" in text
        assert "alice: This is a great point." in text
        assert "bob: I disagree & here is why." in text

    def test_unpinned_search_never_fetches_comments(self):
        hits = [{"objectID": "1", "title": "Story", "url": "https://example.com/one", "points": 10, "num_comments": 2}]
        response = _hn_search_response(hits)
        fake_get = AsyncMock(return_value=response)
        with patch.object(httpx.AsyncClient, "get", fake_get):
            result = asyncio.run(hacker_news.hacker_news_search("story"))

        assert "Top comments:" not in result["result_text"]
        fake_get.assert_awaited_once()

    def test_comment_fetch_failure_degrades_gracefully_without_raising(self):
        hits = [{"objectID": "1", "title": "Story", "url": "https://example.com/one", "points": 10, "num_comments": 2}]
        search_response = _hn_search_response(hits)
        with patch.object(
            httpx.AsyncClient, "get",
            AsyncMock(side_effect=[search_response, httpx.ConnectError("refused")]),
        ):
            result = asyncio.run(hacker_news.hacker_news_search("story", url="https://example.com/one"))

        assert result["is_miss"] is False
        assert "Top comments:" not in result["result_text"]
        assert "10 points · 2 comments" in result["result_text"]


# ---------------------------------------------------------------------------
# chart.generate_chart — direct unit tests
# ---------------------------------------------------------------------------

_BAR_ARGS = dict(
    chart_type = "bar",
    labels     = ["apples", "oranges", "bananas"],
    datasets   = [{"label": "Fruit count", "data": [5, 3, 7]}],
    title      = "Fruit Inventory",
)

_LINE_ARGS = dict(
    chart_type = "line",
    labels     = ["Jan", "Feb", "Mar"],
    datasets   = [{"label": "Revenue", "data": [10, 20, 15]}],
    title      = "",
)

_PIE_ARGS = dict(
    chart_type = "pie",
    labels     = ["A", "B", "C"],
    datasets   = [{"label": "Share", "data": [1, 2, 3]}],
    title      = "Distribution",
)


class TestGenerateChart:
    def test_bar_chart_success(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        result = chart.generate_chart(**_BAR_ARGS)

        assert result["summary"] == "Generated bar chart: Fruit Inventory"
        assert result["png_path"].startswith("charts/") and result["png_path"].endswith(".png")
        assert result["chart_config"] == {
            "chart_type": "bar",
            "title":      "Fruit Inventory",
            "labels":     _BAR_ARGS["labels"],
            "datasets":   _BAR_ARGS["datasets"],
        }
        png_file = tmp_path / result["png_path"]
        assert png_file.exists()
        assert png_file.stat().st_size > 0

    def test_line_chart_success(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        result = chart.generate_chart(**_LINE_ARGS)
        assert result["summary"] == "Generated line chart: Jan, Feb, Mar"
        assert (tmp_path / result["png_path"]).exists()

    def test_pie_chart_success(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        result = chart.generate_chart(**_PIE_ARGS)
        assert result["summary"] == "Generated pie chart: Distribution"
        assert (tmp_path / result["png_path"]).exists()

    def test_invalid_chart_type_raises(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError, match="ERROR:.*chart_type invalid or missing"):
            chart.generate_chart(chart_type="scatter", labels=["a"], datasets=[{"label": "x", "data": [1]}])

    def test_empty_labels_raises(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError, match="ERROR:.*labels is an empty array"):
            chart.generate_chart(chart_type="bar", labels=[], datasets=[{"label": "x", "data": []}])

    def test_empty_datasets_raises(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError, match="ERROR:.*datasets missing, not an array, or empty"):
            chart.generate_chart(chart_type="bar", labels=["a"], datasets=[])

    def test_dataset_length_mismatch_raises(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError, match="ERROR:.*data length .* != labels length"):
            chart.generate_chart(
                chart_type = "bar",
                labels     = ["a", "b"],
                datasets   = [{"label": "x", "data": [1]}],
            )

    def test_pie_with_multiple_datasets_raises(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError, match="ERROR:.*pie chart_type should have exactly one dataset"):
            chart.generate_chart(
                chart_type = "pie",
                labels     = ["a", "b"],
                datasets   = [
                    {"label": "x", "data": [1, 2]},
                    {"label": "y", "data": [3, 4]},
                ],
            )

    def test_no_chart_written_on_validation_failure(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError):
            chart.generate_chart(chart_type="bad", labels=[], datasets=[])
        charts_dir = tmp_path / "charts"
        assert not charts_dir.exists() or not list(charts_dir.iterdir())


# ---------------------------------------------------------------------------
# ocr.extract_text — direct unit tests
#
# _ocr_image_bytes (the actual PyObjC/Vision call) is mocked throughout —
# Vision can't run in CI the way EmbeddingEngine.embed is already mocked in
# this suite. Live-verified separately against the real Vision framework and
# real PyMuPDF during development (see docs/architecture/22-local-ocr-service.md).
# PDF fixtures are built with real PyMuPDF (fast, deterministic, already a
# hard dependency) rather than mocking fitz's API surface.
# ---------------------------------------------------------------------------

def _write_image(tmp_path: Path, name: str = "photo.png") -> None:
    (tmp_path / name).write_bytes(b"not-real-png-bytes-ocr-is-mocked-anyway")


def _write_text_layer_pdf(tmp_path: Path, name: str = "doc.pdf", pages: int = 1) -> None:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=300, height=100)
        page.insert_text((20, 50), "This page has a real embedded text layer.")
    doc.save(str(tmp_path / name))
    doc.close()


def _write_blank_pdf(tmp_path: Path, name: str = "scanned.pdf", pages: int = 1) -> None:
    """No text layer at all — triggers the rasterize+OCR fallback path.
    get_pixmap() rasterizes fine on a blank page, and _ocr_image_bytes is
    mocked in every test that uses this, so the actual pixel content
    (or lack of it) never matters."""
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page(width=300, height=100)
    doc.save(str(tmp_path / name))
    doc.close()


class TestOcrExtractImages:
    def test_success_returns_ocr_text(self, tmp_path: Path):
        ocr.set_upload_root(tmp_path)
        _write_image(tmp_path)
        with patch.object(ocr, "_is_apple_silicon", return_value=True), \
             patch.object(ocr, "_ocr_image_bytes", return_value="Recognized text"):
            result = ocr.extract_text("photo.png", "image/png")
        assert result == "Recognized text"

    def test_non_apple_silicon_platform_raises(self, tmp_path: Path):
        ocr.set_upload_root(tmp_path)
        _write_image(tmp_path)
        with patch.object(ocr, "_is_apple_silicon", return_value=False):
            with pytest.raises(ValueError, match="requires macOS on Apple Silicon"):
                ocr.extract_text("photo.png", "image/png")

    def test_missing_file_raises(self, tmp_path: Path):
        ocr.set_upload_root(tmp_path)
        with patch.object(ocr, "_is_apple_silicon", return_value=True):
            with pytest.raises(ValueError, match="file not found"):
                ocr.extract_text("ghost.png", "image/png")

    def test_unsupported_mime_type_raises(self, tmp_path: Path):
        ocr.set_upload_root(tmp_path)
        _write_image(tmp_path)
        with patch.object(ocr, "_is_apple_silicon", return_value=True):
            with pytest.raises(ValueError, match="unsupported mime_type"):
                ocr.extract_text("photo.png", "application/octet-stream")

    def test_near_empty_result_raises(self, tmp_path: Path):
        ocr.set_upload_root(tmp_path)
        _write_image(tmp_path)
        with patch.object(ocr, "_is_apple_silicon", return_value=True), \
             patch.object(ocr, "_ocr_image_bytes", return_value="   "):
            with pytest.raises(ValueError, match="no readable text detected"):
                ocr.extract_text("photo.png", "image/png")

    def test_path_traversal_blocked(self, tmp_path: Path):
        ocr.set_upload_root(tmp_path)
        with patch.object(ocr, "_is_apple_silicon", return_value=True):
            with pytest.raises(ValueError, match="path traversal"):
                ocr.extract_text("../../etc/passwd", "image/png")

    def test_heic_uses_same_code_path_as_any_other_image(self, tmp_path: Path):
        """No format-specific branching — HEIC support falls out of always
        using VNImageRequestHandler's data-based initializer (see ocr.py's
        module docstring), not a dedicated code path to test separately."""
        ocr.set_upload_root(tmp_path)
        (tmp_path / "photo.heic").write_bytes(b"not-real-heic-bytes")
        with patch.object(ocr, "_is_apple_silicon", return_value=True), \
             patch.object(ocr, "_ocr_image_bytes", return_value="Recognized HEIC text"):
            result = ocr.extract_text("photo.heic", "image/heic")
        assert result == "Recognized HEIC text"


class TestOcrExtractPdf:
    def test_text_layer_fast_path_skips_ocr(self, tmp_path: Path):
        ocr.set_upload_root(tmp_path)
        _write_text_layer_pdf(tmp_path)
        with patch.object(ocr, "_is_apple_silicon", return_value=True), \
             patch.object(ocr, "_ocr_image_bytes") as mock_ocr:
            result = ocr.extract_text("doc.pdf", "application/pdf")
        assert "real embedded text layer" in result
        mock_ocr.assert_not_called()

    def test_blank_pdf_falls_back_to_ocr(self, tmp_path: Path):
        ocr.set_upload_root(tmp_path)
        _write_blank_pdf(tmp_path)
        with patch.object(ocr, "_is_apple_silicon", return_value=True), \
             patch.object(ocr, "_ocr_image_bytes", return_value="Scanned page text"):
            result = ocr.extract_text("scanned.pdf", "application/pdf")
        assert "--- page 1 ---" in result
        assert "Scanned page text" in result

    def test_multi_page_blank_pdf_ocrs_each_page(self, tmp_path: Path):
        ocr.set_upload_root(tmp_path)
        _write_blank_pdf(tmp_path, pages=2)
        with patch.object(ocr, "_is_apple_silicon", return_value=True), \
             patch.object(ocr, "_ocr_image_bytes", side_effect=["Page one", "Page two"]):
            result = ocr.extract_text("scanned.pdf", "application/pdf")
        assert "--- page 1 ---\nPage one" in result
        assert "--- page 2 ---\nPage two" in result

    def test_page_cap_exceeded_raises(self, tmp_path: Path):
        ocr.set_upload_root(tmp_path)
        _write_blank_pdf(tmp_path, pages=3)
        with patch.object(ocr, "_is_apple_silicon", return_value=True):
            with pytest.raises(ValueError, match="exceeding the 2-page OCR limit"):
                ocr.extract_text("scanned.pdf", "application/pdf", max_pdf_pages=2)

    def test_explicit_zero_page_cap_is_not_treated_as_unset(self, tmp_path: Path):
        """Regression test — max_pdf_pages=0 was previously swallowed by an
        `or get_max_pdf_pages()` falsy-zero bug (0 is falsy in Python) that
        silently fell back to the default cap instead of rejecting. Found
        via live verification during Phase 1 build, fixed with an explicit
        `is not None` check."""
        ocr.set_upload_root(tmp_path)
        _write_blank_pdf(tmp_path, pages=1)
        with patch.object(ocr, "_is_apple_silicon", return_value=True):
            with pytest.raises(ValueError, match="exceeding the 0-page OCR limit"):
                ocr.extract_text("scanned.pdf", "application/pdf", max_pdf_pages=0)

    def test_text_layer_pdf_ignores_page_cap(self, tmp_path: Path):
        """A real text layer is read directly regardless of page count —
        the cap only bounds the rasterize+OCR fallback path."""
        ocr.set_upload_root(tmp_path)
        _write_text_layer_pdf(tmp_path, pages=5)
        with patch.object(ocr, "_is_apple_silicon", return_value=True), \
             patch.object(ocr, "_ocr_image_bytes") as mock_ocr:
            result = ocr.extract_text("doc.pdf", "application/pdf", max_pdf_pages=1)
        assert "real embedded text layer" in result
        mock_ocr.assert_not_called()

    def test_near_empty_ocr_result_raises(self, tmp_path: Path):
        ocr.set_upload_root(tmp_path)
        _write_blank_pdf(tmp_path)
        with patch.object(ocr, "_is_apple_silicon", return_value=True), \
             patch.object(ocr, "_ocr_image_bytes", return_value=""):
            with pytest.raises(ValueError, match="no readable text detected"):
                ocr.extract_text("scanned.pdf", "application/pdf")


class TestOcrGetMaxPdfPages:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("LOCALIST_OCR_MAX_PDF_PAGES", raising=False)
        assert ocr.get_max_pdf_pages() == ocr._DEFAULT_MAX_PDF_PAGES

    def test_reads_valid_override(self, monkeypatch):
        monkeypatch.setenv("LOCALIST_OCR_MAX_PDF_PAGES", "5")
        assert ocr.get_max_pdf_pages() == 5

    def test_invalid_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("LOCALIST_OCR_MAX_PDF_PAGES", "not-a-number")
        assert ocr.get_max_pdf_pages() == ocr._DEFAULT_MAX_PDF_PAGES


# ---------------------------------------------------------------------------
# MCP tool wiring — in-process client session (no network)
# ---------------------------------------------------------------------------

async def _call_tool(name: str, arguments: dict) -> tuple[str, bool]:
    async with create_connected_server_and_client_session(mcp_app) as session:
        result = await session.call_tool(name, arguments)
        text = "\n".join(b.text for b in result.content if hasattr(b, "text"))
        return text, result.isError


class TestMCPToolsInProcess:
    def test_read_file_tool_success(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        (tmp_path / "notes.md").write_text("hi from mcp", encoding="utf-8")
        text, is_error = asyncio.run(_call_tool("read_file", {"path": "notes.md"}))
        assert is_error is False
        assert text == "hi from mcp"

    def test_read_file_tool_error_surfaces_as_is_error(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        text, is_error = asyncio.run(_call_tool("read_file", {"path": "ghost.md"}))
        assert is_error is True
        assert "file not found" in text

    def test_write_file_tool_success(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        text, is_error = asyncio.run(
            _call_tool("write_file", {"path": "out.md", "content": "written via mcp"})
        )
        assert is_error is False
        assert "OK: wrote" in text
        assert (tmp_path / "out.md").read_text(encoding="utf-8") == "written via mcp"

    def test_append_file_tool_success(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        (tmp_path / "log.txt").write_text("first\n", encoding="utf-8")
        text, is_error = asyncio.run(
            _call_tool("append_file", {"path": "log.txt", "content": "second\n"})
        )
        assert is_error is False
        assert (tmp_path / "log.txt").read_text(encoding="utf-8") == "first\nsecond\n"

    def test_path_traversal_blocked_over_mcp(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        text, is_error = asyncio.run(
            _call_tool("read_file", {"path": "../../etc/passwd"})
        )
        assert is_error is True
        assert "path traversal" in text

    def test_fetch_url_tool_success(self):
        with patch.object(url_fetch, "_fetch", AsyncMock(return_value=_raw_response(_SAMPLE_ARTICLE_HTML))):
            text, is_error = asyncio.run(
                _call_tool("fetch_url", {"url": "https://example.com/article"})
            )
        assert is_error is False
        data = json.loads(text)
        assert data["title"] == "Test Article Title"
        assert data["word_count"] > 0

    def test_fetch_url_tool_error_surfaces_as_is_error(self):
        with patch.object(url_fetch, "_fetch", AsyncMock(side_effect=httpx.ConnectError("refused"))):
            text, is_error = asyncio.run(
                _call_tool("fetch_url", {"url": "https://unreachable.example"})
            )
        assert is_error is True
        assert "connection_error" in text

    def test_web_search_tool_success(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        response = _langsearch_response([{"name": "T", "snippet": "s", "url": "https://e.com"}])
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
            text, is_error = asyncio.run(_call_tool("web_search", {"query": "test query"}))
        assert is_error is False
        data = json.loads(text)
        assert data["result_count"] == 1

    def test_web_search_tool_missing_api_key_surfaces_as_is_error(self, monkeypatch):
        monkeypatch.delenv("LANGSEARCH_API_KEY", raising=False)
        text, is_error = asyncio.run(_call_tool("web_search", {"query": "test query"}))
        assert is_error is True
        assert "LANGSEARCH_API_KEY not configured" in text

    def test_generate_chart_tool_success(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        text, is_error = asyncio.run(_call_tool("generate_chart", _BAR_ARGS))
        assert is_error is False
        data = json.loads(text)
        assert data["summary"] == "Generated bar chart: Fruit Inventory"
        assert (tmp_path / data["png_path"]).exists()

    def test_generate_chart_tool_error_surfaces_as_is_error(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        text, is_error = asyncio.run(
            _call_tool("generate_chart", {"chart_type": "scatter", "labels": ["a"], "datasets": [{"label": "x", "data": [1]}]})
        )
        assert is_error is True
        assert "chart_type invalid or missing" in text

    def test_github_search_tool_success(self):
        items = [{
            "full_name": "o/r", "description": "d",
            "html_url": "https://github.com/o/r", "stargazers_count": 1,
        }]
        response = _github_search_response(items)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            text, is_error = asyncio.run(_call_tool("github_search", {"query": "test query"}))
        assert is_error is False
        data = json.loads(text)
        assert data["result_count"] == 1

    def test_github_search_tool_unknown_kind_surfaces_as_is_error(self):
        text, is_error = asyncio.run(_call_tool("github_search", {"query": "q", "kind": "issues"}))
        assert is_error is True
        assert "unknown github_search kind" in text

    def test_github_read_tool_success(self):
        response = _github_raw_response("readme text")
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            text, is_error = asyncio.run(_call_tool("github_read", {"owner": "o", "repo": "r"}))
        assert is_error is False
        data = json.loads(text)
        assert data["content"] == "readme text"

    def test_github_read_tool_not_found_surfaces_as_is_error(self):
        response = httpx.Response(
            404, json={"message": "Not Found"},
            request=httpx.Request("GET", f"{github._GITHUB_API_BASE}/repos/o/r/readme"),
        )
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            text, is_error = asyncio.run(_call_tool("github_read", {"owner": "o", "repo": "r"}))
        assert is_error is True
        assert "not found" in text

    def test_github_release_tool_success(self):
        data = {
            "tag_name": "v1.0", "name": "v1.0", "body": "notes",
            "published_at": "x", "html_url": "y",
        }
        response = _github_release_response(data)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            text, is_error = asyncio.run(_call_tool("github_release", {"owner": "o", "repo": "r"}))
        assert is_error is False
        data_out = json.loads(text)
        assert data_out["tag_name"] == "v1.0"

    def test_github_release_tool_no_releases_surfaces_as_is_error(self):
        response = httpx.Response(
            404, json={"message": "Not Found"},
            request=httpx.Request("GET", f"{github._GITHUB_API_BASE}/repos/o/r/releases/latest"),
        )
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            text, is_error = asyncio.run(_call_tool("github_release", {"owner": "o", "repo": "r"}))
        assert is_error is True
        assert "has no releases" in text

    def test_hacker_news_search_tool_success(self):
        hits = [{"objectID": "1", "title": "t", "url": "https://e.com", "points": 1, "num_comments": 0}]
        response = _hn_search_response(hits)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            text, is_error = asyncio.run(_call_tool("hacker_news_search", {"query": "test query"}))
        assert is_error is False
        data = json.loads(text)
        assert data["result_count"] == 1

    def test_hacker_news_search_tool_miss_surfaces_empty_result(self):
        response = _hn_search_response([])
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            text, is_error = asyncio.run(_call_tool("hacker_news_search", {"query": "nothing found query"}))
        assert is_error is False
        data = json.loads(text)
        assert data["is_miss"] is True

    def test_hacker_news_search_tool_url_pins_result(self):
        hits = [
            {"objectID": "1", "title": "one", "url": "https://e.com/one", "points": 1, "num_comments": 0},
            {"objectID": "2", "title": "two", "url": "https://e.com/two", "points": 2, "num_comments": 0},
        ]
        response = _hn_search_response(hits)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            text, is_error = asyncio.run(
                _call_tool("hacker_news_search", {"query": "test query", "url": "https://e.com/two"})
            )
        assert is_error is False
        data = json.loads(text)
        assert data["result_count"] == 1
        assert "two" in data["result_text"]
        assert "one" not in data["result_text"]

    def test_ocr_extract_tool_success(self, tmp_path: Path):
        ocr.set_upload_root(tmp_path)
        _write_image(tmp_path)
        with patch.object(ocr, "_is_apple_silicon", return_value=True), \
             patch.object(ocr, "_ocr_image_bytes", return_value="Recognized via MCP"):
            text, is_error = asyncio.run(
                _call_tool("ocr_extract", {"path": "photo.png", "mime_type": "image/png"})
            )
        assert is_error is False
        assert text == "Recognized via MCP"

    def test_ocr_extract_tool_error_surfaces_as_is_error(self, tmp_path: Path):
        ocr.set_upload_root(tmp_path)
        with patch.object(ocr, "_is_apple_silicon", return_value=True):
            text, is_error = asyncio.run(
                _call_tool("ocr_extract", {"path": "ghost.png", "mime_type": "image/png"})
            )
        assert is_error is True
        assert "file not found" in text
