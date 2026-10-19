"""Tools: plain Python functions that agents call to reach the outside world.

A tool is NOT an agent. It has no reasoning and makes no routing decisions.
The researcher node decides *when* to call it and *what* to do with the result.

Contract of web_search(): it always returns a ToolCall record, never raises.
On failure it returns ok=False with an error message and zero results, so the
caller can never mistake a failed search for a successful one.
"""

import ipaddress
import logging
import re
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser

import telemetry
from state import SearchHit, ToolCall
from telemetry import SpanKind

log = logging.getLogger("tools")


def _ddgs_search(query: str, max_results: int) -> list[SearchHit]:
    from ddgs import DDGS  # imported lazily so mock mode needs no network library

    raw = DDGS().text(query, max_results=max_results) or []
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in raw
    ]


def _mock_search(query: str, max_results: int) -> list[SearchHit]:
    # Deterministic and clearly labelled: these are NOT real web results.
    base = [
        ("[MOCK] AWS fundamentals for AI engineers",
         "Mock snippet: cloud basics (IAM, networking, managed services) help when deploying agents."),
        ("[MOCK] Building production AI agents",
         "Mock snippet: production agents need evaluation, observability, retries and cost control."),
        ("[MOCK] Learn by shipping",
         "Mock snippet: many engineers learn cloud services incrementally while building real projects."),
    ]
    return [
        {"title": t, "url": f"https://example.com/mock/{i}", "snippet": s}
        for i, (t, s) in enumerate(base[:max_results], start=1)
    ]


PROVIDERS = {"ddgs": _ddgs_search, "mock": _mock_search}


def web_search(query: str, provider: str = "ddgs", max_results: int = 5) -> ToolCall:
    """Search the web and return a ToolCall record (inputs, outputs, errors)."""
    record: ToolCall = {
        "tool": "web_search", "provider": provider, "input": query,
        "ok": False, "result_count": 0, "error": None, "results": [],
    }
    fn = PROVIDERS.get(provider)
    # CLIENT span: this is where the request leaves the process (except for the mock).
    with telemetry.span(f"web_search {provider}", SpanKind.INTERNAL if provider == "mock" else SpanKind.CLIENT,
                        {"app.search.provider": provider, "app.search.max_results": max_results,
                         "app.search.query": query if telemetry.capture_content() else None,
                         telemetry.OI_KIND: "RETRIEVER",
                         "input.value": query if telemetry.capture_content() else None}) as s:
        if fn is None:
            record["error"] = f"Unknown search provider '{provider}'. Use one of {list(PROVIDERS)}."
        else:
            try:
                results = fn(query, max_results)
                if results:
                    record.update(ok=True, results=results, result_count=len(results))
                else:
                    record["error"] = "Search returned no results."
            except ImportError:
                record["error"] = "Search library 'ddgs' is not installed (pip install ddgs)."
            except Exception as exc:  # network errors, rate limits, parsing errors...
                record["error"] = f"Search request failed: {type(exc).__name__}: {exc}"
        s.set_attribute("app.search.result_count", record["result_count"])
        if telemetry.capture_content():
            # Phoenix shows RETRIEVER spans as a ranked document list.
            for i, hit in enumerate(record["results"]):
                s.set_attribute(f"retrieval.documents.{i}.document.id", hit["url"])
                s.set_attribute(f"retrieval.documents.{i}.document.content", telemetry.redact(f"{hit['title']}\n{hit['snippet']}"))
        if record["error"]:
            telemetry.fail(s, record["error"], "no_results" if record["error"].startswith("Search returned no")
                           else "search_failed")

    log.info("web_search provider=%s ok=%s results=%d query=%r error=%s",
             provider, record["ok"], record["result_count"], query, record["error"])
    return record


# --------------------------------------------------------------------------- fetch_page
_SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "iframe"}
_BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"}


class _TextExtractor(HTMLParser):
    """Keeps readable text, drops scripts, navigation and other page chrome."""

    def __init__(self):
        super().__init__()  # convert_charrefs=True decodes &amp; etc.
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    lines = (re.sub(r"\s+", " ", line).strip() for line in "".join(parser.parts).splitlines())
    # Very short lines are mostly leftover menu items and buttons.
    return "\n".join(l for l in lines if len(l) > 30)


class UnsafeURLError(ValueError):
    """The URL points somewhere the fetcher must never go (SSRF protection)."""


def check_url_safe(url: str) -> None:
    """Allow only http(s) on standard ports to public internet addresses.

    URLs come from search results, i.e. from the open web. Without this check a
    result (or a redirect) could make the server fetch http://169.254.169.254/
    (cloud credentials), http://localhost:8000/api/memory, or an internal admin page.

    Residual risk: DNS can change between this check and the actual connection
    (DNS rebinding). Production setups add an egress proxy or pin the resolved IP.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise UnsafeURLError(f"scheme '{parts.scheme}' is not allowed")
    if not parts.hostname:
        raise UnsafeURLError("URL has no host")
    if parts.port not in (None, 80, 443):
        raise UnsafeURLError(f"port {parts.port} is not allowed")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"cannot resolve host '{parts.hostname}': {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global or ip.is_multicast:
            raise UnsafeURLError(f"host '{parts.hostname}' resolves to non-public address {ip}")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect target: a public page may redirect to an internal one."""

    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url_safe(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SafeRedirectHandler)


def _http_fetch(url: str, timeout: float) -> str:
    check_url_safe(url)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (research-assistant learning project)",
        "Accept": "text/html,text/plain;q=0.9",
    })
    with _opener.open(req, timeout=timeout) as resp:
        telemetry.event("response", {"http.response.status_code": resp.status, "url.full": resp.geturl()})
        ctype = resp.headers.get_content_type()
        if ctype not in {"text/html", "text/plain", "application/xhtml+xml"}:
            raise ValueError(f"unsupported content type '{ctype}'")
        raw = resp.read(2_000_000)  # cap download size
        charset = resp.headers.get_content_charset() or "utf-8"
    text = raw.decode(charset, errors="replace")
    return text if ctype == "text/plain" else html_to_text(text)


def _mock_fetch(url: str, timeout: float) -> str:
    return (f"[MOCK] Full page text for {url}. Teams that deploy agents typically need IAM, "
            "logging and a managed model endpoint; most learn these while shipping a first project.")


FETCHERS = {"http": _http_fetch, "mock": _mock_fetch}


def fetch_page(url: str, provider: str = "http", max_chars: int = 3000, timeout: float = 10) -> ToolCall:
    """Download a page and return its readable text in a ToolCall record. Never raises."""
    record: ToolCall = {
        "tool": "fetch_page", "provider": provider, "input": url,
        "ok": False, "result_count": 0, "error": None, "results": [], "content": "",
    }
    fn = FETCHERS.get(provider)
    parts = urllib.parse.urlsplit(url)
    # Span name is the method and host only: URL paths are unbounded (bad for grouping) and may be private.
    with telemetry.span(f"GET {parts.hostname or '?'}", SpanKind.INTERNAL if provider == "mock" else SpanKind.CLIENT,
                        {"http.request.method": "GET", "server.address": parts.hostname, "app.fetch.provider": provider,
                         telemetry.OI_KIND: "CHAIN",
                         "url.full": url if telemetry.capture_content() else f"{parts.scheme}://{parts.hostname}/…"}) as s:
        if fn is None:
            record["error"] = f"Unknown fetch provider '{provider}'. Use one of {list(FETCHERS)}."
        elif not url.startswith(("http://", "https://")):
            record["error"] = "Only http(s) URLs can be fetched."
        else:
            try:
                text = fn(url, timeout).strip()
                if text:
                    truncated = len(text) > max_chars
                    record.update(ok=True, result_count=1,
                                  content=text[:max_chars] + (" …[truncated]" if truncated else ""))
                else:
                    record["error"] = "Page had no readable text (it may need JavaScript)."
            except UnsafeURLError as exc:
                record["error"] = f"Blocked unsafe URL: {exc}"
                s.set_attribute("app.fetch.blocked", True)
            except Exception as exc:  # HTTP 403/404, timeouts, TLS errors, PDFs...
                record["error"] = f"Fetch failed: {type(exc).__name__}: {exc}"
                if (code := getattr(exc, "code", None)) is not None:
                    s.set_attribute("http.response.status_code", code)
        s.set_attribute("app.fetch.content_chars", len(record["content"]))
        if record["error"]:
            telemetry.fail(s, record["error"], "blocked_url" if record["error"].startswith("Blocked")
                           else "fetch_failed")

    log.info("fetch_page provider=%s ok=%s chars=%d url=%s error=%s",
             provider, record["ok"], len(record["content"]), url, record["error"])
    return record
