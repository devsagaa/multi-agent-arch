"""Runtime configuration, read from environment variables (and a local .env file).

Credentials only ever come from the environment -- never from source code.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str            # "openai" or "mock"
    openai_model: str
    llm_timeout_s: float         # per-attempt timeout for a model call
    llm_max_retries: int         # retries for transient errors (429, 5xx, timeouts)
    price_input_per_m: float     # USD per 1M input tokens, for cost tracking
    price_output_per_m: float    # USD per 1M output tokens
    search_provider: str         # "ddgs" or "mock"
    search_max_results: int
    fetch_pages: int             # how many top results to read in full (0 = snippets only)
    page_max_chars: int          # page text kept per fetched page
    max_revisions: int           # how many times the reviewer may send work back
    force_reject_first_review: bool  # deterministic way to exercise the retry path
    memory_db_path: str
    # Answers built from canned model output or fake search results must never become
    # long-term "facts". Tests switch this on to exercise saving against a temp database.
    save_mock_answers: bool = False
    # Researcher style. "agentic": the model chooses web_search/fetch_page calls in a loop.
    # "pipeline": fixed code path (search once, read the top pages, one model call).
    researcher_mode: str = "agentic"
    # Hard budgets for the agentic loop -- the model decides WHAT to call, code decides HOW MUCH.
    max_tool_turns: int = 4      # model turns that may request tools; one more turn must answer
    max_searches: int = 3
    max_fetches: int = 4
    # Stuck detection: turns in a row that add no new sources or page text before the
    # loop is told to stop searching and answer (tool_choice="none").
    max_stale_turns: int = 2
    # Where the tool code runs. "local": plain function calls in this process.
    # "mcp": the same tools served by mcp_server.py, called over the Model Context Protocol.
    tool_transport: str = "local"
    # Optional: connect to an already-running MCP server over Streamable HTTP instead of
    # spawning one (e.g. http://127.0.0.1:8765/mcp). MCP_TOKEN is sent as a bearer token.
    mcp_url: str = ""
    mcp_token: str = ""


def load_settings(force_mock: bool = False, force_reject_first: bool | None = None) -> Settings:
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    llm = os.getenv("LLM_PROVIDER", "auto").lower()
    if llm == "auto":
        llm = "openai" if has_key else "mock"
    search = os.getenv("SEARCH_PROVIDER", "ddgs").lower()
    if force_mock:
        llm, search = "mock", "mock"

    return Settings(
        llm_provider=llm,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        llm_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "30")),
        llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        # Defaults are gpt-4.1-mini list prices at the time of writing -- check your provider's pricing page.
        price_input_per_m=float(os.getenv("PRICE_INPUT_PER_M", "0.40")),
        price_output_per_m=float(os.getenv("PRICE_OUTPUT_PER_M", "1.60")),
        search_provider=search,
        search_max_results=int(os.getenv("SEARCH_MAX_RESULTS", "5")),
        fetch_pages=int(os.getenv("FETCH_PAGES", "3")),
        page_max_chars=int(os.getenv("PAGE_MAX_CHARS", "3000")),
        max_revisions=int(os.getenv("MAX_REVISIONS", "1")),
        force_reject_first_review=(
            force_reject_first
            if force_reject_first is not None
            else _flag("FORCE_REJECT_FIRST_REVIEW")
        ),
        memory_db_path=os.getenv("MEMORY_DB_PATH", "memory.db"),
        researcher_mode=os.getenv("RESEARCHER_MODE", "agentic").lower(),
        max_tool_turns=int(os.getenv("MAX_TOOL_TURNS", "4")),
        max_searches=int(os.getenv("MAX_SEARCHES", "3")),
        max_fetches=int(os.getenv("MAX_FETCHES", "4")),
        max_stale_turns=int(os.getenv("MAX_STALE_TURNS", "2")),
        tool_transport=os.getenv("TOOL_TRANSPORT", "local").lower(),
        mcp_url=os.getenv("MCP_URL", ""),
        mcp_token=os.getenv("MCP_TOKEN", ""),
    )
