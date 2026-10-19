# Multi-Agent Research Assistant (LangGraph)

A small learning project: a supervisor routes work between a **researcher** (which calls a real web
search tool), a **reviewer** (which approves the research or sends it back once) and a **finalizer**.
Everything is coordinated by a LangGraph `StateGraph`, with long-term memory stored in SQLite.

## Setup

```powershell
cd multi-agent-research
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # optional: add OPENAI_API_KEY to use a real model
```

| Need | Required? | Notes |
|---|---|---|
| `OPENAI_API_KEY` | No | Without it the app uses `MockLLM` (canned output labelled `[MOCK]`). |
| Search API key | No | `ddgs` (DuckDuckGo search) needs no key. `--mock` or `SEARCH_PROVIDER=mock` runs fully offline. |
| Page fetching | No | `FETCH_PAGES` (default 3) and `PAGE_MAX_CHARS` (default 3000) set how much of each top result the researcher reads. `FETCH_PAGES=0` uses snippets only. |

## Run the demo

```powershell
# Real search, real model (if a key is set), otherwise the mock model is used and labelled as such
python main.py "Should a software engineer learn AWS before building production AI agents?"

# Fully offline + deterministic, and forces one review rejection so you can watch the retry path
python main.py --mock --force-reject --show-checkpoints "Should I learn AWS first?"
python main.py --show-prompts "Should I learn AWS first?"   # every prompt sent to the model + its raw output
python main.py --trace "Should I learn AWS first?"          # save model calls to traces/<run id>.jsonl
python main.py --pipeline "Should I learn AWS first?"       # fixed researcher instead of model-driven tool calling

# Long-term memory
python main.py --remember "I prefer concise answers with bullet points"
python main.py --show-memory
python main.py --clear-memory

# Python logging from every node and tool call
python main.py --mock -v "..."
```

The trace prints each node as it runs, why the supervisor routed where it did, every tool call
(provider, query, results or error), the findings, the review decision, whether a retry happened, the
final answer, and optionally the LangGraph checkpoints and saved memory.

## Web inspector

```powershell
python server.py            # then open http://127.0.0.1:8000
```

The page streams each step as it runs:

- a live graph that highlights the active node and the edge just taken
- for every step, four tabs:
  - **State in:** the full state the node received, with the keys it actually reads highlighted
  - **Returned update:** the keys it changed, marked as replaced or appended
  - **LLM:** the exact system and user prompts, the response and the timing
  - **Tools:** the search query, provider and results
- the final answer and outcome
- a long-term memory panel where you can add preferences or clear memory

Tick **Fixed pipeline researcher** to compare the old fixed researcher with model-driven tool calling.
In the agentic mode, the researcher's LLM tab shows one call per turn: the conversation so far, the
model's requested tool calls (`finish_reason: tool_calls`) and finally its text answer.

**HTTP lab** (`/http`, code in `http_lab.py`): a separate learning page that calls the OpenAI API with
plain `httpx` (no SDK) and shows the raw exchange: status line, response headers (request id, rate
limits, processing time) and JSON body, with a guide to every field. Presets cover a normal answer,
`finish_reason: length`, a strict schema, a tool call, a tool-result round-trip, streaming (raw SSE
chunks) and three kinds of error. Output is capped at 500 tokens, and the key is never sent to the page.

`server.py` uses only the Python standard library, binds to localhost only, and captures the trace from
LangGraph's `stream_mode="debug"` plus a `RecordingLLM` wrapper in `llm.py`.

## Run the tests

```powershell
python -m pytest -v
```

The 119 tests run offline. They cover routing rules, shared-state updates, the reviewer reading the
researcher's findings from state, the retry path, the retry limit, search and model failures (no
fabricated success), page fetching and fallback, and memory persistence across reconnections.

## Architecture

```
            START
              |
         load_memory          (SQLite -> state.memory_context)
              |
   +----> supervisor ---------(conditional edge on state.next_step)----+
   |       |      |       |                                            |
   |  researcher reviewer finalizer                              save_memory
   |       |      |       |                                            |
   +-------+------+-------+                                           END
```

**Why hub-and-spoke?** Every worker returns to the supervisor, so all routing goes through one
pure function, `decide_next_step()` in `agents.py`. No worker can skip review or exceed the revision
limit, and the routing rules can be unit-tested without running any model. Loading and saving
memory are plain pipeline steps, not routing choices, so they use fixed edges.

The supervisor is **rule-based**, not LLM-based. With only four states, explicit rules are
cheaper, deterministic and testable. An LLM supervisor only pays off when the routing choice
genuinely needs judgment (for example, choosing among many specialist agents).

### Execution flow

1. `load_memory` copies relevant SQLite memories into `state.memory_context`.
2. `supervisor` sees `review_status="not_started"` and routes to `researcher`.
3. `researcher` calls `web_search()`, then `fetch_page()` on the top 3 results in parallel, and
   writes `tool_calls`, `sources` (snippets plus full page text), `research_findings` and
   `review_status="pending"`. A failed fetch isn't fatal: that source keeps only its snippet.
4. `supervisor` sees `pending` and routes to `reviewer`.
5. `reviewer` reads `research_findings` **from state** and writes `review_status` = approved or rejected.
6. If the work was rejected and `revision_count < MAX_REVISIONS`, the supervisor increments
   `revision_count` and routes back to `researcher`, which uses the feedback to refine its search.
   Otherwise it routes to `finalizer`.
7. `finalizer` writes `final_answer` and `outcome` (approved, unapproved or failed).
8. `save_memory` persists a summary **only if** `outcome == "approved"`. Then the graph ends.

If any node sets `error`, the supervisor sends the run straight to the finalizer, which reports the
failure honestly without calling the model.

## Components

| Component | Responsibility | Receives | Returns / changes | Invoked by | If removed |
|---|---|---|---|---|---|
| `state.py` `ResearchState` | Schema of the shared working memory | — | Defines keys and reducers (`history` and `tool_calls` are append-only) | LangGraph at every step | Nodes have no shared contract; updates overwrite logs |
| `graph.py` | Wires nodes and edges, compiles with a checkpointer | `ResearchAgents` | Compiled graph | `main.py`, tests | Nothing runs; the nodes are just unconnected functions |
| `supervisor` | Chooses the next node, owns `revision_count` | Whole state | `next_step`, `revision_count`, history | Fixed edges from every worker | No routing, so no retry loop and no retry limit |
| `researcher` | Searches, then synthesises findings with citations | `question`, `memory_context`, `review_feedback` | `tool_calls`, `sources`, `research_findings`, `review_status=pending` (or `error`) | Supervisor's conditional edge | No evidence; the reviewer has nothing to check |
| `reviewer` | Quality gate: approve or reject | `question`, `research_findings`, `sources` | `review_status`, `review_feedback` (or `error`) | Supervisor's conditional edge | Unchecked research goes straight to the user |
| `finalizer` | Writes the user-facing answer, or an honest failure message | Findings, sources, memory, `error` | `final_answer`, `outcome` | Supervisor's conditional edge | The user gets raw findings, or nothing |
| `load_memory` / `save_memory` | Bridge between SQLite and graph state | `question` / `outcome`, `final_answer` | `memory_context` / a new SQLite row | Fixed edges | No personalisation or learning across runs |
| `tools.web_search` | Calls the search provider, always returns a `ToolCall` record and never raises | query, provider | `{ok, results, error, ...}` | Researcher | No external information |
| `tools.fetch_page` | Downloads a page and extracts its readable text (stdlib `urllib` + `html.parser`); never raises | URL | `{ok, content, error, ...}` | Researcher, for the top `FETCH_PAGES` results, in parallel | The model sees only 1–2 sentence snippets, so answers are thinly grounded |
| `llm.py` | Vendor-neutral model interface (`OpenAILLM`, `MockLLM`) | system and user prompt | text, or raises `LLMError` | Researcher, reviewer, finalizer | Agents get tied to one SDK and can't be tested offline |
| `memory.MemoryStore` | SQLite persistence and keyword retrieval | facts and preferences | rows | Memory nodes, CLI | Nothing survives a restart |

## Key concepts

**Agent vs node vs edge vs tool**

- **Node:** a function `state -> partial update` registered in the graph. It is the unit LangGraph runs.
- **Agent:** a node with a role that makes a decision, usually by calling a model (researcher,
  reviewer, finalizer) or by applying rules (supervisor). Every agent here is a node, but not every
  node is an agent: `load_memory` and `save_memory` are plumbing.
- **Edge:** a transition between nodes. A fixed edge always fires (`researcher -> supervisor`). A
  conditional edge calls a function on the state to pick the target (`supervisor -> ?`).
- **Tool:** a plain function that touches the outside world (`web_search`). It has no reasoning and
  doesn't route. The agent decides when to call it and what to do with the result.

**The five kinds of "memory". These are not interchangeable.**

| | What | Lifetime | Here |
|---|---|---|---|
| 1. Agent context | The exact text sent in one model call | One API call | The `system`/`user` strings each node builds. The model sees nothing else. |
| 2. Shared graph state | Typed dict passed between nodes | One graph run | `ResearchState`: the researcher writes `research_findings`, the reviewer reads it |
| 3. Conversation history | The sequence of chat turns with a user | One chat session | Not implemented. Each run is a single question; you could add a `messages` key |
| 4. Persistent long-term memory | Application data saved deliberately | Across restarts | SQLite `facts` table (`memory.py`) |
| 5. LangGraph checkpointing | Automatic snapshot of graph state after each step, per `thread_id` | As long as the checkpointer keeps it (in-memory here) | `InMemorySaver`, shown with `--show-checkpoints`. Used for resume, replay and human-in-the-loop, **not** for remembering facts across questions |

**Shared state vs model-call context.** The graph state can hold anything (tool logs, counters,
errors), but a model only ever sees what a node explicitly copies into its prompt. For example, the
reviewer's prompt includes the findings and sources but not `tool_calls` or `history`. Deciding
what goes into each prompt is your job as the engineer.

## Evaluation

Tests prove the code behaves as designed. Evals measure **how good the answers are**, on a fixed question
set, so you can tell whether a change (a prompt, a model, agentic vs pipeline) actually helped.

```powershell
python -m evals.run_evals                                   # agentic vs pipeline, deterministic checks only
python -m evals.run_evals --judge --trials 2                # + LLM judge, 2 trials per case
python -m evals.run_evals --judge --judge-model gpt-4.1     # a stronger, different judge model
python -m evals.run_evals --cases lambda-timeout,rag-basics # a subset
python -m evals.run_evals --judge --save-baseline           # record today's numbers
python -m evals.run_evals --judge --fail-on-regression      # exit 1 if worse than the baseline (CI gate)
python -m evals.run_evals --mock                            # offline: tests the harness, not quality
```

**Dataset** (`evals/dataset.json`): 12 cases across factual, concept, comparison, trade-off and security
questions. Each has key points written as groups of acceptable phrasings, and there's one **unanswerable**
case (a future event) where the right behaviour is to decline.

**What each run is scored on:**

| Check | Kind | Pass condition |
|---|---|---|
| Outcome | code | `approved` by the reviewer |
| Citations | code | the answer cites sources, and every `[n]` exists |
| Key-point coverage | code | at least the case's `min_coverage` of key-point groups mentioned |
| Grounding | code (heuristic) | share of cited sentences whose words appear in the cited source (reported, not gating) |
| Uncited claims | code (heuristic) | long sentences with no citation (reported) |
| Abstention | code | unanswerable cases must decline or fail honestly |
| Faithfulness, completeness, citation quality | LLM judge, strict schema, rubric 1–5 | faithfulness ≥ 4 |

**How the harness keeps results honest:**
- Every run uses a **fresh, empty memory database**, so saved facts can't leak into answers. Evals never touch `memory.db`.
- **Trials:** models are nondeterministic. Cases that pass in some trials and fail in others are reported as **flaky**.
- A **cost ceiling** (`--max-cost`, default $1) stops the eval before it can run away.
- **Baseline and regressions:** `evals/baseline.json` stores a summary. Later runs are compared with tolerances
  (pass rate −10 points, faithfulness −0.5, coverage −10 points, mean cost +30%).
- Reports: `evals/results/<timestamp>.md` (readable) and `.json` (every run, every score, every answer).

Limits to keep in mind: keyword coverage misses paraphrases; a judge using the same model as the system tends
to favour it (use `--judge-model`); live web search changes over time, so compare runs made close together.

## Model-driven tool calling (agentic researcher)

`RESEARCHER_MODE=agentic` (the default) turns the researcher into a tool-calling loop. The model
decides what to search, which results to read and when it has enough evidence:

```
messages = [system, user(question + memory + reviewer feedback)]
loop (at most MAX_TOOL_TURNS turns that may use tools):
    reply = llm.chat(messages, tools=[web_search, fetch_page])
    if reply has no tool calls -> reply.text is the findings; stop
    append the assistant's tool_calls; run each tool; append one role:"tool" result per call id
final turn: tool_choice="none" -> the model must answer in text
```

The model chooses **what** to call; code still controls **how much** and **what is allowed**:

| Guardrail | Why |
|---|---|
| `MAX_TOOL_TURNS`, `MAX_SEARCHES`, `MAX_FETCHES`, plus a forced final text turn | No runaway loops or cost |
| `fetch_page(source_id)` takes an **id from search results, not a URL** | Injected text can't send the agent to arbitrary or internal URLs (SSRF checks still apply underneath) |
| Strict JSON schemas on tool arguments; bad JSON, unknown tools and invalid ids become **error results the model can react to** | One bad call doesn't crash the run |
| Every tool result is fenced in `<source>` tags and escaped | Same prompt-injection defence as before |
| Parallel tool calls in one turn run concurrently | Faster page reads |
| No sources gathered means an **honest failure**, whatever the model wrote | Never fabricated research |

The reviewer, finalizer, supervisor and graph are unchanged. The researcher still writes
`research_findings`, `sources` and `tool_calls` into state, so the rest of the system doesn't know how
the evidence was gathered. `RESEARCHER_MODE=pipeline` (or `--pipeline`) keeps the fixed workflow for comparison.

## Tools over MCP (optional)

`TOOL_TRANSPORT=mcp` (or the **Tools via MCP** checkbox in the inspector) runs the same `web_search`
and `fetch_page` code in a separate process, `mcp_server.py`, and calls it over the
[Model Context Protocol](https://modelcontextprotocol.io). Nothing else changes:

```
OpenAI model --tool_calls--> researcher loop (guardrails, source ids) --search(q)/fetch(url)--> ?
   local: tools.web_search(...)                       in-process function call
   mcp:   MCPToolbox.call("web_search", {...})  ==JSON-RPC over stdio==>  mcp_server.py -> tools.web_search(...)
```

**Tool calling and MCP sit at different layers.** Tool calling is the model-to-app protocol: the model
asks for a call and the app runs it. MCP is the app-to-tool-server protocol: it decides where the tool code
runs and how the app discovers it (`initialize`, `tools/list`) and invokes it (`tools/call`). The model never
speaks MCP. It still sees our `web_search(query)` / `fetch_page(source_id)` schemas, so the source-id guardrail,
budgets and stuck detection all keep working.

| File | Role |
|---|---|
| `mcp_server.py` | FastMCP server. Function signatures become `inputSchema`s; logs go to stderr because stdout is the protocol channel |
| `mcp_client.py` | `MCPToolbox`: a blocking wrapper (background event loop, one long-lived session); `mcp_tool_functions` keeps the never-raise `ToolCall` contract; `openai_tools()` shows how a generic host turns discovered tools into OpenAI `tools` |

```powershell
python mcp_client.py --mock     # watch initialize -> tools/list -> tools/call -> close
python mcp_client.py            # same, with live search, plus an SSRF probe that the server blocks
$env:TOOL_TRANSPORT="mcp"; python main.py "What problem does MCP solve?"   # trace lines say "via MCP"
```

Design notes:
- **Secrets:** the server starts with the SDK's minimal environment (`PATH` and similar) plus
  `SEARCH_PROVIDER`. It never sees `OPENAI_API_KEY`.
- **Security travels with the tool:** SSRF checks run inside the server, because it's the process
  that makes the request.
- **Process cost:** starting a server takes about 1s, so a single shared session per search provider
  is reused across runs. Parallel `tools/call` requests overlap (async tools plus worker threads).
- **Failures:** a crashed server, a timeout or a protocol error becomes an `ok=False` record, just like
  a failed search. Nothing is fabricated.
- **Two transports:** by default the client starts the server as a subprocess (stdio). You can also
  run it as a long-lived HTTP service and connect by URL, using the same tools and JSON-RPC messages:

  ```powershell
  $env:MCP_TOKEN="pick-a-random-token"; python mcp_server.py --http          # http://127.0.0.1:8765/mcp
  # second terminal:
  $env:MCP_TOKEN="pick-a-random-token"; python mcp_client.py --url=http://127.0.0.1:8765/mcp
  $env:TOOL_TRANSPORT="mcp"; $env:MCP_URL="http://127.0.0.1:8765/mcp"; python main.py "..."
  ```

  | | stdio | Streamable HTTP |
  |---|---|---|
  | Who starts the server | the client, as a subprocess | you, as a service, independently |
  | Clients | exactly one | many, local or remote |
  | Wire | JSON lines on stdin/stdout | `POST /mcp` per message; replies as JSON or SSE; `Mcp-Session-Id` header |
  | Auth | the process boundary | HTTP: `Authorization: Bearer` here, OAuth 2.1 in the spec |
  | Use for | local tools for one app or IDE | shared or remote tools, containers, other machines |

  HTTP mode binds to `127.0.0.1` on port 8765 (8000 is `server.py`). The SDK's DNS-rebinding protection
  rejects requests whose `Host` or `Origin` is not localhost (HTTP 403). Set `MCP_TOKEN` to require a bearer
  token (otherwise HTTP 401). For HTTPS, pass `--cert`/`--key`; locally, `mkcert localhost 127.0.0.1` creates a
  certificate your machine trusts, or point clients at the cert with `MCP_CA_FILE`.
- **See every message:** the `/mcp` page of `server.py` (or `python mcp_lab.py --mock [--http] "question"`) runs one
  question through a small agent loop and records every hop raw. That covers the JSON-RPC lines on the server's
  stdin/stdout (from a hand-written client, no SDK) and the HTTPS requests and responses to the model, drawn
  as a clickable sequence diagram. Switch the transport to **Streamable HTTP** to see the HTTP framing (`POST /mcp`,
  `Mcp-Session-Id`, SSE replies, `DELETE`). Use your own running server via its URL and token, or let the lab start a
  temporary one. **Call a tool directly** sends your own `tools/call` arguments, with no model involved.
- **Do it yourself:** `/mcp-console` hands you an open MCP connection (stdio or HTTP) and a raw request editor for both the
  server and the model. Guided steps fill in each next message from the previous replies (`initialize`, `notifications/initialized`,
  `tools/list`, convert the tools, ask the model, turn its `tool_call` into `tools/call`, add the result as a `role:"tool"` message).
  A state panel shows what the app holds between calls. Experiments break things on purpose: calling before
  `initialize`, unknown methods, missing arguments, invalid JSON, SSRF, and omitting tool results.
- **Same server, other hosts:** any MCP host can use `mcp_server.py`, for example
  `claude mcp add research-tools -- <venv python> mcp_server.py` in Claude Code. That reuse is the
  point of MCP.

## Production hardening (stage 1)

| Concern | What the code does | Where |
|---|---|---|
| **SSRF** | `fetch_page` only fetches http(s) on ports 80/443 to **public** IPs. It resolves the hostname and blocks private, loopback, link-local and metadata addresses (e.g. `169.254.169.254`), and re-checks every redirect. | `tools.check_url_safe`, `_SafeRedirectHandler` |
| **Prompt injection** | Web content is fenced in `<source>` tags, which pages can't close early, and every prompt containing it carries `UNTRUSTED_NOTICE`: treat it as evidence, never as instructions. | `agents._format_sources`, `UNTRUSTED_NOTICE` |
| **Model reliability** | Per-attempt timeout; retries with exponential backoff and jitter **only** for transient errors (429, 5xx, timeouts, dropped connections). 4xx errors fail fast. Retries are bounded. | `llm.OpenAILLM`, `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES` |
| **Structured output** | The reviewer's verdict is enforced by a strict JSON Schema (`approved`, `feedback`, `unsupported_claims`), so there's no fragile free-form parsing. | `agents.REVIEW_SCHEMA` |
| **Citation verification** | Before any model review, code checks that findings cite sources and that every `[n]` exists. Failures are rejected for free, with no tokens spent. | `agents.check_citations` → `state.citation_check` |
| **Cost and usage** | Every model call records tokens, attempts, latency and estimated cost in `state.llm_usage` (append reducer). Shown in the CLI summary and the inspector. | `agents._usage`, `PRICE_*_PER_M` |
| **Graph step limit** | `recursion_limit` is set explicitly to the longest legitimate path plus headroom (13 + 4 steps with one revision). A routing bug that creates a cycle fails fast with `GraphRecursionError`. LangGraph's own default is about 10,000. | `graph.recursion_limit`, `run_config` |
| **Stuck detection (tool loop)** | A repeated search (same normalised query) isn't run again; the model is told to try a different query. After `MAX_STALE_TURNS` (2) turns in a row with no new sources or page text, the loop is forced to answer. | `agents._research_agentic`, `_run_tool_calls` |
| **Stuck detection (revisions)** | The reviewer returns a 0–10 `score`. The supervisor sends work back again only if the last revision **improved** the score; otherwise it stops early instead of spending the remaining revisions. | `REVIEW_SCHEMA`, `decide_next_step`, `state.review_score` |
| **Memory hygiene** | One fact per question (upsert on a normalised topic); old duplicates are collapsed automatically when the database opens. | `memory.upsert_fact`, `_migrate` |

Known limits that remain: DNS rebinding between the check and the connection (production adds an egress
proxy); the prompt-injection defence lowers the risk but can't remove it; the model reviewer still grades
another model's output.

**Next stages:** a real search API (Tavily or Brave), per-user memory namespaces, a Postgres checkpointer,
and async execution with a job queue. (Evaluation and tracing are done: see [Evaluation](#evaluation) and
[Observability](#observability-opentelemetry).)

## Observability (OpenTelemetry)

Every question becomes one **trace**: a tree of timed **spans** recorded with the OpenTelemetry SDK and written
locally as JSONL. View them at **http://127.0.0.1:8000/traces**. `main.py` also prints each run's trace id.

```
invoke_agent research-assistant                  root span: outcome, tokens, cost, tool failures (= the run's metrics)
  node load_memory / node supervisor             one span per graph step; the supervisor's span has app.route.next/.reason
  invoke_agent researcher
    chat gpt-4.1-mini            CLIENT          gen_ai.usage.*, finish_reasons, cost, attempts (+ a "retry" event per retry)
    execute_tool web_search                      gen_ai.tool.call.id = the model's tool_call id
      tools/call web_search      CLIENT          only with TOOL_TRANSPORT=mcp
        tools/call web_search    SERVER          ← in the mcp_server.py process, same trace
          web_search ddgs        CLIENT          the provider call in tools.py
    execute_tool fetch_page
      GET en.wikipedia.org       CLIENT          outbound fetch (status code, blocked by SSRF check...)
```

| Piece | What it does | Where |
|---|---|---|
| Spans | The node wrapper, model wrapper (`TracedLLM`) and tool wrapper (`traced_tool`) add spans without changing the agents' code. Names follow the OTel **GenAI** and **MCP** semantic conventions (`gen_ai.*`, `mcp.*`); our own fields use `app.*`. | `telemetry.py`, `graph.build_graph`, `agents.ResearchAgents.__init__` |
| Across processes | The MCP client puts W3C `traceparent` into `params._meta` of every `tools/call`; the server continues the trace from it. It's inside the JSON-RPC message, so it works over stdio (no headers there) and HTTP. | `mcp_client.MCPToolbox.call`, `mcp_server._server_span` |
| Across threads | Parallel page fetches run in a thread pool, which starts with an empty context. `_parallel_map` copies the context in, so those spans keep their parent. | `agents._parallel_map` |
| Structured logs | Every log line is also written as JSON with the active `trace_id`/`span_id`, so a span shows its log lines and a log line links to its trace. The console stays human-readable. | `telemetry.JsonLogHandler` |
| Metrics | Computed from spans: per run (root span attributes), across runs (p50/p95 duration, cost, tool failure rate, retries), per operation, and failure categories (each failure counted once, where it started). | `trace_store.py`, `/api/traces` |
| Viewer | Run list, waterfall per trace (process boundaries marked), span attributes, events and logs, operations and failures tables. | `static/traces.html` |
| Evals | Every eval run is a trace too, and each result row stores its `trace_id`, so a failing case links to what happened. | `evals/run_evals.py` |

Privacy and safety defaults:
- Prompts, answers, questions and tool inputs are **not** recorded in spans unless you set
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` (the GenAI convention's switch).
- API keys and bearer tokens are redacted from every attribute and log line.
- The JSON logs follow the app's existing log statements, which include search queries and URLs. Treat
  `traces/` as containing user data. It is gitignored.

Turn it off with `TELEMETRY=off`. To send the same spans to another backend (Phoenix below, Jaeger, Grafana
Tempo, Langfuse, or AWS X-Ray / CloudWatch through the ADOT collector), set `OTEL_EXPORTER_OTLP_ENDPOINT`. If the
endpoint isn't reachable at startup, the app logs one warning and keeps writing local files. It never waits on a
missing backend.

### Phoenix (Arize): an LLM-aware trace UI

[Phoenix](https://github.com/Arize-ai/phoenix) is Arize's open-source, local trace viewer for LLM apps. It builds its
LLM views from **OpenInference** attributes, not the OTel `gen_ai.*` names, so every span carries both:

| Concept | OTel GenAI | OpenInference (Phoenix) |
|---|---|---|
| Kind of step | `gen_ai.operation.name` = chat / invoke_agent / execute_tool | `openinference.span.kind` = LLM / AGENT / TOOL / CHAIN / RETRIEVER |
| Model, tokens, cost | `gen_ai.request.model`, `gen_ai.usage.*` | `llm.model_name`, `llm.token_count.*`, `llm.cost.total` |
| Messages (opt-in) | `gen_ai.input.messages` | `llm.input_messages.N.message.role/content/tool_calls...`, `input.value` / `output.value` |
| Search results (opt-in) | — | `retrieval.documents.N.document.id/content` |

Phoenix runs in its **own** venv: it's a separate service with many dependencies, and they shouldn't change the app's.

```powershell
# once
.venv\Scripts\python -m venv .venv-phoenix
.venv-phoenix\Scripts\python -m pip install arize-phoenix

# start it (a second terminal). PHOENIX_HOST keeps it on this machine: the default is 0.0.0.0 (your whole network).
$env:PHOENIX_HOST="127.0.0.1"; .venv-phoenix\Scripts\phoenix serve       # UI at http://127.0.0.1:6006

# point the app at it (or put these two lines in .env), then run anything
$env:OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:6006"; $env:PHOENIX_PROJECT_NAME="research-assistant"
.venv\Scripts\python main.py --mock "What is MCP?"
```

Open the `research-assistant` project. The first start takes a minute while Phoenix creates its database (in
`~\.phoenix`). Prompts and answers show up only with `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`.
The MCP server receives the endpoint through `telemetry.child_env()`, so its spans reach Phoenix in the same trace.
`OTEL_EXPORTER_OTLP_HEADERS` is deliberately *not* passed on: it can hold a backend API key.

**Not covered yet:** OTel *metrics* instruments (counters and histograms pushed to a backend). The numbers here are
derived from spans, which is enough locally. There is also no retention: delete old files in `traces/otel/` yourself.

## AWS mapping (not deployed)

How this local MVP could evolve into a cloud service:

| Local piece | AWS option | Notes |
|---|---|---|
| `OpenAILLM` | **Amazon Bedrock** | Add a `BedrockLLM` class that implements the same `complete()` interface using `boto3` `bedrock-runtime` `converse`. Bedrock provides model access only; it does not store your agent's memory or state. |
| `main.py` request handling | **AWS Lambda** (behind API Gateway) | Fine for short runs. A run makes several model and search calls, so watch the 15-minute limit and latency. For longer or streaming work, use ECS/Fargate. Lambda is stateless: memory must live elsewhere. |
| `MemoryStore` (SQLite) | **Amazon DynamoDB** | Partition key `user_id`, sort key `created_at`. SQLite on Lambda would be lost between invocations. Aurora Serverless or RDS also work if you want SQL. |
| LangGraph checkpointer | DynamoDB, Postgres (`langgraph-checkpoint-postgres`) or S3 | Only needed if you want to resume or inspect runs after a failure. |
| Documents / reports | **Amazon S3** | Store final reports and source documents. Optional for this MVP, which has no documents. |
| `.env` | **AWS Secrets Manager** (or SSM Parameter Store) | Fetch the API keys at cold start. Never put them in Lambda env vars in plain text or in code. |
| `logging`, `telemetry.py` | **Amazon CloudWatch** Logs, **AWS X-Ray** / CloudWatch Application Signals | Lambda logs go there automatically. The OTel spans export unchanged through the AWS Distro for OpenTelemetry (ADOT) collector, so traces, including the MCP server hop, show up in X-Ray. |
| LangGraph loop | **AWS Step Functions** (optional) | Use it when you need durable, auditable orchestration with built-in retries, timeouts and recovery across long or human-approval steps. For a single fast request, LangGraph inside one Lambda is simpler, so Step Functions is optional. Step Functions keeps execution state, not agent knowledge. |

**Optional vs core:** to host this app you need compute (Lambda or ECS), a durable store for memory
(DynamoDB) and secrets (Secrets Manager). Bedrock is optional if you keep OpenAI. S3 and Step
Functions are optional until you have documents or long, multi-step durable workflows. CloudWatch
comes almost for free with Lambda.

## Exercises

1. **LLM supervisor:** replace `decide_next_step` with a model call that returns JSON
   `{"next": ..., "reason": ...}`, but keep the revision limit enforced in code. Why must the
   limit stay in code?
2. **Conversation history:** add a `messages: Annotated[list, add_messages]` key and an interactive
   loop that reuses one `thread_id`, so follow-up questions see earlier turns. Compare this with
   long-term memory.
3. **Persistent checkpoints:** swap `InMemorySaver` for `SqliteSaver` (`langgraph-checkpoint-sqlite`),
   kill the process mid-run, and resume from the last checkpoint.
4. **Smarter reading:** `fetch_page` keeps the first `PAGE_MAX_CHARS` of each page. Instead, split
   the page into paragraphs and keep only those that share the most keywords with the question.
   Compare answer quality and prompt size in the web inspector's LLM tab.
5. **Bedrock:** implement `BedrockLLM.complete()` with `boto3` `converse`, select it with
   `LLM_PROVIDER=bedrock`, and confirm that the tests still pass unchanged.
