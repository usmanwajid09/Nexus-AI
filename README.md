# Nexus AI

> A modular AI operating system for knowledge workers — it remembers, retrieves, reasons, routes to specialists, understands code and images, and researches the live web.

Phases 1–5 of the original plan are implemented as one coherent codebase: a routed LangGraph pipeline over a single Postgres+pgvector store, with every capability behind a small interface and every pure function unit-tested (48 tests).

## What shipped vs. the original 10-module spec

| Original plan | What shipped | Why this shape |
|---|---|---|
| Postgres + Qdrant + Neo4j + Redis + Kafka | **One Postgres with pgvector** | Vector, keyword, and relational data in one database. Interfaces make Qdrant/Neo4j a swap-in when a measured limit demands it. |
| Basic RAG → "self-improving RAG" | **Query rewriting → multi-query hybrid retrieval (RRF) → LLM reranking → groundedness grading** + golden-set eval harness | Every stage of the plan's pipeline, without torch or a search cluster. The eval harness (`scripts/run_evals.py`) is what makes "self-improving" measurable. |
| 4 separate memory systems | **One `memories` table** (episodic/semantic/procedural types) written by a **background memory writer** | Extraction runs after the response is sent — zero added chat latency. Access stats are stored for future decay/reinforcement. |
| CEO→Manager→…→Deployment agent hierarchy | **Router + specialists**: `general`, `code`, and `research` routes as conditional graph edges | Specialists are graph branches, not microservices. Adding one = one node + one edge. |
| Browser agent (Playwright) | **Research agent on Claude's server-side `web_search` + `web_fetch` tools** (handles `pause_turn` continuation) | Web-grounded answers with source URLs, no browser infrastructure to babysit. Full click-and-type automation is the one deliberately deferred piece (see Deferred). |
| Autonomous engineer (tree-sitter, PR bot) | **Repo ingestion with definition-boundary code chunking** (`kind="code"`), code route proposes **unified diffs** grounded in retrieved source | tree-sitter is a native extension — blocked on locked-down machines (this one blocks unsigned DLLs). The heuristic chunker sits behind the same signature so tree-sitter can slot in. |
| Vision module (YOLO, SAM, Florence-2) | **`/vision/analyze` on Claude's native multimodality**, optional ingestion of the analysis into the KB | Claude reads screenshots, diagrams, charts, and documents directly. A CV-model zoo would be five dependencies solving a solved problem. |
| Enterprise backend (JWT, RBAC, orgs, Kafka) | **JWT auth (HS256)** with a dev-mode off switch; single-tenant | Auth is on every data endpoint when `AUTH_SECRET` is set. RBAC/orgs deferred until there are two users. |
| OpenTelemetry + Grafana + Prometheus | **Prometheus metrics** (latency, LLM calls/tokens, retrieval sizes, answer confidence) + compose profile for **Prometheus + Grafana** | `/metrics` is real and scraped; dashboards run with `--profile observability`. |
| Kubernetes + NGINX | **docker-compose** | Compose runs the whole stack on a laptop. Kubernetes before users is résumé-driven ops. |

## Architecture

```
  POST /chat   POST /documents   POST /repos   POST /vision/analyze   GET /memories/search
      |              |               |                |                      |
      v              v               v                v                      v
  FastAPI gateway ── JWT auth (optional) ── request IDs ── Prometheus /metrics
      |
      v
  LangGraph orchestrator
      START ─> route ─┬─> research ──────────────────────────────> END
                      |   (server-side web_search + web_fetch)
                      └─> recall ─> rewrite ─> retrieve ─> generate ─> grade ─> END
                          memory    1-3       hybrid+RRF   Claude     LLM judge:
                          recall    queries   + LLM rerank            confidence +
                                              (code route            unsupported
                                               filters kind)          claims
      (memory extraction runs as a background task after the response)
                      |
                      v
  +──────────────────────────────────────────────+
  |            PostgreSQL + pgvector             |
  |  memories | documents/chunks (text & code)   |
  |  messages | vector + full-text search        |
  +──────────────────────────────────────────────+
```

LLM: `claude-opus-4-8` with adaptive thinking; structured outputs (strict JSON schemas) for every machine-read LLM call (routing, rewriting, reranking, grading, memory extraction) — each with a pure, unit-tested parser that degrades gracefully on garbage.

## Quickstart

```powershell
docker compose up -d                                  # Postgres + pgvector
copy .env.example .env                                # set ANTHROPIC_API_KEY
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn nexus.api.main:app --reload
```

Optional observability stack: `docker compose --profile observability up -d` → Grafana at `:3000`, Prometheus at `:9090` scraping `:8000/metrics`.

### Try every capability

```powershell
# Memory: teach it something (extracted in the background)
curl -X POST localhost:8000/chat -H "Content-Type: application/json" `
  -d '{"message": "Remember that our backend uses FastAPI and we deploy with make deploy."}'

# RAG: feed the knowledge base, then ask (rewrite -> hybrid retrieve -> rerank -> grade)
curl -X POST localhost:8000/documents -H "Content-Type: application/json" `
  -d '{"title": "Architecture notes", "text": "Authentication is handled by the gateway using JWT..."}'
curl -X POST localhost:8000/chat -H "Content-Type: application/json" `
  -d '{"message": "How does auth work?"}'          # response includes confidence + sources

# Code route: ingest a repo, then ask about it
curl -X POST localhost:8000/repos -H "Content-Type: application/json" `
  -d '{"path": "C:/Data/Treasure"}'
curl -X POST localhost:8000/chat -H "Content-Type: application/json" `
  -d '{"message": "Where is hybrid retrieval implemented and how does the fusion work?"}'

# Research route: needs live web (server-side web search)
curl -X POST localhost:8000/chat -H "Content-Type: application/json" `
  -d '{"message": "Search the web: what is the latest stable PostgreSQL release?"}'

# Vision: analyze an image, optionally ingest the analysis
curl -X POST localhost:8000/vision/analyze -F "file=@diagram.png" `
  -F "question=Explain this architecture diagram" -F "ingest_result=true"

# Inspect memories
curl "localhost:8000/memories/search?q=backend"
```

### Auth (Phase 2)

```powershell
# .env: AUTH_SECRET=<long random string>   -> all data endpoints require a bearer token
python scripts/make_token.py alice
curl -H "Authorization: Bearer <token>" ...
```

### Evals (Phase 2)

```powershell
python scripts/seed_demo.py     # seed demo documents + memories (no API keys needed)
python scripts/run_evals.py     # hit@k + MRR baseline against the seeded data
```

The checked-in [evals/golden.jsonl](evals/golden.jsonl) matches the seed documents, so a fresh clone gets a meaningful retrieval baseline immediately; edit both to match your own corpus. Run the evals before and after touching chunking, embeddings, rewriting, fusion, or reranking — this is the regression gate that makes the RAG "self-improving" in practice.

### Tests

```powershell
pytest    # 48 tests over the pure logic: chunking (text+code), RRF, all LLM-output parsers, auth, metrics
```

## Deliberately deferred (and why)

- **Playwright browser automation** — the research agent covers web information needs server-side; driving a real browser adds a large security/ops surface that deserves its own design pass (sandboxing, action confirmation).
- **tree-sitter AST parsing** — native extension; blocked by Application Control policies on locked-down Windows machines (the same policy required the `uuid_utils` shim in [_compat.py](src/nexus/_compat.py)). The heuristic chunker in [code/chunker.py](src/nexus/code/chunker.py) shares its signature.
- **Neo4j knowledge graph** — earns its place when code-relationship queries ("every API touching the Orders table") become a real workload, i.e. deep in Phase 4 usage, not before.
- **Alembic migrations** — `create_all` is right while the schema is still moving; generate the initial Alembic revision against a live DB once it stabilizes.
- **RBAC / organizations / Kafka** — multi-tenant concerns, pointless before a second user exists.

## Design rules this codebase follows

1. Every external dependency (LLM, embeddings, reranker, storage) sits behind a small interface — swapping providers is a one-file change.
2. Every LLM output that machines consume goes through structured outputs **plus** a pure, unit-tested parser that tolerates garbage.
3. Failures in auxiliary intelligence (rewriting, reranking, grading, memory extraction) degrade to the simpler behavior — they never fail the user's request.
4. New capability = new node/edge in the graph, not a new service, until scale says otherwise.
