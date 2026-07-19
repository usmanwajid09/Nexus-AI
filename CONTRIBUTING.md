# Contributing to Nexus AI

Thanks for your interest. This project follows a small set of rules that keep the codebase healthy and reviewable.

## Ground rules

1. **Every external dependency behind a small interface.** LLMs, embeddings, rerankers, storage — swapping providers should be a one-file change.
2. **Every LLM output that machines consume goes through structured outputs plus a pure defensive parser** that tolerates garbage.
3. **Auxiliary intelligence degrades gracefully** — rewriting, reranking, grading, memory extraction — none of them may fail the user's request.
4. **New capability = new node or edge in the graph**, not a new service, until scale actually demands otherwise.

## Development setup

```bash
docker compose up -d          # Postgres + pgvector
cp .env.example .env          # set ANTHROPIC_API_KEY
pip install -e ".[dev]"
uvicorn nexus.api.main:app --reload
```

## Before opening a PR

- `make test` — 60+ unit tests over pure logic; should stay under two seconds
- `make lint` — ruff on `src`, `tests`, `scripts`, `migrations`
- If you touched retrieval (chunking, embeddings, rewriting, fusion, reranking, grading), run `make evals` before and after — the harness is the regression gate
- Write tests for anything that is a pure function
- Keep every function short enough to fit on one screen; when it doesn't, extract

## Commit style

- Imperative present tense: "Add /version endpoint", not "Added" or "Adds"
- First line ≤ 72 characters
- Body wraps at ~72 columns, explains *why* the change was needed
- Group related edits into one commit; avoid mixing unrelated changes

## Code style

- Python 3.12+, `ruff` for formatting and linting (config in `pyproject.toml`)
- Type hints everywhere; `Mapped[...]` for ORM columns
- Prefer `async def` for anything that touches I/O
- Prefer pure functions for anything that can be one

## Reporting issues

Bug reports should include: what you ran, what happened, what you expected, the exact traceback, and — where possible — a minimal repro.
