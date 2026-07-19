# Common commands. Works with `make` on macOS/Linux; on Windows use `mingw32-make` or run the commands directly.

.PHONY: help install dev test lint db-up db-down seed evals token clean

help:
	@echo "Nexus AI - development commands"
	@echo ""
	@echo "  make install   - install project + dev dependencies"
	@echo "  make dev       - run the API with auto-reload"
	@echo "  make test      - run the test suite"
	@echo "  make lint      - run ruff"
	@echo "  make db-up     - start Postgres + pgvector"
	@echo "  make db-down   - stop Postgres (keeps data)"
	@echo "  make seed      - seed demo documents and memories"
	@echo "  make evals     - run retrieval evaluation (hit@k, MRR)"
	@echo "  make token     - mint a JWT (AUTH_SECRET must be set)"
	@echo "  make clean     - remove caches"

install:
	pip install -e ".[dev]"

dev:
	uvicorn nexus.api.main:app --reload

test:
	pytest -q

lint:
	ruff check src tests scripts migrations

db-up:
	docker compose up -d

db-down:
	docker compose down

seed:
	python scripts/seed_demo.py

evals:
	python scripts/run_evals.py

token:
	python scripts/make_token.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
