"""Prometheus metrics. Scraped from GET /metrics; dashboards via the
"observability" docker-compose profile (Prometheus + Grafana)."""

from prometheus_client import Counter, Histogram, make_asgi_app

REQUEST_LATENCY = Histogram(
    "nexus_request_seconds",
    "HTTP request latency",
    ["method", "path", "status"],
)

LLM_CALLS = Counter(
    "nexus_llm_calls_total",
    "LLM API calls by kind",
    ["kind"],  # complete | complete_json | vision | research
)

LLM_TOKENS = Counter(
    "nexus_llm_tokens_total",
    "LLM tokens consumed",
    ["direction"],  # input | output
)

RETRIEVAL_CHUNKS = Histogram(
    "nexus_retrieval_chunks",
    "Chunks returned per retrieval",
    buckets=(0, 1, 2, 4, 6, 8, 12),
)

ANSWER_CONFIDENCE = Histogram(
    "nexus_answer_confidence",
    "Groundedness score of generated answers (0-1)",
    buckets=(0.2, 0.4, 0.6, 0.8, 0.9, 1.0),
)

metrics_app = make_asgi_app()


def record_llm_usage(kind: str, usage) -> None:
    LLM_CALLS.labels(kind=kind).inc()
    if usage is not None:
        LLM_TOKENS.labels(direction="input").inc(usage.input_tokens or 0)
        LLM_TOKENS.labels(direction="output").inc(usage.output_tokens or 0)
