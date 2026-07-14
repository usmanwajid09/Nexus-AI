"""Golden-set retrieval evaluation.

Usage:
    python scripts/run_evals.py [evals/golden.jsonl] [--k 6]

Each line of the golden file:
    {"question": "...", "expected_source": "<document title>"}

Reports hit@k and MRR over the golden set against the live database. This is
the regression gate for retrieval changes: run it before and after touching
chunking, embeddings, rewriting, fusion, or reranking.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from nexus.config import get_settings
from nexus.db.session import get_session_factory
from nexus.embeddings import get_embedder
from nexus.evals.metrics import hit_at_k, mean, reciprocal_rank
from nexus.rag.retriever import retrieve


async def run(golden_path: Path, k: int) -> None:
    settings = get_settings()
    embedder = get_embedder(settings)
    session_factory = get_session_factory()

    cases = [json.loads(line) for line in golden_path.read_text().splitlines() if line.strip()]
    if not cases:
        sys.exit(f"no cases in {golden_path}")

    hits: list[float] = []
    rrs: list[float] = []
    for case in cases:
        question, expected = case["question"], case["expected_source"]
        async with session_factory() as session:
            chunks = await retrieve(session, embedder, [question], limit=k)
        titles = [c.document_title for c in chunks]
        hit = hit_at_k(titles, {expected}, k)
        rr = reciprocal_rank(titles, {expected})
        hits.append(1.0 if hit else 0.0)
        rrs.append(rr)
        print(f"{'HIT ' if hit else 'MISS'} rr={rr:.2f}  {question[:70]}")

    print(f"\ncases={len(cases)}  hit@{k}={mean(hits):.2%}  MRR={mean(rrs):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("golden", nargs="?", default="evals/golden.jsonl")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        sys.exit(
            f"{golden_path} not found. The checked-in evals/golden.jsonl matches the "
            "demo data from scripts/seed_demo.py; edit it to match your own documents."
        )
    asyncio.run(run(golden_path, args.k))


if __name__ == "__main__":
    main()
