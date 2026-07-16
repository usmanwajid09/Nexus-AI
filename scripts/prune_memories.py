"""Prune faded memories.

Usage:
    python scripts/prune_memories.py [--days 130] [--dry-run]

Deletes memories that were never recalled (access_count = 0) and are older
than the cutoff. The default cutoff is the age at which an unreinforced
memory's decay weight drops below ~5% (half_life * log2(20) ~ 4.32 half-lives;
130 days at the default 30-day half-life). Reinforced memories are never
pruned by this script — recall keeps them alive by design.
"""

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select

from nexus.db.models import Memory
from nexus.db.session import get_session_factory


async def run(days: float, dry_run: bool) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    session_factory = get_session_factory()
    async with session_factory() as session:
        condition = (Memory.access_count == 0) & (Memory.created_at < cutoff)
        count = (await session.scalar(select(func.count()).select_from(Memory).where(condition))) or 0
        if dry_run:
            print(f"would prune {count} memories (never recalled, older than {days:g} days)")
            return
        await session.execute(delete(Memory).where(condition))
        await session.commit()
        print(f"pruned {count} memories (never recalled, older than {days:g} days)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=130.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.days, args.dry_run))


if __name__ == "__main__":
    main()
