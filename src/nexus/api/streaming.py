"""Server-Sent Events formatting. Pure — no I/O, unit-tested."""

import json
from typing import Any


def sse_format(event: str, data: dict[str, Any]) -> str:
    """Render one SSE frame. Data is JSON on a single line, as SSE requires."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"
