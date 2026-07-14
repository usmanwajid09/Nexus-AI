"""Paragraph-aware text chunking. Pure functions — no I/O, fully unit-testable."""


def chunk_text(text: str, *, max_chars: int = 1600, overlap: int = 200) -> list[str]:
    """Split text into chunks of at most max_chars.

    Packs whole paragraphs together while they fit; paragraphs longer than
    max_chars are hard-split with a trailing overlap so no sentence context is
    lost at the boundary.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    # Overlap only applies when hard-splitting oversized paragraphs; when a small
    # max_chars collides with the (larger) default overlap, shrink the overlap
    # instead of failing the call.
    if overlap >= max_chars:
        overlap = max_chars // 4

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(para, max_chars=max_chars, overlap=overlap))
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)
    return chunks


def _hard_split(text: str, *, max_chars: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    step = max_chars - overlap
    start = 0
    while start < len(text):
        pieces.append(text[start : start + max_chars])
        if start + max_chars >= len(text):
            break
        start += step
    return pieces
