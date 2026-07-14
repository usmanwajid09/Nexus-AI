"""Code-aware chunking (Phase 4). Pure functions — no I/O, fully unit-tested.

Splits source files at top-level definition boundaries so a chunk is usually a
whole function/class rather than an arbitrary window. Heuristic and
language-light by design: tree-sitter would be more precise, but it is a native
extension (blocked on locked-down machines this project targets) — it can slot
in later behind this same function signature.
"""

import re

from nexus.rag.chunker import chunk_text

LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".cs": "csharp",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".sql": "sql",
    ".sh": "shell",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
}

# Zero-indent lines that begin a new top-level definition, per language family.
_BOUNDARY_PATTERNS = {
    "python": re.compile(r"^(def |class |async def |@)"),
    "javascript": re.compile(
        r"^(export\s+)?(default\s+)?(async\s+)?(function|class|const|let|var)\b"
    ),
    "typescript": re.compile(
        r"^(export\s+)?(default\s+)?(async\s+)?(function|class|const|let|var|interface|type|enum)\b"
    ),
    "java": re.compile(r"^(public|private|protected|class|interface|enum|abstract|final)\b"),
    "csharp": re.compile(r"^(public|private|protected|internal|class|interface|namespace)\b"),
    "go": re.compile(r"^(func|type|var|const)\b"),
    "rust": re.compile(r"^(pub\s+)?(fn|struct|enum|impl|trait|mod|const|static)\b"),
    "ruby": re.compile(r"^(def |class |module )"),
    "php": re.compile(r"^(function|class|interface|trait)\b"),
    "c": re.compile(r"^\w[\w\s\*]*\("),
    "cpp": re.compile(r"^(\w[\w\s\*:<>]*\(|class |struct |namespace )"),
}


def chunk_code(source: str, *, language: str, max_chars: int = 2400) -> list[str]:
    """Split source code into chunks at top-level definition boundaries.

    Falls back to plain text chunking for languages without a boundary pattern
    (markdown, yaml, json, ...) and for oversized blocks.
    """
    pattern = _BOUNDARY_PATTERNS.get(language)
    if pattern is None:
        return chunk_text(source, max_chars=max_chars)

    blocks = _split_blocks(source, pattern)
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(chunk_text(block, max_chars=max_chars))
            continue
        candidate = f"{current}\n{block}" if current else block
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = block
    if current:
        chunks.append(current)
    return [c for c in (chunk.strip("\n") for chunk in chunks) if c.strip()]


def _split_blocks(source: str, pattern: re.Pattern) -> list[str]:
    lines = source.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if pattern.match(line) and current and any(entry.strip() for entry in current):
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(b) for b in blocks]
