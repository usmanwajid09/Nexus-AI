from nexus.code.chunker import LANGUAGE_BY_EXTENSION, chunk_code

PY_SOURCE = '''\
import os

def first():
    return 1

def second():
    """Docstring."""
    return 2

class Widget:
    def method(self):
        return 3
'''


def test_python_splits_at_definitions():
    chunks = chunk_code(PY_SOURCE, language="python", max_chars=60)
    joined = "\n".join(chunks)
    assert "def first" in joined and "def second" in joined and "class Widget" in joined
    # small max_chars forces the definitions into separate chunks
    assert len(chunks) >= 3


def test_small_file_stays_whole():
    chunks = chunk_code(PY_SOURCE, language="python", max_chars=5000)
    assert len(chunks) == 1


def test_respects_max_chars():
    big = "\n\n".join(f"def f{i}():\n    return {i}" for i in range(100))
    chunks = chunk_code(big, language="python", max_chars=300)
    assert all(len(c) <= 300 for c in chunks)


def test_unknown_language_falls_back_to_text_chunking():
    chunks = chunk_code("# Title\n\nSome prose paragraph.", language="markdown")
    assert chunks == ["# Title\n\nSome prose paragraph."]


def test_go_boundaries():
    src = "package main\n\nfunc A() int {\n\treturn 1\n}\n\nfunc B() int {\n\treturn 2\n}\n"
    chunks = chunk_code(src, language="go", max_chars=40)
    joined = "\n".join(chunks)
    assert "func A" in joined and "func B" in joined


def test_extension_map_has_common_languages():
    assert LANGUAGE_BY_EXTENSION[".py"] == "python"
    assert LANGUAGE_BY_EXTENSION[".ts"] == "typescript"
    assert LANGUAGE_BY_EXTENSION[".go"] == "go"
