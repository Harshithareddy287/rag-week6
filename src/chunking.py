# Chunk size / overlap rationale:
# CHUNK_SIZE = 500 characters is roughly 110-125 tokens for this English
# policy text, safely under all-MiniLM-L6-v2's 256-token limit (and close to
# the ~128 tokens it was trained on), so nothing gets truncated when embedded.
# CHUNK_OVERLAP = 75 characters repeats the end of one chunk at the start of
# the next, so a sentence that crosses a split keeps its context.
# Sizes apply to the chunk body; the "[title › section] " prefix is added after.

import re
from pathlib import Path

from src import config


def _is_table(block):
    lines = block.strip().splitlines()
    return bool(lines) and all(line.lstrip().startswith("|") for line in lines)


def _tail(text):
    """ Last CHUNK_OVERLAP chars, starting on a word boundary. Never from a table."""
    if "|" in text[-config.CHUNK_OVERLAP:]:
        return ""
    tail = text[-config.CHUNK_OVERLAP:]
    return tail.split(" ", 1)[1] if " " in tail else tail


# Split levels: paragraphs, then sentences, then words.
_LEVELS = [
    (lambda t: t.split("\n\n"), "\n\n"),
    (lambda t: re.split(r"(?<=[.!?])\s+", t), " "),
    (lambda t: t.split(), " "),
]


def _split(text, level=0):
    """ Recursively split text into pieces <= CHUNK_SIZE, with overlap. Tables stay whole."""
    if len(text) <= config.CHUNK_SIZE or level >= len(_LEVELS):
        return [text]

    split_fn, sep = _LEVELS[level]
    pieces = []
    for p in split_fn(text):
        p = p.strip()
        if not p:
            continue
        if _is_table(p) or len(p) <= config.CHUNK_SIZE:
            pieces.append(p)
        else:
            pieces.extend(_split(p, level + 1))

    chunks, cur = [], ""
    for p in pieces:
        candidate = f"{cur}{sep}{p}" if cur else p
        if not cur or len(candidate) <= config.CHUNK_SIZE:
            cur = candidate
            continue
        chunks.append(cur)
        tail = _tail(cur)
        with_tail = f"{tail}{sep}{p}"
        cur = with_tail if tail and len(with_tail) <= config.CHUNK_SIZE else p
    if cur:
        chunks.append(cur)
    return chunks


def _parse_doc(path):
    """ Return (title, header_metadata, [(heading, body), ...])."""
    title, meta, sections = "", {}, []
    heading, body = None, []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## ") or line.startswith("### "):
            if heading is not None:
                sections.append((heading, "\n".join(body).strip()))
            heading, body = line.lstrip("#").strip(), []
        elif heading is None:
            if ":" in line:  # header lines like "Version: 2.0"
                key, value = line.split(":", 1)
                meta[key.strip().lower()] = value.strip()
        else:
            body.append(line)
    if heading is not None:
        sections.append((heading, "\n".join(body).strip()))
    return title, meta, sections


def chunk_documents(docs_dir):
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        title, meta, sections = _parse_doc(path)
        i = 0
        for heading, body in sections:
            if not body:  # e.g. "## 3. Expense Rules" that only holds ### rules
                continue
            for piece in _split(body):
                chunks.append({
                    "text": f"[{title} › {heading}] {piece}",
                    "metadata": {
                        "doc_name": path.name,
                        "section": heading,
                        "version": meta.get("version", ""),
                        "effective_date": meta.get("effective date", ""),
                        "chunk_id": f"{path.name}#{i}",
                    },
                })
                i += 1
    return chunks
