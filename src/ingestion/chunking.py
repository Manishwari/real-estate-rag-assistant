"""
Chunks Documents into overlapping, size-bounded text chunks suitable
for embedding. Chunking is done on sentence boundaries (falling back to
whitespace) so chunks stay coherent instead of splitting mid-sentence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    format: str
    category: str
    text: str
    chunk_index: int


def _split_sentences(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(para) if s.strip())
    return sentences


def chunk_document(
    doc: Document,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """
    Greedily packs sentences into chunks up to `chunk_size` characters,
    carrying `chunk_overlap` characters of trailing context into the next
    chunk so retrieval doesn't lose context at chunk boundaries.
    """
    sentences = _split_sentences(doc.text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    idx = 0

    def flush():
        nonlocal current, current_len, idx
        if not current:
            return
        text = " ".join(current)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::chunk{idx}",
                doc_id=doc.doc_id,
                source=doc.source,
                format=doc.format,
                category=doc.category,
                text=text,
                chunk_index=idx,
            )
        )
        idx += 1

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            flush()
            # carry overlap: keep trailing sentences whose combined length
            # is <= chunk_overlap to seed the next chunk
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            current = overlap_sentences
            current_len = overlap_len
        current.append(sentence)
        current_len += len(sentence)

    flush()
    return chunks


def chunk_documents(docs: list[Document], **kwargs) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, **kwargs))
    return all_chunks
