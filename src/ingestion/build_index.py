"""
Build (or rebuild) the vector index from data/raw/*.

Usage:
    python -m src.ingestion.build_index [--reset] [--raw-dir data/raw] [--persist-dir chroma_db]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as `python -m src.ingestion.build_index` from repo root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ingestion.loaders import load_all_documents
from src.ingestion.chunking import chunk_documents
from src.rag.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description="Build the Real Estate RAG vector index.")
    parser.add_argument("--raw-dir", default="data/raw", help="Folder with pdf/docx/html/markdown subfolders")
    parser.add_argument("--persist-dir", default="chroma_db", help="Where to persist the Chroma index")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--reset", action="store_true", help="Wipe existing index before rebuilding")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"ERROR: raw dir {raw_dir} does not exist.")
        sys.exit(1)

    t0 = time.time()
    print(f"Loading documents from {raw_dir} ...")
    docs = load_all_documents(raw_dir)
    print(f"  loaded {len(docs)} documents")

    print("Chunking documents ...")
    chunks = chunk_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"  produced {len(chunks)} chunks")

    print(f"Building index at {args.persist_dir} (this downloads the embedding model on first run) ...")
    retriever = Retriever(persist_dir=args.persist_dir)
    if args.reset:
        retriever.reset()
    retriever.add_chunks(chunks)

    print(f"Done in {time.time() - t0:.1f}s. Index now has {retriever.count()} vectors.")


if __name__ == "__main__":
    main()
