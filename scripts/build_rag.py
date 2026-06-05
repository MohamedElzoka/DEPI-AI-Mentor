"""
Script to build or rebuild the ChromaDB vector store from the knowledge base.
Run this once before starting the application, and whenever new documents are added.

Usage:
    python scripts/build_rag.py
    python scripts/build_rag.py --rebuild
"""

import argparse
import os
import shutil
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import settings
from rag.retriever import build_vector_store


def main():
    parser = argparse.ArgumentParser(description="Build the DEPI knowledge base vector store.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing vector store and rebuild from scratch.",
    )
    args = parser.parse_args()

    persist_dir = settings.CHROMA_PERSIST_DIR

    if args.rebuild and os.path.exists(persist_dir):
        print(f"[RAG] Deleting existing vector store at '{persist_dir}'...")
        shutil.rmtree(persist_dir)

    print("[RAG] Building vector store...")
    store = build_vector_store(persist_dir)
    print("[RAG] Done. Vector store is ready.")

    # Quick retrieval test
    print("\n[RAG] Running retrieval test...")
    from rag.retriever import RAGRetriever
    retriever = RAGRetriever(persist_dir)
    results = retriever.retrieve_as_list("Python data cleaning pandas")
    print(f"[RAG] Retrieved {len(results)} chunks for test query 'Python data cleaning pandas'.")
    for i, r in enumerate(results[:2], 1):
        print(f"\n  Chunk {i} [{r['source']}]:\n  {r['text'][:200]}...")

    print("\n[RAG] Vector store is ready for use.")


if __name__ == "__main__":
    main()
