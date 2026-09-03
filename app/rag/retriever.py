"""Retrieval module for RAG assistant.

Thin wrapper around vector_store.query() that adds relevance-threshold
filtering.  Chunks whose L2 distance exceeds RELEVANCE_THRESHOLD are
discarded — if *all* chunks fail the filter, an empty list is returned,
which signals generator.py to return the "not covered" response without
calling the LLM at all.

Contract:
    retrieve(query_text: str, top_k: int = 5) -> List[ScoredChunk]

    Returns only chunks whose L2 distance score is <= RELEVANCE_THRESHOLD.
    ScoredChunk is (Chunk, float) as defined in vector_store.py.
"""

from __future__ import annotations

import sys
from typing import List

from app.rag.vector_store import ScoredChunk, query as vs_query

# ---------------------------------------------------------------------------
# Relevance threshold (L2 / Euclidean distance — lower is more similar)
# ---------------------------------------------------------------------------
# Picked empirically from Phase 3 testing on Crash Course Physics #1:
#   - On-topic queries ("what is acceleration") score ~0.8–1.0
#   - Off-topic queries ("how do I bake a cake")  score ~1.8+
#
# A threshold of 1.3 sits comfortably between those two clusters, rejecting
# clearly off-topic chunks while keeping reasonably relevant ones.  This
# value may need tuning as more lectures are indexed or if the embedding
# model changes.
RELEVANCE_THRESHOLD: float = 1.3


def retrieve(query_text: str, top_k: int = 5) -> List[ScoredChunk]:
    """Retrieve top-k relevant course chunks for a student query.

    Queries the vector store for the closest *top_k* chunks, then filters
    out any chunk whose L2 distance exceeds RELEVANCE_THRESHOLD.

    Args:
        query_text: The student question.
        top_k: Maximum number of relevant chunks to return.

    Returns:
        List of (Chunk, distance) tuples that pass the relevance filter,
        sorted by ascending distance (most relevant first).  An empty list
        means the question is not covered in the indexed material.

    Raises:
        ValueError: If query_text is empty or whitespace-only.
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text cannot be empty or whitespace-only.")

    raw_results = vs_query(query_text, k=top_k)

    # Filter by relevance threshold
    filtered = [
        (chunk, score)
        for chunk, score in raw_results
        if score <= RELEVANCE_THRESHOLD
    ]

    return filtered


# ---------------------------------------------------------------------------
# CLI entry-point: python -m app.rag.retriever "<question>" [top_k]
# ---------------------------------------------------------------------------

def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS for display."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.rag.retriever \"<question>\" [top_k]")
        sys.exit(1)

    question = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Query: \"{question}\"  (top-{k}, threshold={RELEVANCE_THRESHOLD})\n")

    results = retrieve(question, top_k=k)

    # Also show the raw (unfiltered) results for comparison
    raw = vs_query(question, k=k)
    total_raw = len(raw)
    total_passed = len(results)

    print(f"Raw results from vector store: {total_raw}")
    print(f"Passed relevance threshold:    {total_passed}")
    print(f"Threshold:                     {RELEVANCE_THRESHOLD}")
    print()

    if not raw:
        print("  (no results — vector store is empty)")
    else:
        for i, (chunk, score) in enumerate(raw, 1):
            passed = "[PASS]" if score <= RELEVANCE_THRESHOLD else "[FAIL]"
            ts = _format_timestamp(chunk["timestamp_start"])
            print(f"  [{i}]  score={score:.4f}  {passed}")
            print(f"       section:   {chunk['section_title']}")
            print(f"       timestamp: {ts}")
            text_preview = chunk["text"][:150].replace("\n", " ")
            if len(chunk["text"]) > 150:
                text_preview += "..."
            print(f"       text:      {text_preview}")
            print()

    if total_passed == 0:
        print("=> No chunks passed threshold — generator will return 'not covered' response.")
    else:
        print(f"=> {total_passed} chunk(s) will be sent to the generator.")
