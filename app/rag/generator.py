"""Grounded answer generator module.

Takes a student question and retrieved context chunks, then produces a
grounded answer that ONLY uses the provided context — never falling back
to open-domain knowledge.

Contract (ARCHITECTURE.md Section 6):
    answer(query, retrieved_chunks) -> {
        "answer": str,
        "sources": [
            {"lecture_id": str, "section_title": str, "timestamp_start": float},
            ...
        ]
    }

Grounding Rules (ARCHITECTURE.md Section 10):
    - If retrieved_chunks is empty: return the "not covered" response
      immediately WITHOUT calling the LLM.  This saves quota and
      guarantees correctness.
    - The LLM system prompt STRICTLY forbids answering from general
      knowledge, fabricating information, or contradicting the context.
    - Sources are deduplicated: if multiple chunks come from the same
      section, only one source entry is produced.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

from app.llm_client import get_llm_client
from app.rag.vector_store import ScoredChunk

# Type alias
RAGResponse = Dict[str, Any]

# Response returned when no relevant chunks are found — no LLM call is made
NOT_COVERED_RESPONSE: RAGResponse = {
    "answer": "This question isn't covered in this course's material yet.",
    "sources": [],
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a course learning assistant.  You answer student questions using ONLY
the provided lecture context below.  Follow these rules STRICTLY:

1. Base your answer EXCLUSIVELY on the information in the CONTEXT sections.
2. If the context does not fully answer the question, say so explicitly —
   do NOT fill in gaps from your general knowledge.
3. Never contradict or go beyond what is stated in the provided material.
4. Keep your answer clear, concise, and helpful for a student studying this
   course.
5. Do NOT invent or fabricate sources, references, or information.

Respond with a JSON object in EXACTLY this format (no markdown fences, no
extra keys):
{
  "answer": "<your answer text>"
}
"""


def _build_user_prompt(query: str, chunks: List[ScoredChunk]) -> str:
    """Build the user prompt containing the query and all retrieved context.

    Each chunk is labeled with its section title and timestamp so the LLM
    can reference the source material clearly.
    """
    context_parts: List[str] = []
    for i, (chunk, score) in enumerate(chunks, 1):
        ts_start = chunk["timestamp_start"]
        # Format timestamp as MM:SS for readability
        m, s = divmod(int(ts_start), 60)
        ts_str = f"{m:02d}:{s:02d}"
        context_parts.append(
            f"--- CONTEXT {i} ---\n"
            f"Section: {chunk['section_title']}\n"
            f"Timestamp: {ts_str}\n"
            f"Content:\n{chunk['text']}\n"
        )

    context_block = "\n".join(context_parts)

    return (
        f"STUDENT QUESTION:\n{query}\n\n"
        f"LECTURE CONTEXT:\n{context_block}\n\n"
        f"Answer the student's question using ONLY the context above.  "
        f"If the context doesn't fully cover the question, say so explicitly."
    )


def _deduplicate_sources(chunks: List[ScoredChunk]) -> List[Dict[str, Any]]:
    """Extract and deduplicate source entries from retrieved chunks.

    Deduplication key: (lecture_id, section_title) — if multiple chunks
    come from the same section, we only list it once.  We keep the earliest
    timestamp_start among duplicates.

    Returns:
        List of source dicts matching ARCHITECTURE.md Section 6:
        [{"lecture_id", "section_title", "timestamp_start"}, ...]
    """
    seen: Dict[tuple, Dict[str, Any]] = {}
    for chunk, _score in chunks:
        key = (chunk["lecture_id"], chunk["section_title"])
        if key not in seen:
            seen[key] = {
                "lecture_id": chunk["lecture_id"],
                "section_title": chunk["section_title"],
                "timestamp_start": chunk["timestamp_start"],
            }
        else:
            # Keep the earliest timestamp if we've seen this section before
            existing = seen[key]
            if chunk["timestamp_start"] < existing["timestamp_start"]:
                existing["timestamp_start"] = chunk["timestamp_start"]
    return list(seen.values())


def answer(query: str, retrieved_chunks: List[ScoredChunk]) -> RAGResponse:
    """Generate a grounded answer citing source lectures and timestamps.

    Args:
        query: Student question string.
        retrieved_chunks: List of (Chunk, distance_score) tuples from
            the retriever.  If empty, the "not covered" response is
            returned immediately without calling the LLM.

    Returns:
        Dict matching ARCHITECTURE.md Section 6 RAG answer shape:
        {"answer": str, "sources": [{"lecture_id", "section_title", "timestamp_start"}, ...]}

    Raises:
        ValueError: If query is empty or whitespace-only.
    """
    if not query or not query.strip():
        raise ValueError("query cannot be empty or whitespace-only.")

    # ---- Guard: empty retrieval -> skip LLM entirely ----
    if not retrieved_chunks:
        print("[generator] No relevant chunks -- returning 'not covered' (no LLM call)")
        return NOT_COVERED_RESPONSE.copy()

    # ---- Build prompt ----
    user_prompt = _build_user_prompt(query, retrieved_chunks)

    # ---- Call LLM via the shared client ----

    print(f"[generator] Sending prompt to LLM ({len(retrieved_chunks)} chunk(s) as context)...")
    client = get_llm_client()
    raw_response = client.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=1500,
        response_mime_type="application/json",
    )

    # ---- Parse LLM response ----
    try:
        parsed = json.loads(raw_response)
        answer_text = parsed.get("answer", raw_response)
    except (json.JSONDecodeError, TypeError):
        # If the LLM didn't return valid JSON, use the raw text as the answer
        answer_text = raw_response.strip()

    # ---- Build sources from the *actually retrieved* chunks (not from LLM output) ----
    sources = _deduplicate_sources(retrieved_chunks)

    return {
        "answer": answer_text,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# CLI entry-point: python -m app.rag.generator "<question>" [top_k]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.rag.generator \"<question>\" [top_k]")
        sys.exit(1)

    question = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # Import retriever here to keep the module importable without side-effects
    from app.rag.retriever import retrieve, RELEVANCE_THRESHOLD

    print(f"Question: \"{question}\"")
    print(f"Retrieval: top-{k}, threshold={RELEVANCE_THRESHOLD}\n")

    # Step 1: Retrieve
    chunks = retrieve(question, top_k=k)
    print(f"Retrieved {len(chunks)} relevant chunk(s)")
    if chunks:
        for i, (chunk, score) in enumerate(chunks, 1):
            print(f"  [{i}] score={score:.4f}  section=\"{chunk['section_title']}\"")
    print()

    # Step 2: Generate answer
    result = answer(question, chunks)

    # Step 3: Pretty-print the result
    print("=" * 60)
    print("RAG Response:")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
