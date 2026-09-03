"""Vector store management module using embedded ChromaDB.

Provides persistent, cross-lecture vector storage so retrieval can search
across all ingested lectures at once.

Storage:
    - PersistentClient at data/vector_db (per ARCHITECTURE.md Section 5).
    - Single collection "course_notes" (not per-lecture) for cross-lecture
      retrieval.

Contract:
    upsert(chunks) -> None
        Embeds chunks via embedder.embed(), stores vectors + full Chunk
        metadata in ChromaDB.  Idempotent: re-running on the same chunk_ids
        updates existing records rather than duplicating them.

    query(query_text, k) -> List[ScoredChunk]
        Returns top-k chunks with similarity scores.  Each result is a
        tuple of (Chunk, score) where score is the ChromaDB L2 distance
        (lower = more similar).

Design notes:
    - ChromaDB stores metadata values as scalars, so all Chunk fields are
      stored as individual metadata keys (not nested).
    - The "documents" field in ChromaDB holds the embedding input string
      (section_title + text), which ChromaDB also uses for deduplication
      and optional full-text search.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

import chromadb

from app.config import settings
from app.rag import embedder
from app.rag.chunker import Chunk, load_and_chunk

# Type alias: a Chunk paired with its L2 distance score (lower = better)
ScoredChunk = Tuple[Chunk, float]

# Collection name — one collection across all lectures
COLLECTION_NAME: str = "course_notes"

# ---------------------------------------------------------------------------
# ChromaDB client singleton
# ---------------------------------------------------------------------------
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """Lazy-initialise the ChromaDB PersistentClient and collection."""
    global _client, _collection
    if _collection is None:
        db_path = str(settings.VECTOR_DB_DIR)
        settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=db_path)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "l2"},  # L2 (Euclidean) distance
        )
    return _collection


def reset_client() -> None:
    """Reset the module-level client/collection singletons.

    Useful in tests that need a fresh ChromaDB state between runs.
    """
    global _client, _collection
    _client = None
    _collection = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert(chunks: List[Chunk]) -> None:
    """Embed and store (or update) chunks in ChromaDB.

    Idempotent: uses chunk_id as the ChromaDB document ID, so re-upserting
    the same chunks overwrites rather than duplicates.

    Args:
        chunks: List of Chunk dicts (matching ARCHITECTURE.md Section 6).

    Raises:
        ValueError: If any chunk is missing required keys.
    """
    if not chunks:
        return

    # Validate chunk structure before expensive embedding
    _REQUIRED_KEYS = {"chunk_id", "lecture_id", "section_title",
                      "timestamp_start", "timestamp_end", "text"}
    for i, chunk in enumerate(chunks):
        missing = _REQUIRED_KEYS - set(chunk.keys())
        if missing:
            raise ValueError(
                f"Chunk at index {i} (id={chunk.get('chunk_id', '?')}) "
                f"is missing required keys: {missing}"
            )

    collection = _get_collection()
    vectors = embedder.embed(chunks)

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    embeddings_list: List[List[float]] = []

    for chunk, vec in zip(chunks, vectors):
        ids.append(chunk["chunk_id"])
        # Store the embedding-input text as the "document" for ChromaDB
        documents.append(embedder._build_embedding_input(chunk))
        metadatas.append({
            "lecture_id": chunk["lecture_id"],
            "section_title": chunk["section_title"],
            "timestamp_start": chunk["timestamp_start"],
            "timestamp_end": chunk["timestamp_end"],
            "text": chunk["text"],
        })
        embeddings_list.append(vec)

    # ChromaDB's upsert is natively idempotent on the id field
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings_list,
    )


def query(query_text: str, k: int = 5) -> List[ScoredChunk]:
    """Retrieve the top-k most relevant chunks for a query string.

    Args:
        query_text: The natural-language question or search string.
        k: Number of results to return (default 5).

    Returns:
        List of (Chunk, distance_score) tuples, sorted by ascending distance
        (most relevant first).  Distance is L2 (Euclidean) -- lower is better.
        An empty list is returned if the collection has no documents.

    Raises:
        ValueError: If query_text is empty or whitespace-only.
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text cannot be empty or whitespace-only.")

    collection = _get_collection()

    # Nothing to search if the collection is empty
    if collection.count() == 0:
        return []

    # Embed the query using the same model + approach as indexing
    query_vec = embedder.embed([{"text": query_text}])[0]

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(k, collection.count()),
        include=["metadatas", "distances"],
    )

    scored_chunks: List[ScoredChunk] = []
    # results is a dict of lists-of-lists (one outer list per query)
    ids_list = results["ids"][0]
    meta_list = results["metadatas"][0]
    dist_list = results["distances"][0]

    for cid, meta, dist in zip(ids_list, meta_list, dist_list):
        chunk: Chunk = {
            "chunk_id": cid,
            "lecture_id": meta["lecture_id"],
            "section_title": meta["section_title"],
            "timestamp_start": meta["timestamp_start"],
            "timestamp_end": meta["timestamp_end"],
            "text": meta["text"],
        }
        scored_chunks.append((chunk, dist))

    return scored_chunks


# ---------------------------------------------------------------------------
# CLI entry-point: python -m app.rag.vector_store <command> [args]
# ---------------------------------------------------------------------------

def _cli_index(lecture_id: str) -> None:
    """CLI: chunk, embed, and upsert a lecture."""
    print(f"Chunking lecture: {lecture_id}")
    chunks = load_and_chunk(lecture_id)
    print(f"  -> {len(chunks)} chunk(s) created")

    print("Embedding and upserting into ChromaDB...")
    upsert(chunks)

    collection = _get_collection()
    print(f"  -> Collection '{COLLECTION_NAME}' now has {collection.count()} document(s)")
    print("Done.")


def _cli_query(query_text: str, k: int = 5) -> None:
    """CLI: query the vector store and print results."""
    print(f"Query: \"{query_text}\"  (top-{k})\n")
    results = query(query_text, k=k)

    if not results:
        print("  (no results)")
        return

    for i, (chunk, score) in enumerate(results, 1):
        print(f"  [{i}]  score={score:.4f}  (L2 distance - lower is better)")
        print(f"       chunk_id:  {chunk['chunk_id']}")
        print(f"       section:   {chunk['section_title']}")
        print(f"       timestamp: [{chunk['timestamp_start']:.1f}s - {chunk['timestamp_end']:.1f}s]")
        # Show a truncated preview of the text
        text_preview = chunk["text"][:200].replace("\n", " ")
        if len(chunk["text"]) > 200:
            text_preview += "..."
        print(f"       text:      {text_preview}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python -m app.rag.vector_store index <lecture_id>\n"
            "  python -m app.rag.vector_store query \"<question>\" [k]"
        )
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "index":
        if len(sys.argv) < 3:
            print("Error: missing <lecture_id>")
            sys.exit(1)
        _cli_index(sys.argv[2])

    elif command == "query":
        if len(sys.argv) < 3:
            print("Error: missing query text")
            sys.exit(1)
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        _cli_query(sys.argv[2], k=top_k)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
