"""Embeddings module using sentence-transformers (all-MiniLM-L6-v2).

Provides a thin wrapper that:
  - Loads the embedding model once (module-level singleton) to avoid the
    ~2-second model-load cost on every call.
  - Exposes embed(chunks) -> list[list[float]] for use by vector_store.py.

Design decision – prepending section_title to text before embedding:
  We concatenate "{section_title}: {text}" because the section title carries
  topical signal (e.g. "Newton's Second Law") that may not appear verbatim in
  the bullet-point text.  This gives the embedding a stronger topical anchor,
  improving retrieval for queries that name the topic directly.  The ": "
  separator is a lightweight delimiter the model handles naturally.

Contract (ARCHITECTURE.md Section 3 & 6):
    Model: all-MiniLM-L6-v2  (384-dim, local, free)
    embed(chunks) -> List[List[float]]
"""

from __future__ import annotations

from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Model singleton – loaded once on first import, reused across all calls.
# all-MiniLM-L6-v2 is ~80 MB; the first load downloads to ~/.cache if needed.
# ---------------------------------------------------------------------------
MODEL_NAME: str = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model (singleton).

    Raises:
        RuntimeError: If model loading fails (e.g., missing model, disk space).
    """
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{MODEL_NAME}': {e}. "
                f"Ensure sentence-transformers is installed and the model "
                f"can be downloaded (or is cached at ~/.cache/torch/)."
            ) from e
    return _model


def _build_embedding_input(chunk: Dict[str, Any]) -> str:
    """Construct the string that will be embedded for a single chunk.

    Format: "{section_title}: {text}"
    See module docstring for rationale on including the section title.
    """
    title = chunk.get("section_title", "")
    text = chunk.get("text", "")
    if title:
        return f"{title}: {text}"
    return text


def embed(chunks: List[Dict[str, Any]]) -> List[List[float]]:
    """Convert chunks into vector embeddings using all-MiniLM-L6-v2.

    Args:
        chunks: List of Chunk dicts (must contain at least "text";
                "section_title" is used if present -- see module docstring).

    Returns:
        List of embedding vectors, one per chunk, each a list of 384 floats.
        Order matches the input list.

    Raises:
        ValueError: If any chunk is missing the required "text" key.
        RuntimeError: If the embedding model fails to load.
    """
    if not chunks:
        return []

    # Validate that every chunk has a "text" key
    for i, chunk in enumerate(chunks):
        if "text" not in chunk:
            raise ValueError(
                f"Chunk at index {i} is missing required 'text' key. "
                f"Keys present: {list(chunk.keys())}"
            )

    model = _get_model()
    texts = [_build_embedding_input(c) for c in chunks]
    # encode returns a numpy ndarray of shape (n, 384)
    embeddings = model.encode(texts, show_progress_bar=False)
    return [vec.tolist() for vec in embeddings]
