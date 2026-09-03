"""Structured append-only logging for pilot-study data collection.

Writes one JSON object per line (JSONL) to:
  data/logs/query_log.jsonl   — every RAG query + retrieval result
  data/logs/ingest_log.jsonl  — every ingestion attempt with per-stage timing

Design constraints:
  - JSONL format: safe for append-only concurrent writes; survives partial
    writes on crash (each line is a complete, independent JSON record).
  - Both functions are *silent-fail*: any I/O error is printed as a warning
    and swallowed so logging never crashes the calling UI code.
  - data/logs/ is created automatically on first use.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LOGS_DIR: Path = settings.LOGS_DIR
QUERY_LOG: Path = LOGS_DIR / "query_log.jsonl"
INGEST_LOG: Path = LOGS_DIR / "ingest_log.jsonl"


def _ensure_logs_dir() -> None:
    """Create data/logs/ if it doesn't exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a single JSON record as one line to *path*."""
    _ensure_logs_dir()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_query(
    question: str,
    retrieved_chunks: list,
    filtered_out_count: int,
    answer: str,
    sources: list,
    latency_seconds: float,
    used_llm: bool,
) -> None:
    """Append one RAG query record to data/logs/query_log.jsonl.

    Args:
        question: The student's question text.
        retrieved_chunks: List of (Chunk, score) tuples that *passed* the
            relevance filter (i.e. were sent to the generator).
        filtered_out_count: Number of raw vector-store results that were
            discarded by the relevance threshold filter.
        answer: The answer text returned by the generator.
        sources: The deduplicated source list from the generator response.
        latency_seconds: Wall-clock time for the full retrieve+answer call.
        used_llm: True if the LLM was actually called; False when the
            generator short-circuited with the "not covered" response.
    """
    try:
        # Compact chunk metadata — skip full text to keep logs small
        chunk_meta = [
            {
                "lecture_id": chunk.get("lecture_id", ""),
                "section_title": chunk.get("section_title", ""),
                "score": round(float(score), 6),
            }
            for chunk, score in retrieved_chunks
        ]

        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "retrieved_count": len(retrieved_chunks),
            "filtered_out_count": filtered_out_count,
            "chunks": chunk_meta,
            "answer": answer,
            "sources": sources,
            "latency_seconds": round(latency_seconds, 4),
            "used_llm": used_llm,
        }
        _append_jsonl(QUERY_LOG, record)
    except Exception:
        print(
            f"[logger] WARNING: failed to write query log — "
            f"{traceback.format_exc(limit=1).strip()}"
        )


def log_ingestion(
    lecture_id: str,
    source: str,
    stage_durations: Dict[str, float],
    success: bool,
    error: Optional[str] = None,
) -> None:
    """Append one ingestion record to data/logs/ingest_log.jsonl.

    Args:
        lecture_id: The sanitized lecture identifier.
        source: The video URL or local path that was ingested.
        stage_durations: Dict mapping stage name to elapsed seconds, e.g.
            {"download": 4.2, "transcribe": 38.1, "summarize": 12.4, "index": 1.3}
        success: True if the full pipeline completed without error.
        error: Optional error message / traceback summary if success=False.
    """
    try:
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lecture_id": lecture_id,
            "source": source,
            "stage_durations_seconds": {
                k: round(float(v), 4) for k, v in stage_durations.items()
            },
            "total_duration_seconds": round(sum(stage_durations.values()), 4),
            "success": success,
            "error": error,
        }
        _append_jsonl(INGEST_LOG, record)
    except Exception:
        print(
            f"[logger] WARNING: failed to write ingest log — "
            f"{traceback.format_exc(limit=1).strip()}"
        )
