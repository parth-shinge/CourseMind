"""Chunking module for structured notes.

Splits structured lecture notes (section-level) into semantically coherent,
timestamped chunks suitable for embedding and retrieval.

Contract (ARCHITECTURE.md Section 6):
    chunk_notes(lecture_id, sections) -> List[Chunk]
    Chunk schema:
    {
      "chunk_id":        str,    # "{lecture_id}-{3-digit zero-padded index}"
      "lecture_id":      str,
      "section_title":   str,
      "timestamp_start": float,
      "timestamp_end":   float,
      "text":            str
    }
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings
from app.pipeline.notes_store import load_notes

# Type alias for Chunk (matches ARCHITECTURE.md Section 6)
Chunk = Dict[str, Any]

# ---------------------------------------------------------------------------
# Splitting threshold
# ---------------------------------------------------------------------------
# The all-MiniLM-L6-v2 model has a 256 word-piece token limit, but word count
# is a cheaper proxy.  350 words is a conservative threshold that keeps most
# chunks well within the model's context window while avoiding unnecessary
# fragmentation.
MAX_CHUNK_WORDS: int = 350


# ---------------------------------------------------------------------------
# Text-splitting helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    """Return approximate word count."""
    return len(text.split())


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences (or bullet-point lines).

    Handles two common formats in our notes:
      1. Bullet-point lines starting with "- " (each is treated as a sentence)
      2. Prose sentences delimited by period/question-mark/exclamation-mark.

    Returns a list of non-empty strings.
    """
    # Our notes are overwhelmingly bullet-point format ("- ...").
    # Split on newlines first; if a line starts with "- ", treat it as one unit.
    lines = text.split("\n")
    sentences: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # If it looks like a bullet point, keep it whole
        if stripped.startswith("- "):
            sentences.append(stripped)
        else:
            # Fall back to sentence splitting for prose text
            parts = re.split(r"(?<=[.!?])\s+", stripped)
            sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def _split_section_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> List[str]:
    """Split a section's text into sub-chunks if it exceeds *max_words*.

    Splitting is done along sentence/bullet boundaries — never mid-sentence.
    Each sub-chunk is guaranteed to be ≤ max_words (unless a single sentence
    exceeds the limit, in which case it gets its own chunk).

    Returns:
        A list of text strings, each suitable for one Chunk.
    """
    if _word_count(text) <= max_words:
        return [text]

    sentences = _split_into_sentences(text)
    sub_chunks: List[str] = []
    current_parts: List[str] = []
    current_wc = 0

    for sentence in sentences:
        s_wc = _word_count(sentence)
        if current_wc + s_wc > max_words and current_parts:
            # Flush the current accumulator
            sub_chunks.append("\n".join(current_parts))
            current_parts = [sentence]
            current_wc = s_wc
        else:
            current_parts.append(sentence)
            current_wc += s_wc

    # Flush remainder
    if current_parts:
        sub_chunks.append("\n".join(current_parts))

    return sub_chunks


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def chunk_notes(lecture_id: str, sections: List[Dict[str, Any]]) -> List[Chunk]:
    """Split structured notes into timestamped chunks for embedding.

    For each section in *sections*, one or more Chunks are produced.  Short
    sections become a single chunk; long sections (>MAX_CHUNK_WORDS) are
    split along sentence/bullet boundaries into multiple sub-chunks.

    **Timestamp limitation**: sub-chunks within the same section inherit the
    section-level timestamp_start / timestamp_end because finer-grained
    timestamps (per-sentence) are not available in our current notes format.
    This means retrieval can point users to the correct *section* of the
    lecture, but not to the exact second within that section.

    Args:
        lecture_id: Unique lecture identifier (e.g. from the notes JSON).
        sections: List of section dicts, each with keys:
            section_title, timestamp_start, timestamp_end, text.

    Returns:
        List of Chunk dicts matching ARCHITECTURE.md Section 6.

    Raises:
        ValueError: If lecture_id is empty or sections list is empty.
    """
    if not lecture_id or not lecture_id.strip():
        raise ValueError("lecture_id cannot be empty.")
    if not sections:
        raise ValueError(
            f"sections list is empty for lecture '{lecture_id}' "
            f"-- nothing to chunk. Did note generation produce any sections?"
        )

    chunks: List[Chunk] = []
    idx = 0  # global chunk counter across all sections

    for section in sections:
        title = section["section_title"]
        ts_start = section["timestamp_start"]
        ts_end = section["timestamp_end"]
        text = section["text"]

        sub_texts = _split_section_text(text)

        for sub_text in sub_texts:
            idx += 1
            chunk: Chunk = {
                "chunk_id": f"{lecture_id}-{idx:03d}",
                "lecture_id": lecture_id,
                "section_title": title,
                "timestamp_start": ts_start,
                "timestamp_end": ts_end,
                "text": sub_text,
            }
            chunks.append(chunk)

    return chunks


def load_and_chunk(lecture_id: str) -> List[Chunk]:
    """Convenience: load notes from disk and chunk them in one call.

    Args:
        lecture_id: Unique lecture identifier.

    Returns:
        List of Chunk dicts.
    """
    _, metadata = load_notes(lecture_id)
    sections = metadata.get("sections", [])
    if not sections:
        raise ValueError(
            f"No 'sections' found in notes metadata for lecture '{lecture_id}'."
        )
    return chunk_notes(lecture_id, sections)


# ---------------------------------------------------------------------------
# CLI entry-point: python -m app.rag.chunker <lecture_id>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.rag.chunker <lecture_id>")
        sys.exit(1)

    lid = sys.argv[1]
    print(f"Loading notes for lecture: {lid}")
    result_chunks = load_and_chunk(lid)

    print(f"\nDone: Created {len(result_chunks)} chunk(s):\n")
    for c in result_chunks:
        word_ct = _word_count(c["text"])
        print(
            f"  {c['chunk_id']}  |  "
            f"{c['section_title'][:50]:<50}  |  "
            f"[{c['timestamp_start']:.1f}s - {c['timestamp_end']:.1f}s]  |  "
            f"{word_ct} words"
        )
