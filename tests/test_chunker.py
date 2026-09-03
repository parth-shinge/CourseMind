"""Tests for the RAG chunking, embedding, and vector store pipeline.

These tests are NON-MOCKED — they use real data (the existing notes JSON)
and real local models (sentence-transformers, ChromaDB).  Everything runs
locally with zero LLM/API calls.

Covers:
  - Chunk count and shape (all required keys present, correct types)
  - chunk_id format ("{lecture_id}-{3-digit zero-padded index}")
  - Long-section splitting behaviour (MAX_CHUNK_WORDS threshold)
  - Idempotent upsert (indexing the same lecture twice doesn't double count)
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.rag.chunker import (
    MAX_CHUNK_WORDS,
    Chunk,
    _split_section_text,
    _word_count,
    chunk_notes,
    load_and_chunk,
)

# The real lecture ID from Phase 2 test data
LECTURE_ID = "motion_in_a_straight_line_crash_course_physics_1_zm8ecpbuqye"

# Required keys in every Chunk (ARCHITECTURE.md Section 6)
REQUIRED_CHUNK_KEYS = {
    "chunk_id",
    "lecture_id",
    "section_title",
    "timestamp_start",
    "timestamp_end",
    "text",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def real_sections() -> List[Dict[str, Any]]:
    """Load sections from the real notes JSON on disk."""
    from app.config import settings

    json_path = settings.NOTES_DIR / f"{LECTURE_ID}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sections"]


@pytest.fixture()
def real_chunks(real_sections: List[Dict[str, Any]]) -> List[Chunk]:
    """Produce chunks from the real sections."""
    return chunk_notes(LECTURE_ID, real_sections)


@pytest.fixture()
def tmp_vector_db(tmp_path: Path):
    """Provide a temporary ChromaDB directory and reset singletons after use."""
    from app.config import Settings
    from app.rag import vector_store

    # Temporarily override the VECTOR_DB_DIR setting
    original_dir = vector_store.settings.VECTOR_DB_DIR
    # We can't mutate frozen Settings, so we patch the module-level reference
    # in vector_store directly.
    import app.rag.vector_store as vs_module

    # Create a temporary settings-like object with a different VECTOR_DB_DIR
    tmp_db_dir = tmp_path / "vector_db"
    tmp_db_dir.mkdir(parents=True, exist_ok=True)

    # Monkey-patch the settings reference used by the vector_store module
    old_settings = vs_module.settings
    patched = Settings.__new__(Settings)
    object.__setattr__(patched, "VECTOR_DB_DIR", tmp_db_dir)
    # Copy other fields from the original
    for field_name in [
        "PROJECT_ROOT", "DATA_DIR", "RAW_VIDEOS_DIR", "TRANSCRIPTS_DIR",
        "NOTES_DIR", "LLM_PROVIDER", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
        "GEMINI_MODEL", "ANTHROPIC_MODEL", "WHISPER_MODEL_SIZE", "WHISPER_DEVICE",
    ]:
        object.__setattr__(patched, field_name, getattr(old_settings, field_name))

    vs_module.settings = patched
    vs_module.reset_client()

    yield tmp_db_dir

    # Teardown: restore original settings and reset client
    vs_module.settings = old_settings
    vs_module.reset_client()


# ---------------------------------------------------------------------------
# Chunker: count and shape
# ---------------------------------------------------------------------------

class TestChunkShape:
    """Verify that chunks have the correct structure per ARCHITECTURE.md."""

    def test_chunks_not_empty(self, real_chunks: List[Chunk]) -> None:
        """Chunking real notes should produce at least one chunk."""
        assert len(real_chunks) > 0

    def test_at_least_as_many_chunks_as_sections(
        self, real_sections: List[Dict[str, Any]], real_chunks: List[Chunk]
    ) -> None:
        """There should be at least one chunk per section (possibly more if split)."""
        assert len(real_chunks) >= len(real_sections)

    def test_all_required_keys_present(self, real_chunks: List[Chunk]) -> None:
        """Every chunk must contain all keys from the ARCHITECTURE.md contract."""
        for chunk in real_chunks:
            missing = REQUIRED_CHUNK_KEYS - set(chunk.keys())
            assert not missing, (
                f"Chunk {chunk.get('chunk_id', '?')} is missing keys: {missing}"
            )

    def test_correct_value_types(self, real_chunks: List[Chunk]) -> None:
        """Verify types match the contract."""
        for chunk in real_chunks:
            assert isinstance(chunk["chunk_id"], str)
            assert isinstance(chunk["lecture_id"], str)
            assert isinstance(chunk["section_title"], str)
            assert isinstance(chunk["timestamp_start"], (int, float))
            assert isinstance(chunk["timestamp_end"], (int, float))
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"]) > 0

    def test_lecture_id_matches(self, real_chunks: List[Chunk]) -> None:
        """All chunks should carry the correct lecture_id."""
        for chunk in real_chunks:
            assert chunk["lecture_id"] == LECTURE_ID


# ---------------------------------------------------------------------------
# Chunker: chunk_id format
# ---------------------------------------------------------------------------

class TestChunkIdFormat:
    """Verify chunk_id format: '{lecture_id}-{3-digit zero-padded index}'."""

    def test_chunk_id_starts_with_lecture_id(self, real_chunks: List[Chunk]) -> None:
        for chunk in real_chunks:
            assert chunk["chunk_id"].startswith(LECTURE_ID + "-")

    def test_chunk_id_suffix_is_zero_padded(self, real_chunks: List[Chunk]) -> None:
        """The suffix after the last '-' should be a 3-digit zero-padded number."""
        for chunk in real_chunks:
            suffix = chunk["chunk_id"].rsplit("-", 1)[-1]
            assert re.fullmatch(r"\d{3}", suffix), (
                f"chunk_id suffix '{suffix}' is not 3-digit zero-padded"
            )

    def test_chunk_ids_are_unique(self, real_chunks: List[Chunk]) -> None:
        ids = [c["chunk_id"] for c in real_chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"

    def test_chunk_ids_are_sequential(self, real_chunks: List[Chunk]) -> None:
        """Indices should be 1, 2, 3, ... with no gaps."""
        indices = [
            int(c["chunk_id"].rsplit("-", 1)[-1]) for c in real_chunks
        ]
        assert indices == list(range(1, len(real_chunks) + 1))


# ---------------------------------------------------------------------------
# Chunker: long-section splitting
# ---------------------------------------------------------------------------

class TestLongSectionSplitting:
    """Verify that sections exceeding MAX_CHUNK_WORDS are split correctly."""

    def test_no_chunk_exceeds_threshold(self, real_chunks: List[Chunk]) -> None:
        """Every chunk's text should be ≤ MAX_CHUNK_WORDS (unless a single
        sentence exceeds the limit, which shouldn't happen with our data)."""
        for chunk in real_chunks:
            wc = _word_count(chunk["text"])
            assert wc <= MAX_CHUNK_WORDS + 50, (
                f"Chunk {chunk['chunk_id']} has {wc} words "
                f"(threshold is {MAX_CHUNK_WORDS})"
            )

    def test_split_section_text_short(self) -> None:
        """A short text should not be split."""
        short = "This is a short text."
        result = _split_section_text(short)
        assert len(result) == 1
        assert result[0] == short

    def test_split_section_text_long_bullets(self) -> None:
        """A text with many bullet points exceeding the threshold should split."""
        # Create a text that's clearly over MAX_CHUNK_WORDS
        bullets = [f"- This is bullet point number {i} with some extra words to pad" for i in range(80)]
        long_text = "\n".join(bullets)
        assert _word_count(long_text) > MAX_CHUNK_WORDS, "Test setup: text should exceed threshold"

        parts = _split_section_text(long_text)
        assert len(parts) > 1, "Long text should be split into multiple sub-chunks"
        for part in parts:
            assert _word_count(part) <= MAX_CHUNK_WORDS + 50

    def test_split_preserves_all_content(self) -> None:
        """Splitting and re-joining should preserve all original bullet points."""
        bullets = [f"- Point {i}" for i in range(80)]
        long_text = "\n".join(bullets)

        parts = _split_section_text(long_text)
        reconstructed_bullets = []
        for part in parts:
            reconstructed_bullets.extend(part.split("\n"))

        assert len(reconstructed_bullets) == len(bullets)

    def test_sub_chunks_inherit_section_metadata(self) -> None:
        """When a section is split, all sub-chunks should have the same
        section_title and timestamp range."""
        # Build a section with lots of text
        bullets = [f"- Bullet {i} with additional words for padding purposes" for i in range(80)]
        long_text = "\n".join(bullets)
        section = {
            "section_title": "Test Section",
            "timestamp_start": 100.0,
            "timestamp_end": 200.0,
            "text": long_text,
        }
        chunks = chunk_notes("test-lecture", [section])
        assert len(chunks) > 1, "Should produce multiple chunks from long section"

        for chunk in chunks:
            assert chunk["section_title"] == "Test Section"
            assert chunk["timestamp_start"] == 100.0
            assert chunk["timestamp_end"] == 200.0
            assert chunk["lecture_id"] == "test-lecture"


# ---------------------------------------------------------------------------
# Chunker: load_and_chunk integration
# ---------------------------------------------------------------------------

class TestLoadAndChunk:
    """Test the convenience load_and_chunk function with real data."""

    def test_load_and_chunk_returns_chunks(self) -> None:
        chunks = load_and_chunk(LECTURE_ID)
        assert len(chunks) > 0
        assert all(REQUIRED_CHUNK_KEYS <= set(c.keys()) for c in chunks)


# ---------------------------------------------------------------------------
# Vector store: idempotent upsert
# ---------------------------------------------------------------------------

class TestIdempotentUpsert:
    """Indexing the same lecture twice should NOT double the chunk count."""

    def test_upsert_idempotent(
        self, real_chunks: List[Chunk], tmp_vector_db: Path
    ) -> None:
        from app.rag.vector_store import _get_collection, upsert

        # First upsert
        upsert(real_chunks)
        count_after_first = _get_collection().count()

        # Second upsert (same data)
        upsert(real_chunks)
        count_after_second = _get_collection().count()

        assert count_after_first == count_after_second, (
            f"Upsert is not idempotent: {count_after_first} -> {count_after_second}"
        )
        assert count_after_first == len(real_chunks)

    def test_upsert_count_matches_chunks(
        self, real_chunks: List[Chunk], tmp_vector_db: Path
    ) -> None:
        from app.rag.vector_store import _get_collection, upsert

        upsert(real_chunks)
        assert _get_collection().count() == len(real_chunks)
