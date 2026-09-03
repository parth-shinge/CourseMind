"""Tests for Phase 7 error-handling additions.

Covers the new input validation and atomic-write logic added during
the Phase 7 polish pass.  All tests are self-contained (use tmp dirs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# notes_store: atomic writes and validation
# ---------------------------------------------------------------------------

class TestNotesStoreValidation:
    """Tests for save_transcript / save_notes input validation."""

    def test_save_transcript_empty_id_raises(self, tmp_path: Path) -> None:
        from app.pipeline.notes_store import NotesStoreError, save_transcript
        with pytest.raises(NotesStoreError, match="lecture_id cannot be empty"):
            save_transcript("", [{"start": 0, "end": 1, "text": "hi"}])

    def test_save_transcript_whitespace_id_raises(self, tmp_path: Path) -> None:
        from app.pipeline.notes_store import NotesStoreError, save_transcript
        with pytest.raises(NotesStoreError, match="lecture_id cannot be empty"):
            save_transcript("   ", [])

    def test_save_notes_empty_id_raises(self) -> None:
        from app.pipeline.notes_store import NotesStoreError, save_notes
        with pytest.raises(NotesStoreError, match="lecture_id cannot be empty"):
            save_notes("", [{"section_title": "A", "text": "x"}], "# Notes", {})

    def test_save_notes_empty_sections_raises(self) -> None:
        from app.pipeline.notes_store import NotesStoreError, save_notes
        with pytest.raises(NotesStoreError, match="notes_sections cannot be empty"):
            save_notes("test_id", [], "# Notes", {})

    def test_save_notes_empty_markdown_raises(self) -> None:
        from app.pipeline.notes_store import NotesStoreError, save_notes
        with pytest.raises(NotesStoreError, match="notes_markdown cannot be empty"):
            save_notes("test_id", [{"section_title": "A", "text": "x"}], "", {})

    def test_atomic_write_no_partial_on_crash(self, tmp_path: Path) -> None:
        """Verify that _atomic_write_text doesn't leave partial files."""
        from app.pipeline.notes_store import _atomic_write_text

        target = tmp_path / "test.json"

        # First: write a good file
        _atomic_write_text(target, '{"status": "original"}')
        assert target.exists()
        assert json.loads(target.read_text(encoding="utf-8"))["status"] == "original"

        # Second: simulate a crash during write by patching os.replace to fail
        with patch("app.pipeline.notes_store.os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                _atomic_write_text(target, '{"status": "new"}')

        # Original file should be intact (not corrupted)
        assert json.loads(target.read_text(encoding="utf-8"))["status"] == "original"

        # No leftover .tmp files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


# ---------------------------------------------------------------------------
# chunker: input validation
# ---------------------------------------------------------------------------

class TestChunkerValidation:
    def test_empty_lecture_id_raises(self) -> None:
        from app.rag.chunker import chunk_notes
        with pytest.raises(ValueError, match="lecture_id cannot be empty"):
            chunk_notes("", [{"section_title": "A", "timestamp_start": 0, "timestamp_end": 1, "text": "x"}])

    def test_empty_sections_raises(self) -> None:
        from app.rag.chunker import chunk_notes
        with pytest.raises(ValueError, match="sections list is empty"):
            chunk_notes("test_lecture", [])


# ---------------------------------------------------------------------------
# embedder: chunk validation
# ---------------------------------------------------------------------------

class TestEmbedderValidation:
    def test_missing_text_key_raises(self) -> None:
        from app.rag.embedder import embed
        # Patch model to avoid loading the real model for this validation test
        with patch("app.rag.embedder._get_model"):
            with pytest.raises(ValueError, match="missing required 'text' key"):
                embed([{"section_title": "A"}])  # no "text" key


# ---------------------------------------------------------------------------
# vector_store: query validation
# ---------------------------------------------------------------------------

class TestVectorStoreValidation:
    def test_empty_query_raises(self) -> None:
        from app.rag.vector_store import query
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            query("")

    def test_whitespace_query_raises(self) -> None:
        from app.rag.vector_store import query
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            query("   ")

    def test_upsert_missing_keys_raises(self) -> None:
        from app.rag.vector_store import upsert
        bad_chunk = {"chunk_id": "x", "text": "hello"}  # missing several keys
        with pytest.raises(ValueError, match="missing required keys"):
            upsert([bad_chunk])


# ---------------------------------------------------------------------------
# generator: query validation
# ---------------------------------------------------------------------------

class TestGeneratorValidation:
    def test_empty_query_raises(self) -> None:
        from app.rag.generator import answer
        with pytest.raises(ValueError, match="query cannot be empty"):
            answer("", [])

    def test_whitespace_query_raises(self) -> None:
        from app.rag.generator import answer
        with pytest.raises(ValueError, match="query cannot be empty"):
            answer("   ", [])


# ---------------------------------------------------------------------------
# retriever: query validation
# ---------------------------------------------------------------------------

class TestRetrieverValidation:
    def test_empty_query_raises(self) -> None:
        from app.rag.retriever import retrieve
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            retrieve("")

    def test_whitespace_query_raises(self) -> None:
        from app.rag.retriever import retrieve
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            retrieve("   ")


# ---------------------------------------------------------------------------
# load_and_chunk: missing lecture error
# ---------------------------------------------------------------------------

class TestLoadAndChunkError:
    def test_missing_lecture_raises_file_not_found(self) -> None:
        from app.rag.chunker import load_and_chunk
        with pytest.raises(FileNotFoundError):
            load_and_chunk("nonexistent_lecture_xyz_12345")
