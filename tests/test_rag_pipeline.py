"""Unit tests for the RAG pipeline: retriever + generator.

All tests use mocked dependencies (vector store, LLM client) so they
run without an API key or a populated ChromaDB.  No live LLM calls are
made by this test suite.

Coverage:
    - Retriever: threshold filtering (pass, fail, mixed, empty)
    - Generator: empty-chunks-skips-LLM, prompt contains chunk text,
      source deduplication, response shape matches ARCHITECTURE.md
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.rag.chunker import Chunk
from app.rag.vector_store import ScoredChunk


# ---------------------------------------------------------------------------
# Test fixtures: sample chunks and scored chunks
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: str = "lec1-001",
    lecture_id: str = "lec1",
    section_title: str = "Newton's Laws",
    timestamp_start: float = 60.0,
    timestamp_end: float = 120.0,
    text: str = "Force equals mass times acceleration (F = ma).",
) -> Chunk:
    """Create a Chunk dict for testing."""
    return {
        "chunk_id": chunk_id,
        "lecture_id": lecture_id,
        "section_title": section_title,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "text": text,
    }


def _make_scored(chunk: Chunk, score: float) -> ScoredChunk:
    """Wrap a Chunk as a ScoredChunk tuple."""
    return (chunk, score)


# Reusable chunks
CHUNK_A = _make_chunk(
    chunk_id="lec1-001",
    section_title="Newton's Second Law",
    timestamp_start=60.0,
    text="Force equals mass times acceleration. F = ma.",
)
CHUNK_B = _make_chunk(
    chunk_id="lec1-002",
    section_title="Kinematics",
    timestamp_start=180.0,
    text="Velocity is the rate of change of displacement.",
)
CHUNK_C = _make_chunk(
    chunk_id="lec1-003",
    section_title="Newton's Second Law",  # same section as CHUNK_A
    timestamp_start=90.0,
    text="The net force on an object determines its acceleration.",
)


# ============================================================================
# RETRIEVER TESTS
# ============================================================================

class TestRetriever:
    """Tests for app.rag.retriever.retrieve()."""

    @patch("app.rag.retriever.vs_query")
    def test_all_chunks_pass_threshold(self, mock_query: MagicMock) -> None:
        """When all chunks score below threshold, all are returned."""
        from app.rag.retriever import retrieve, RELEVANCE_THRESHOLD

        mock_query.return_value = [
            _make_scored(CHUNK_A, 0.8),
            _make_scored(CHUNK_B, 1.0),
        ]
        result = retrieve("what is force", top_k=5)

        assert len(result) == 2
        assert result[0][1] == 0.8
        assert result[1][1] == 1.0
        mock_query.assert_called_once_with("what is force", k=5)

    @patch("app.rag.retriever.vs_query")
    def test_all_chunks_fail_threshold(self, mock_query: MagicMock) -> None:
        """When all chunks score above threshold, an empty list is returned."""
        from app.rag.retriever import retrieve, RELEVANCE_THRESHOLD

        mock_query.return_value = [
            _make_scored(CHUNK_A, 1.8),
            _make_scored(CHUNK_B, 2.1),
        ]
        result = retrieve("how do I bake a cake", top_k=5)

        assert len(result) == 0

    @patch("app.rag.retriever.vs_query")
    def test_mixed_chunks_partial_pass(self, mock_query: MagicMock) -> None:
        """Only chunks below threshold are kept; others are filtered out."""
        from app.rag.retriever import retrieve, RELEVANCE_THRESHOLD

        mock_query.return_value = [
            _make_scored(CHUNK_A, 0.9),   # pass
            _make_scored(CHUNK_B, 1.5),   # fail (> 1.3)
            _make_scored(CHUNK_C, 1.1),   # pass
        ]
        result = retrieve("what is acceleration", top_k=5)

        assert len(result) == 2
        # Check that only the passing chunks are included
        chunk_ids = [c["chunk_id"] for c, _ in result]
        assert "lec1-001" in chunk_ids
        assert "lec1-003" in chunk_ids
        assert "lec1-002" not in chunk_ids

    @patch("app.rag.retriever.vs_query")
    def test_empty_vector_store(self, mock_query: MagicMock) -> None:
        """When vector store returns nothing, retrieve returns empty list."""
        from app.rag.retriever import retrieve

        mock_query.return_value = []
        result = retrieve("anything", top_k=5)

        assert result == []

    @patch("app.rag.retriever.vs_query")
    def test_threshold_boundary_exact(self, mock_query: MagicMock) -> None:
        """A chunk scoring exactly at the threshold should PASS (<=)."""
        from app.rag.retriever import retrieve, RELEVANCE_THRESHOLD

        mock_query.return_value = [
            _make_scored(CHUNK_A, RELEVANCE_THRESHOLD),  # exactly 1.3
        ]
        result = retrieve("borderline question", top_k=5)

        assert len(result) == 1
        assert result[0][1] == RELEVANCE_THRESHOLD

    @patch("app.rag.retriever.vs_query")
    def test_threshold_boundary_just_above(self, mock_query: MagicMock) -> None:
        """A chunk scoring just above the threshold should FAIL."""
        from app.rag.retriever import retrieve, RELEVANCE_THRESHOLD

        mock_query.return_value = [
            _make_scored(CHUNK_A, RELEVANCE_THRESHOLD + 0.001),
        ]
        result = retrieve("borderline question", top_k=5)

        assert len(result) == 0

    @patch("app.rag.retriever.vs_query")
    def test_top_k_passed_through(self, mock_query: MagicMock) -> None:
        """The top_k parameter is passed through to the vector store."""
        from app.rag.retriever import retrieve

        mock_query.return_value = []
        retrieve("test", top_k=3)

        mock_query.assert_called_once_with("test", k=3)


# ============================================================================
# GENERATOR TESTS
# ============================================================================

class TestGeneratorEmptyChunks:
    """Tests that the generator skips the LLM when chunks are empty."""

    @patch("app.rag.generator.get_llm_client")
    def test_empty_chunks_returns_not_covered(self, mock_get_client: MagicMock) -> None:
        """Empty retrieved_chunks → 'not covered' response, no LLM call."""
        from app.rag.generator import answer, NOT_COVERED_RESPONSE

        result = answer("how do I bake a cake", [])

        assert result["answer"] == NOT_COVERED_RESPONSE["answer"]
        assert result["sources"] == []
        # The LLM client should NOT have been instantiated at all
        mock_get_client.assert_not_called()

    @patch("app.rag.generator.get_llm_client")
    def test_empty_chunks_llm_generate_not_called(self, mock_get_client: MagicMock) -> None:
        """Double-check: even if get_llm_client were somehow called,
        generate() itself must not be invoked for empty chunks."""
        from app.rag.generator import answer

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = answer("irrelevant question", [])

        mock_client.generate.assert_not_called()


class TestGeneratorWithChunks:
    """Tests for generator behavior when chunks ARE provided."""

    def _mock_llm_response(self, answer_text: str) -> str:
        """Return a JSON string like the LLM would produce."""
        return json.dumps({"answer": answer_text})

    @patch("app.rag.generator.get_llm_client")
    def test_prompt_contains_chunk_text(self, mock_get_client: MagicMock) -> None:
        """The user prompt sent to the LLM must contain the retrieved chunk text."""
        from app.rag.generator import answer

        mock_client = MagicMock()
        mock_client.generate.return_value = self._mock_llm_response(
            "Force equals mass times acceleration."
        )
        mock_get_client.return_value = mock_client

        chunks: List[ScoredChunk] = [
            _make_scored(CHUNK_A, 0.8),
            _make_scored(CHUNK_B, 1.0),
        ]
        answer("what is force", chunks)

        # Extract the user_prompt that was sent to generate()
        call_args = mock_client.generate.call_args
        user_prompt = call_args.kwargs.get("user_prompt") or call_args[1].get("user_prompt") or call_args[0][1]

        # Verify that both chunks' text appears in the prompt
        assert CHUNK_A["text"] in user_prompt
        assert CHUNK_B["text"] in user_prompt

    @patch("app.rag.generator.get_llm_client")
    def test_prompt_contains_section_titles(self, mock_get_client: MagicMock) -> None:
        """The prompt must include section titles for grounding context."""
        from app.rag.generator import answer

        mock_client = MagicMock()
        mock_client.generate.return_value = self._mock_llm_response("answer")
        mock_get_client.return_value = mock_client

        chunks: List[ScoredChunk] = [_make_scored(CHUNK_A, 0.8)]
        answer("what is force", chunks)

        call_args = mock_client.generate.call_args
        user_prompt = call_args.kwargs.get("user_prompt") or call_args[1].get("user_prompt") or call_args[0][1]

        assert CHUNK_A["section_title"] in user_prompt

    @patch("app.rag.generator.get_llm_client")
    def test_prompt_contains_timestamps(self, mock_get_client: MagicMock) -> None:
        """The prompt must include timestamps for each chunk."""
        from app.rag.generator import answer

        mock_client = MagicMock()
        mock_client.generate.return_value = self._mock_llm_response("answer")
        mock_get_client.return_value = mock_client

        chunks: List[ScoredChunk] = [_make_scored(CHUNK_A, 0.8)]
        answer("what is force", chunks)

        call_args = mock_client.generate.call_args
        user_prompt = call_args.kwargs.get("user_prompt") or call_args[1].get("user_prompt") or call_args[0][1]

        # timestamp_start=60.0 → "01:00"
        assert "01:00" in user_prompt


class TestGeneratorSourceDeduplication:
    """Tests that sources are properly deduplicated."""

    @patch("app.rag.generator.get_llm_client")
    def test_duplicate_sections_deduplicated(self, mock_get_client: MagicMock) -> None:
        """Multiple chunks from the same section → only one source entry."""
        from app.rag.generator import answer

        mock_client = MagicMock()
        mock_client.generate.return_value = json.dumps({"answer": "F = ma"})
        mock_get_client.return_value = mock_client

        # CHUNK_A and CHUNK_C are both from "Newton's Second Law" in "lec1"
        chunks: List[ScoredChunk] = [
            _make_scored(CHUNK_A, 0.8),
            _make_scored(CHUNK_C, 0.9),
        ]
        result = answer("what is force", chunks)

        assert len(result["sources"]) == 1
        assert result["sources"][0]["section_title"] == "Newton's Second Law"
        assert result["sources"][0]["lecture_id"] == "lec1"

    @patch("app.rag.generator.get_llm_client")
    def test_duplicate_sections_keep_earliest_timestamp(self, mock_get_client: MagicMock) -> None:
        """When deduplicating, the earliest timestamp_start is kept."""
        from app.rag.generator import answer

        mock_client = MagicMock()
        mock_client.generate.return_value = json.dumps({"answer": "answer"})
        mock_get_client.return_value = mock_client

        # CHUNK_A timestamp_start=60.0, CHUNK_C timestamp_start=90.0
        chunks: List[ScoredChunk] = [
            _make_scored(CHUNK_C, 0.8),  # 90.0 — comes first in list
            _make_scored(CHUNK_A, 0.9),  # 60.0 — earlier timestamp
        ]
        result = answer("what is force", chunks)

        assert result["sources"][0]["timestamp_start"] == 60.0

    @patch("app.rag.generator.get_llm_client")
    def test_different_sections_not_deduplicated(self, mock_get_client: MagicMock) -> None:
        """Chunks from different sections produce separate source entries."""
        from app.rag.generator import answer

        mock_client = MagicMock()
        mock_client.generate.return_value = json.dumps({"answer": "answer"})
        mock_get_client.return_value = mock_client

        # CHUNK_A is "Newton's Second Law", CHUNK_B is "Kinematics"
        chunks: List[ScoredChunk] = [
            _make_scored(CHUNK_A, 0.8),
            _make_scored(CHUNK_B, 1.0),
        ]
        result = answer("physics question", chunks)

        assert len(result["sources"]) == 2
        titles = {s["section_title"] for s in result["sources"]}
        assert "Newton's Second Law" in titles
        assert "Kinematics" in titles


class TestGeneratorResponseShape:
    """Tests that the response matches ARCHITECTURE.md Section 6."""

    @patch("app.rag.generator.get_llm_client")
    def test_response_has_answer_and_sources(self, mock_get_client: MagicMock) -> None:
        """Response must have exactly 'answer' (str) and 'sources' (list)."""
        from app.rag.generator import answer

        mock_client = MagicMock()
        mock_client.generate.return_value = json.dumps({"answer": "F = ma"})
        mock_get_client.return_value = mock_client

        result = answer("what is force", [_make_scored(CHUNK_A, 0.8)])

        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)

    @patch("app.rag.generator.get_llm_client")
    def test_source_entry_has_required_keys(self, mock_get_client: MagicMock) -> None:
        """Each source entry must have lecture_id, section_title, timestamp_start."""
        from app.rag.generator import answer

        mock_client = MagicMock()
        mock_client.generate.return_value = json.dumps({"answer": "answer"})
        mock_get_client.return_value = mock_client

        result = answer("question", [_make_scored(CHUNK_A, 0.8)])

        for source in result["sources"]:
            assert "lecture_id" in source
            assert "section_title" in source
            assert "timestamp_start" in source

    @patch("app.rag.generator.get_llm_client")
    def test_not_covered_response_shape(self, mock_get_client: MagicMock) -> None:
        """The 'not covered' response has the same shape with empty sources."""
        from app.rag.generator import answer

        result = answer("off topic question", [])

        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) == 0

    @patch("app.rag.generator.get_llm_client")
    def test_malformed_llm_json_handled_gracefully(self, mock_get_client: MagicMock) -> None:
        """If the LLM returns non-JSON, the raw text is used as the answer."""
        from app.rag.generator import answer

        mock_client = MagicMock()
        mock_client.generate.return_value = "This is not valid JSON response"
        mock_get_client.return_value = mock_client

        result = answer("question", [_make_scored(CHUNK_A, 0.8)])

        assert result["answer"] == "This is not valid JSON response"
        assert len(result["sources"]) == 1

    @patch("app.rag.generator.get_llm_client")
    def test_system_prompt_enforces_grounding(self, mock_get_client: MagicMock) -> None:
        """The system prompt must instruct the LLM to use ONLY provided context."""
        from app.rag.generator import answer, SYSTEM_PROMPT

        mock_client = MagicMock()
        mock_client.generate.return_value = json.dumps({"answer": "answer"})
        mock_get_client.return_value = mock_client

        answer("question", [_make_scored(CHUNK_A, 0.8)])

        call_args = mock_client.generate.call_args
        system_prompt = call_args.kwargs.get("system_prompt") or call_args[1].get("system_prompt") or call_args[0][0]

        # Check that the system prompt contains grounding instructions
        assert "ONLY" in system_prompt or "only" in system_prompt.lower()
        assert "context" in system_prompt.lower()
