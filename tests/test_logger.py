"""Tests for Phase 6 logging: app/utils/logger.py and app/utils/log_summary.py.

All tests are fully self-contained with temp directories — no live LLM calls,
no dependency on the real data/logs/ directory.
"""

from __future__ import annotations

import json
import os
import sys
import importlib
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read all JSON lines from a .jsonl file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _make_chunk(lecture_id: str = "lec1", section: str = "Intro", score: float = 0.9):
    """Return a minimal (Chunk, score) tuple for test use."""
    chunk = {
        "chunk_id": f"{lecture_id}-001",
        "lecture_id": lecture_id,
        "section_title": section,
        "timestamp_start": 10.0,
        "timestamp_end": 60.0,
        "text": "Sample text.",
    }
    return (chunk, score)


# ---------------------------------------------------------------------------
# Fixtures: patch LOGS_DIR to a temp directory for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_logs_dir(tmp_path, monkeypatch):
    """Redirect all logger I/O to a fresh temp dir per test."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    # Patch at the module level so the already-imported constants update too
    import app.utils.logger as logger_mod
    monkeypatch.setattr(logger_mod, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logger_mod, "QUERY_LOG", logs_dir / "query_log.jsonl")
    monkeypatch.setattr(logger_mod, "INGEST_LOG", logs_dir / "ingest_log.jsonl")

    yield logs_dir


# ---------------------------------------------------------------------------
# log_query tests
# ---------------------------------------------------------------------------

class TestLogQuery:
    def test_creates_jsonl_file_on_first_call(self, patch_logs_dir):
        from app.utils.logger import log_query

        query_log = patch_logs_dir / "query_log.jsonl"
        assert not query_log.exists()

        log_query(
            question="what is velocity?",
            retrieved_chunks=[_make_chunk()],
            filtered_out_count=2,
            answer="Velocity is speed with direction.",
            sources=[{"lecture_id": "lec1", "section_title": "Intro", "timestamp_start": 10.0}],
            latency_seconds=1.234,
            used_llm=True,
        )

        assert query_log.exists()

    def test_record_schema(self, patch_logs_dir):
        from app.utils.logger import log_query

        log_query(
            question="what is force?",
            retrieved_chunks=[_make_chunk("lec2", "Forces", 0.75)],
            filtered_out_count=1,
            answer="Force equals mass times acceleration.",
            sources=[{"lecture_id": "lec2", "section_title": "Forces", "timestamp_start": 30.0}],
            latency_seconds=2.5,
            used_llm=True,
        )

        records = _read_jsonl(patch_logs_dir / "query_log.jsonl")
        assert len(records) == 1
        r = records[0]

        # Required top-level fields
        assert "timestamp" in r
        assert r["question"] == "what is force?"
        assert r["retrieved_count"] == 1
        assert r["filtered_out_count"] == 1
        assert r["answer"] == "Force equals mass times acceleration."
        assert r["latency_seconds"] == pytest.approx(2.5, abs=0.001)
        assert r["used_llm"] is True

        # Compact chunk metadata (no full text)
        assert len(r["chunks"]) == 1
        chunk_meta = r["chunks"][0]
        assert chunk_meta["lecture_id"] == "lec2"
        assert chunk_meta["section_title"] == "Forces"
        assert "score" in chunk_meta
        assert "text" not in chunk_meta  # must NOT include full text

    def test_not_covered_record(self, patch_logs_dir):
        """When used_llm=False (no chunks), record should reflect that."""
        from app.utils.logger import log_query

        log_query(
            question="how do I bake a cake?",
            retrieved_chunks=[],
            filtered_out_count=5,
            answer="This question isn't covered in this course's material yet.",
            sources=[],
            latency_seconds=0.05,
            used_llm=False,
        )

        records = _read_jsonl(patch_logs_dir / "query_log.jsonl")
        r = records[0]
        assert r["used_llm"] is False
        assert r["retrieved_count"] == 0
        assert r["chunks"] == []

    def test_appends_multiple_records(self, patch_logs_dir):
        from app.utils.logger import log_query

        for i in range(3):
            log_query(
                question=f"question {i}",
                retrieved_chunks=[_make_chunk()],
                filtered_out_count=0,
                answer=f"answer {i}",
                sources=[],
                latency_seconds=float(i),
                used_llm=True,
            )

        records = _read_jsonl(patch_logs_dir / "query_log.jsonl")
        assert len(records) == 3
        assert records[2]["question"] == "question 2"

    def test_does_not_raise_on_io_error(self, patch_logs_dir, monkeypatch):
        """A broken write must not propagate an exception to the caller."""
        import app.utils.logger as logger_mod

        def boom(*args, **kwargs):
            raise OSError("Disk full")

        monkeypatch.setattr(logger_mod, "_append_jsonl", boom)

        # Should print a warning but NOT raise
        log_query = logger_mod.log_query
        log_query(
            question="test",
            retrieved_chunks=[],
            filtered_out_count=0,
            answer="ans",
            sources=[],
            latency_seconds=0.1,
            used_llm=False,
        )  # Must not raise


# ---------------------------------------------------------------------------
# log_ingestion tests
# ---------------------------------------------------------------------------

class TestLogIngestion:
    def test_creates_jsonl_file_on_first_call(self, patch_logs_dir):
        from app.utils.logger import log_ingestion

        ingest_log = patch_logs_dir / "ingest_log.jsonl"
        assert not ingest_log.exists()

        log_ingestion(
            lecture_id="lec1",
            source="https://youtube.com/xyz",
            stage_durations={"download": 3.2, "transcribe": 45.0, "summarize": 12.1, "index": 0.8},
            success=True,
        )

        assert ingest_log.exists()

    def test_record_schema_success(self, patch_logs_dir):
        from app.utils.logger import log_ingestion

        log_ingestion(
            lecture_id="unit3-lec4",
            source="https://youtube.com/abc",
            stage_durations={"download": 4.2, "transcribe": 38.1, "summarize": 12.4, "index": 1.3},
            success=True,
            error=None,
        )

        records = _read_jsonl(patch_logs_dir / "ingest_log.jsonl")
        assert len(records) == 1
        r = records[0]

        assert "timestamp" in r
        assert r["lecture_id"] == "unit3-lec4"
        assert r["source"] == "https://youtube.com/abc"
        assert r["success"] is True
        assert r["error"] is None

        stages = r["stage_durations_seconds"]
        assert stages["download"] == pytest.approx(4.2, abs=0.01)
        assert stages["transcribe"] == pytest.approx(38.1, abs=0.01)

        expected_total = 4.2 + 38.1 + 12.4 + 1.3
        assert r["total_duration_seconds"] == pytest.approx(expected_total, abs=0.1)

    def test_record_schema_failure(self, patch_logs_dir):
        """Partial failure: only some stages timed, error message captured."""
        from app.utils.logger import log_ingestion

        log_ingestion(
            lecture_id="lec_bad",
            source="local_file.mp4",
            stage_durations={"download": 1.5, "transcribe": 20.0},  # summarize never ran
            success=False,
            error="LLM quota exceeded",
        )

        records = _read_jsonl(patch_logs_dir / "ingest_log.jsonl")
        r = records[0]
        assert r["success"] is False
        assert r["error"] == "LLM quota exceeded"
        assert "summarize" not in r["stage_durations_seconds"]
        assert r["total_duration_seconds"] == pytest.approx(1.5 + 20.0, abs=0.01)

    def test_does_not_raise_on_io_error(self, patch_logs_dir, monkeypatch):
        import app.utils.logger as logger_mod

        def boom(*args, **kwargs):
            raise OSError("Permission denied")

        monkeypatch.setattr(logger_mod, "_append_jsonl", boom)

        log_ingestion = logger_mod.log_ingestion
        log_ingestion(
            lecture_id="lec1",
            source="url",
            stage_durations={},
            success=False,
        )  # Must not raise


# ---------------------------------------------------------------------------
# log_summary tests
# ---------------------------------------------------------------------------

class TestLogSummary:
    """Tests for summarize_query_log() and summarize_ingest_log() functions."""

    def _write_query_lines(self, path: Path, lines: List[Dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for rec in lines:
                f.write(json.dumps(rec) + "\n")

    def _write_ingest_lines(self, path: Path, lines: List[Dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for rec in lines:
                f.write(json.dumps(rec) + "\n")

    def test_query_summary_totals_and_percentages(self, tmp_path):
        from app.utils.log_summary import summarize_query_log

        records = [
            {"question": "q1", "retrieved_count": 3, "filtered_out_count": 1,
             "latency_seconds": 2.0, "used_llm": True},
            {"question": "q2", "retrieved_count": 0, "filtered_out_count": 5,
             "latency_seconds": 0.1, "used_llm": False},
            {"question": "q3", "retrieved_count": 4, "filtered_out_count": 0,
             "latency_seconds": 3.0, "used_llm": True},
        ]

        stats = summarize_query_log(records)

        assert stats["total_queries"] == 3
        assert stats["pct_not_covered"] == pytest.approx(33.3, abs=0.5)
        assert stats["avg_latency_seconds"] == pytest.approx((2.0 + 0.1 + 3.0) / 3, abs=0.01)
        assert stats["avg_chunks_retrieved"] == pytest.approx((3 + 0 + 4) / 3, abs=0.01)

    def test_query_summary_empty(self):
        from app.utils.log_summary import summarize_query_log

        stats = summarize_query_log([])
        assert stats["total_queries"] == 0
        assert stats["pct_not_covered"] == 0.0
        assert stats["avg_latency_seconds"] == 0.0

    def test_ingest_summary_averages(self):
        from app.utils.log_summary import summarize_ingest_log

        records = [
            {
                "success": True,
                "stage_durations_seconds": {"download": 4.0, "transcribe": 40.0, "summarize": 12.0, "index": 2.0},
                "total_duration_seconds": 58.0,
            },
            {
                "success": True,
                "stage_durations_seconds": {"download": 6.0, "transcribe": 60.0, "summarize": 18.0, "index": 3.0},
                "total_duration_seconds": 87.0,
            },
            {
                "success": False,
                "stage_durations_seconds": {"download": 2.0},
                "total_duration_seconds": 2.0,
            },
        ]

        stats = summarize_ingest_log(records)

        assert stats["total_ingestions"] == 3
        assert stats["pct_success"] == pytest.approx(66.7, abs=0.5)
        avg_stages = stats["avg_stage_durations_seconds"]
        assert avg_stages["download"] == pytest.approx((4.0 + 6.0 + 2.0) / 3, abs=0.01)
        assert avg_stages["transcribe"] == pytest.approx((40.0 + 60.0) / 2, abs=0.01)

    def test_ingest_summary_empty(self):
        from app.utils.log_summary import summarize_ingest_log

        stats = summarize_ingest_log([])
        assert stats["total_ingestions"] == 0
        assert stats["pct_success"] == 0.0

    def test_load_jsonl_skips_malformed_lines(self, tmp_path):
        from app.utils.log_summary import _load_jsonl

        log_path = tmp_path / "test.jsonl"
        log_path.write_text(
            '{"a": 1}\n'
            'NOT VALID JSON\n'
            '{"b": 2}\n',
            encoding="utf-8",
        )

        records = _load_jsonl(log_path)
        assert len(records) == 2
        assert records[0] == {"a": 1}
        assert records[1] == {"b": 2}

    def test_load_jsonl_missing_file_returns_empty(self, tmp_path):
        from app.utils.log_summary import _load_jsonl

        records = _load_jsonl(tmp_path / "nonexistent.jsonl")
        assert records == []
