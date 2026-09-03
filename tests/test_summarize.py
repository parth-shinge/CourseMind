"""Tests for the summarization pipeline and LLM client.

All tests mock the LLM client to avoid real API calls. The tests verify:
  - Output sections match the expected shape (section_title, timestamps, text)
  - Timestamps in output are derived from real segment data, not hallucinated
  - Markdown rendering includes proper headings and timestamp labels
  - Notes store round-trips correctly (save → load)
  - LLM client raises on missing API keys
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.llm_client import LLMClient, LLMConfigError, LLMTruncationError, get_llm_client
from app.pipeline.notes_store import load_notes, save_notes
from app.pipeline.summarize import (
    Notes,
    Segment,
    _extract_section_segments,
    _format_timestamp,
    _render_markdown,
    _strip_json_fences,
    generate_notes,
)
from app.utils import sanitize_lecture_id


# ---------------------------------------------------------------------------
# Fixtures / sample data
# ---------------------------------------------------------------------------

SAMPLE_SEGMENTS: List[Segment] = [
    {"start": 0.0, "end": 30.5, "text": "Welcome to today's lecture on Newton's Laws of Motion."},
    {"start": 31.0, "end": 60.0, "text": "The first law, also called the law of inertia, states that an object at rest stays at rest."},
    {"start": 61.0, "end": 90.0, "text": "An object in motion stays in motion unless acted upon by an external force."},
    {"start": 91.0, "end": 120.0, "text": "Now let's move on to Newton's second law."},
    {"start": 121.0, "end": 150.0, "text": "The second law states that F equals m times a, or F = ma."},
    {"start": 151.0, "end": 180.0, "text": "Force is directly proportional to mass and acceleration."},
    {"start": 181.0, "end": 210.0, "text": "Finally, the third law: for every action there is an equal and opposite reaction."},
    {"start": 211.0, "end": 240.0, "text": "This is sometimes called the action-reaction principle."},
]

SAMPLE_META = {
    "title": "Newton's Laws of Motion",
    "source": "https://youtube.com/watch?v=example",
    "date": "2026-09-02",
}

# LLM mock response for Pass 1 (topic outlining)
MOCK_OUTLINE_RESPONSE = json.dumps([
    {"section_title": "Introduction and First Law of Motion", "timestamp_start": 0.0, "timestamp_end": 90.0},
    {"section_title": "Newton's Second Law (F = ma)", "timestamp_start": 91.0, "timestamp_end": 180.0},
    {"section_title": "Newton's Third Law (Action-Reaction)", "timestamp_start": 181.0, "timestamp_end": 240.0},
])

# LLM mock responses for Pass 2 (per-section notes)
MOCK_NOTES_RESPONSES = [
    "- Newton's First Law: law of inertia\n- An object at rest stays at rest\n- An object in motion stays in motion unless acted upon by an external force",
    "- Newton's Second Law: F = ma\n- Force is directly proportional to mass and acceleration",
    "- Newton's Third Law: every action has an equal and opposite reaction\n- Also known as the action-reaction principle",
]


def _make_mock_client(call_count_holder: list | None = None) -> LLMClient:
    """Create a mock LLMClient that returns pre-canned responses.

    First call returns the outline JSON, subsequent calls return section notes.
    """
    mock = MagicMock(spec=LLMClient)
    mock.provider = "mock"

    call_index = [0]  # mutable counter

    def side_effect(system_prompt: str, user_prompt: str, max_tokens: int = 2000, **kwargs) -> str:
        idx = call_index[0]
        call_index[0] += 1
        if call_count_holder is not None:
            call_count_holder.append(idx)

        if idx == 0:
            # Pass 1: outline
            return MOCK_OUTLINE_RESPONSE
        else:
            # Pass 2: section notes (idx 1 -> section 0, idx 2 -> section 1, etc.)
            section_idx = idx - 1
            if section_idx < len(MOCK_NOTES_RESPONSES):
                return MOCK_NOTES_RESPONSES[section_idx]
            return "- (No content)"

    mock.generate.side_effect = side_effect
    return mock


# ---------------------------------------------------------------------------
# Tests: sanitize_lecture_id
# ---------------------------------------------------------------------------

class TestSanitizeLectureId:
    def test_spaces_replaced(self) -> None:
        assert sanitize_lecture_id("Me at the zoo") == "me_at_the_zoo"

    def test_special_characters(self) -> None:
        result = sanitize_lecture_id("Unit 3 — Lec #4 (Newton)")
        assert result == "unit_3_lec_4_newton"
        assert " " not in result
        assert "#" not in result

    def test_already_clean(self) -> None:
        assert sanitize_lecture_id("unit3-lec4") == "unit3-lec4"

    def test_collapses_underscores(self) -> None:
        assert sanitize_lecture_id("foo___bar") == "foo_bar"

    def test_empty_string(self) -> None:
        assert sanitize_lecture_id("") == "untitled"

    def test_truncation(self) -> None:
        long_name = "a" * 200
        result = sanitize_lecture_id(long_name)
        assert len(result) <= 80


# ---------------------------------------------------------------------------
# Tests: _format_timestamp helper
# ---------------------------------------------------------------------------

class TestFormatTimestamp:
    def test_seconds_only(self) -> None:
        assert _format_timestamp(45.0) == "00:45"

    def test_minutes_and_seconds(self) -> None:
        assert _format_timestamp(340.0) == "05:40"

    def test_hours(self) -> None:
        assert _format_timestamp(3661.0) == "01:01:01"

    def test_zero(self) -> None:
        assert _format_timestamp(0.0) == "00:00"


# ---------------------------------------------------------------------------
# Tests: _extract_section_segments
# ---------------------------------------------------------------------------

class TestExtractSectionSegments:
    def test_extracts_correct_range(self) -> None:
        result = _extract_section_segments(SAMPLE_SEGMENTS, 91.0, 180.0)
        assert len(result) == 3
        assert result[0]["start"] == 91.0
        assert result[-1]["end"] == 180.0

    def test_full_range(self) -> None:
        result = _extract_section_segments(SAMPLE_SEGMENTS, 0.0, 240.0)
        assert len(result) == len(SAMPLE_SEGMENTS)

    def test_empty_range(self) -> None:
        result = _extract_section_segments(SAMPLE_SEGMENTS, 500.0, 600.0)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: generate_notes (mocked LLM)
# ---------------------------------------------------------------------------

class TestGenerateNotes:
    def test_output_shape(self) -> None:
        """Verify that output sections have the correct keys matching Chunk contract."""
        mock_client = _make_mock_client()
        notes = generate_notes(SAMPLE_SEGMENTS, SAMPLE_META, client=mock_client)

        assert "sections" in notes
        assert "markdown" in notes
        assert isinstance(notes["sections"], list)
        assert len(notes["sections"]) == 3

        # Verify each section has the keys chunker.py will need
        required_keys = {"section_title", "timestamp_start", "timestamp_end", "text"}
        for section in notes["sections"]:
            assert required_keys.issubset(section.keys()), (
                f"Section missing keys: {required_keys - section.keys()}"
            )

    def test_timestamps_from_real_segments(self) -> None:
        """Verify timestamps are pulled from actual segment data, not invented."""
        mock_client = _make_mock_client()
        notes = generate_notes(SAMPLE_SEGMENTS, SAMPLE_META, client=mock_client)

        all_starts = {s["start"] for s in SAMPLE_SEGMENTS}
        all_ends = {s["end"] for s in SAMPLE_SEGMENTS}
        min_start = min(all_starts)
        max_end = max(all_ends)

        for section in notes["sections"]:
            # Timestamps should be within the bounds of the actual segments
            assert section["timestamp_start"] >= min_start, (
                f"timestamp_start {section['timestamp_start']} < min segment start {min_start}"
            )
            assert section["timestamp_end"] <= max_end, (
                f"timestamp_end {section['timestamp_end']} > max segment end {max_end}"
            )
            assert section["timestamp_start"] < section["timestamp_end"]

    def test_section_titles_present(self) -> None:
        mock_client = _make_mock_client()
        notes = generate_notes(SAMPLE_SEGMENTS, SAMPLE_META, client=mock_client)

        titles = [s["section_title"] for s in notes["sections"]]
        assert "Introduction and First Law of Motion" in titles
        assert "Newton's Second Law (F = ma)" in titles
        assert "Newton's Third Law (Action-Reaction)" in titles

    def test_section_text_is_bullet_notes(self) -> None:
        mock_client = _make_mock_client()
        notes = generate_notes(SAMPLE_SEGMENTS, SAMPLE_META, client=mock_client)

        for section in notes["sections"]:
            assert section["text"].startswith("- "), (
                f"Section text should start with bullet: {section['text'][:50]}"
            )

    def test_llm_called_correct_number_of_times(self) -> None:
        """Pass 1 (outline) + 3 sections = 4 LLM calls total."""
        call_log: list = []
        mock_client = _make_mock_client(call_count_holder=call_log)
        generate_notes(SAMPLE_SEGMENTS, SAMPLE_META, client=mock_client)

        assert len(call_log) == 4  # 1 outline + 3 sections

    def test_empty_segments_raises(self) -> None:
        mock_client = _make_mock_client()
        with pytest.raises(ValueError, match="empty segment list"):
            generate_notes([], SAMPLE_META, client=mock_client)


# ---------------------------------------------------------------------------
# Tests: markdown rendering
# ---------------------------------------------------------------------------

class TestMarkdownRendering:
    def test_contains_title_heading(self) -> None:
        mock_client = _make_mock_client()
        notes = generate_notes(SAMPLE_SEGMENTS, SAMPLE_META, client=mock_client)
        md = notes["markdown"]

        assert "# Lecture: Newton's Laws of Motion" in md

    def test_contains_section_headings_with_timestamps(self) -> None:
        mock_client = _make_mock_client()
        notes = generate_notes(SAMPLE_SEGMENTS, SAMPLE_META, client=mock_client)
        md = notes["markdown"]

        # Each section heading should include a timestamp range
        assert "## Introduction and First Law of Motion (00:00–01:30)" in md
        assert "## Newton's Second Law (F = ma) (01:31–03:00)" in md
        assert "## Newton's Third Law (Action-Reaction) (03:01–04:00)" in md

    def test_contains_source(self) -> None:
        mock_client = _make_mock_client()
        notes = generate_notes(SAMPLE_SEGMENTS, SAMPLE_META, client=mock_client)
        assert "https://youtube.com/watch?v=example" in notes["markdown"]


# ---------------------------------------------------------------------------
# Tests: notes_store round-trip
# ---------------------------------------------------------------------------

class TestNotesStoreRoundTrip:
    def test_save_and_load_notes(self) -> None:
        """Verify save_notes -> load_notes round-trip preserves data."""
        lecture_id = "test_roundtrip_notes"
        sections = [
            {
                "section_title": "Test Section",
                "timestamp_start": 0.0,
                "timestamp_end": 60.0,
                "text": "- Test bullet point\n- Another point",
            }
        ]
        markdown = "# Test\n\n## Test Section (00:00–01:00)\n\n- Test bullet point\n"
        meta = {"title": "Test Lecture", "source": "", "date": "2026-09-02"}

        try:
            md_path, json_path = save_notes(lecture_id, sections, markdown, meta)

            assert md_path.exists()
            assert json_path.exists()

            loaded_md, loaded_meta = load_notes(lecture_id)

            assert loaded_md == markdown
            assert loaded_meta["lecture_id"] == lecture_id
            assert loaded_meta["title"] == "Test Lecture"
            assert len(loaded_meta["sections"]) == 1
            assert loaded_meta["sections"][0]["section_title"] == "Test Section"
        finally:
            # Clean up
            for p in [
                settings.NOTES_DIR / f"{lecture_id}.md",
                settings.NOTES_DIR / f"{lecture_id}.json",
            ]:
                if p.exists():
                    p.unlink()


# ---------------------------------------------------------------------------
# Tests: LLM client configuration errors
# ---------------------------------------------------------------------------

class TestLLMClientConfig:
    def test_missing_api_key_raises(self) -> None:
        """get_llm_client should raise LLMConfigError when the key is empty."""
        from app.config import Settings
        with patch.object(Settings, "get_api_key", return_value=""):
            with pytest.raises(LLMConfigError, match="API key"):
                get_llm_client(provider="gemini")

    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(LLMConfigError, match="Unsupported"):
            LLMClient(provider="openai")


# ---------------------------------------------------------------------------
# Tests: LLM client retry logic
# ---------------------------------------------------------------------------

class TestLLMClientRetry:
    def test_retry_on_503_then_success(self) -> None:
        """Verify that a 503 transient error triggers backoff retry and succeeds."""
        mock_response = MagicMock()
        mock_response.text = "Notes generated after retry."

        mock_genai_client = MagicMock()
        # First attempt raises 503, second attempt succeeds
        mock_genai_client.models.generate_content.side_effect = [
            RuntimeError("503 UNAVAILABLE. High demand."),
            mock_response,
        ]

        with patch("app.llm_client.settings") as mock_settings, \
             patch("google.genai.Client", return_value=mock_genai_client), \
             patch("app.llm_client.time.sleep") as mock_sleep:

            mock_settings.GEMINI_API_KEY = "fake-key"
            mock_settings.GEMINI_MODEL = "gemini-3.6-flash"

            client = LLMClient(provider="gemini")
            result = client.generate(system_prompt="sys", user_prompt="user")

            assert result == "Notes generated after retry."
            assert mock_genai_client.models.generate_content.call_count == 2
            assert mock_sleep.call_count == 1
            mock_sleep.assert_called_once_with(2.0)

    def test_permanent_404_fails_immediately(self) -> None:
        """Verify that a 404 non-transient error fails immediately without retrying."""
        mock_genai_client = MagicMock()
        mock_genai_client.models.generate_content.side_effect = RuntimeError(
            "404 NOT_FOUND: model not found"
        )

        with patch("app.llm_client.settings") as mock_settings, \
             patch("google.genai.Client", return_value=mock_genai_client), \
             patch("app.llm_client.time.sleep") as mock_sleep:

            mock_settings.GEMINI_API_KEY = "fake-key"
            mock_settings.GEMINI_MODEL = "gemini-3.6-flash"

            client = LLMClient(provider="gemini")
            with pytest.raises(RuntimeError, match="404"):
                client.generate(system_prompt="sys", user_prompt="user")

            assert mock_genai_client.models.generate_content.call_count == 1
            assert mock_sleep.call_count == 0

    def test_retry_exhausted_raises_after_max_attempts(self) -> None:
        """Verify that persisting 503 errors exhaust retries and raise clear error."""
        mock_genai_client = MagicMock()
        mock_genai_client.models.generate_content.side_effect = RuntimeError(
            "503 UNAVAILABLE. High demand."
        )

        with patch("app.llm_client.settings") as mock_settings, \
             patch("google.genai.Client", return_value=mock_genai_client), \
             patch("app.llm_client.time.sleep") as mock_sleep:

            mock_settings.GEMINI_API_KEY = "fake-key"
            mock_settings.GEMINI_MODEL = "gemini-3.6-flash"

            client = LLMClient(provider="gemini")
            with pytest.raises(RuntimeError, match="failed after 3 retries"):
                client.generate(system_prompt="sys", user_prompt="user")

            # 1 initial call + 3 retries = 4 total calls
            assert mock_genai_client.models.generate_content.call_count == 4
            assert mock_sleep.call_count == 3


# ---------------------------------------------------------------------------
# Tests: LLM truncation detection
# ---------------------------------------------------------------------------

class TestLLMTruncation:
    def test_gemini_max_tokens_raises_truncation_error(self) -> None:
        """Verify LLMTruncationError is raised when Gemini finish_reason is MAX_TOKENS."""
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "MAX_TOKENS"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.text = '{"partial": "data'

        mock_genai_client = MagicMock()
        mock_genai_client.models.generate_content.return_value = mock_response

        with patch("app.llm_client.settings") as mock_settings, \
             patch("google.genai.Client", return_value=mock_genai_client):

            mock_settings.GEMINI_API_KEY = "fake-key"
            mock_settings.GEMINI_MODEL = "gemini-3.5-flash"

            client = LLMClient(provider="gemini")
            with pytest.raises(LLMTruncationError, match="truncated"):
                client.generate(system_prompt="sys", user_prompt="user", max_tokens=100)

    def test_gemini_stop_finish_reason_returns_normally(self) -> None:
        """Verify normal STOP finish_reason returns text without error."""
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.text = "Complete response"

        mock_genai_client = MagicMock()
        mock_genai_client.models.generate_content.return_value = mock_response

        with patch("app.llm_client.settings") as mock_settings, \
             patch("google.genai.Client", return_value=mock_genai_client):

            mock_settings.GEMINI_API_KEY = "fake-key"
            mock_settings.GEMINI_MODEL = "gemini-3.5-flash"

            client = LLMClient(provider="gemini")
            result = client.generate(system_prompt="sys", user_prompt="user")
            assert result == "Complete response"


# ---------------------------------------------------------------------------
# Tests: JSON fence stripping
# ---------------------------------------------------------------------------

class TestJsonFenceStripping:
    def test_strips_json_code_fence(self) -> None:
        """Verify ```json ... ``` fences are stripped correctly."""
        fenced = '```json\n[{"section_title": "Test", "timestamp_start": 0.0, "timestamp_end": 60.0}]\n```'
        result = _strip_json_fences(fenced)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["section_title"] == "Test"

    def test_strips_plain_code_fence(self) -> None:
        """Verify ``` ... ``` fences (no language tag) are stripped."""
        fenced = '```\n[{"section_title": "Test", "timestamp_start": 0.0, "timestamp_end": 60.0}]\n```'
        result = _strip_json_fences(fenced)
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_returns_raw_json_unchanged(self) -> None:
        """Verify unfenced JSON passes through unchanged."""
        raw = '[{"section_title": "Test", "timestamp_start": 0.0, "timestamp_end": 60.0}]'
        result = _strip_json_fences(raw)
        assert result == raw

    def test_fenced_json_in_generate_notes_pipeline(self) -> None:
        """Verify that markdown-fenced JSON from LLM is parsed correctly in the pipeline."""
        fenced_outline = '```json\n' + json.dumps([
            {"section_title": "Topic A", "timestamp_start": 0.0, "timestamp_end": 90.0},
            {"section_title": "Topic B", "timestamp_start": 91.0, "timestamp_end": 180.0},
        ]) + '\n```'

        mock = MagicMock(spec=LLMClient)
        mock.provider = "mock"
        call_index = [0]

        def side_effect(system_prompt: str, user_prompt: str, max_tokens: int = 2000, **kwargs) -> str:
            idx = call_index[0]
            call_index[0] += 1
            if idx == 0:
                return fenced_outline  # Pass 1: fenced JSON
            return "- Key point from transcript"

        mock.generate.side_effect = side_effect

        notes = generate_notes(
            SAMPLE_SEGMENTS[:4],  # Use first 4 segments
            {"title": "Fence Test", "source": "", "date": "2026-09-03"},
            client=mock,
        )
        assert len(notes["sections"]) == 2
        assert notes["sections"][0]["section_title"] == "Topic A"


# ---------------------------------------------------------------------------
# Tests: Pass 2 section truncation retry
# ---------------------------------------------------------------------------

class TestPass2TruncationRetry:
    def test_section_retries_on_truncation_with_doubled_budget(self) -> None:
        """Verify that a truncated section retries once with doubled max_tokens."""
        from app.pipeline.summarize import _generate_section_notes

        mock_client = MagicMock(spec=LLMClient)
        mock_client.provider = "mock"

        call_index = [0]

        def side_effect(system_prompt: str, user_prompt: str, max_tokens: int = 2000, **kwargs) -> str:
            idx = call_index[0]
            call_index[0] += 1
            if idx == 0:
                # First attempt: truncated
                raise LLMTruncationError("Truncated at 4000 tokens")
            else:
                # Retry: succeeds
                return "- Complete bullet point one\n- Complete bullet point two"

        mock_client.generate.side_effect = side_effect

        result = _generate_section_notes(
            "Test Section",
            [{"start": 0.0, "end": 30.0, "text": "Test transcript content."}],
            mock_client,
        )

        assert "Complete bullet point one" in result
        assert mock_client.generate.call_count == 2

        # Verify first call used 4000, retry used 8000
        first_call_kwargs = mock_client.generate.call_args_list[0]
        second_call_kwargs = mock_client.generate.call_args_list[1]
        assert first_call_kwargs.kwargs["max_tokens"] == 4000
        assert second_call_kwargs.kwargs["max_tokens"] == 8000

    def test_section_raises_on_double_truncation(self) -> None:
        """Verify that two consecutive truncations raise RuntimeError with section name."""
        from app.pipeline.summarize import _generate_section_notes

        mock_client = MagicMock(spec=LLMClient)
        mock_client.provider = "mock"
        mock_client.generate.side_effect = LLMTruncationError("Always truncated")

        with pytest.raises(RuntimeError, match='Section "Hard Section"'):
            _generate_section_notes(
                "Hard Section",
                [{"start": 0.0, "end": 30.0, "text": "Test content."}],
                mock_client,
            )

        # Two calls: initial + one retry
        assert mock_client.generate.call_count == 2


