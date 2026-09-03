"""Tests for transcription and video ingestion pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.pipeline.notes_store import load_transcript, save_transcript
from app.pipeline.transcribe import TranscriptionError, transcribe
from app.pipeline.video_ingest import VideoIngestError, download_video


class DummyWhisperSegment:
    """Mock segment object imitating faster-whisper's Segment namedtuple."""

    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class DummyTranscriptionInfo:
    """Mock info object imitating faster-whisper's TranscriptionInfo."""

    def __init__(self, language: str = "en", language_probability: float = 0.98) -> None:
        self.language = language
        self.language_probability = language_probability


def test_transcribe_mocked_output(tmp_path: Path) -> None:
    """Verify that transcribe() normalizes output according to Section 6 contracts and saves JSON."""
    # Create a temporary dummy video file
    dummy_video = tmp_path / "unit1-lec01.mp4"
    dummy_video.write_bytes(b"dummy video content")

    mock_segments = [
        DummyWhisperSegment(12.423, 18.911, "  Newton's second law states that force equals mass times acceleration.  "),
        DummyWhisperSegment(19.0, 25.556, "Let us write this down as F = ma."),
    ]
    mock_info = DummyTranscriptionInfo()

    mock_model_instance = MagicMock()
    mock_model_instance.transcribe.return_value = (iter(mock_segments), mock_info)

    with patch("faster_whisper.WhisperModel", return_value=mock_model_instance):
        result = transcribe(dummy_video, model_size="tiny")

    # 1. Verify schema contract matching ARCHITECTURE.md Section 6
    assert len(result) == 2
    assert result[0] == {
        "start": 12.42,
        "end": 18.91,
        "text": "Newton's second law states that force equals mass times acceleration.",
    }
    assert result[1] == {
        "start": 19.0,
        "end": 25.56,
        "text": "Let us write this down as F = ma.",
    }

    # 2. Verify transcript was persisted to disk in data/transcripts/
    expected_transcript = settings.TRANSCRIPTS_DIR / "unit1-lec01.json"
    assert expected_transcript.exists()

    # 3. Verify load_transcript returns the exact data
    loaded_data = load_transcript("unit1-lec01")
    assert loaded_data == result

    # Clean up test artifact
    if expected_transcript.exists():
        expected_transcript.unlink()


def test_transcribe_file_not_found() -> None:
    """Verify transcribe() raises FileNotFoundError on missing input."""
    with pytest.raises(FileNotFoundError):
        transcribe("data/raw_videos/non_existent_lecture_xyz.mp4")


def test_video_ingest_local_file(tmp_path: Path) -> None:
    """Verify download_video() resolves an existing local file without copying."""
    sample_file = tmp_path / "sample_lecture.mp4"
    sample_file.write_text("dummy video")

    resolved = download_video(str(sample_file))
    assert Path(resolved).resolve() == sample_file.resolve()


def test_video_ingest_missing_local_file() -> None:
    """Verify download_video() raises FileNotFoundError for non-existent local file."""
    with pytest.raises(FileNotFoundError):
        download_video("data/raw_videos/does_not_exist_12345.mp4")


def test_video_ingest_empty_input() -> None:
    """Verify download_video() raises VideoIngestError for empty input."""
    with pytest.raises(VideoIngestError):
        download_video("")
