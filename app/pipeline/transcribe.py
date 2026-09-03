"""Transcription module using faster-whisper.

Contract:
    transcribe(video_path: str) -> list[Segment]
    Segment data format: {"start": float, "end": float, "text": str}
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings
from app.pipeline.notes_store import save_transcript
from app.utils import sanitize_lecture_id

# Type alias for Segment data contract (Section 6)
Segment = Dict[str, Any]


class TranscriptionError(Exception):
    """Raised when audio/video transcription fails."""


def transcribe(
    video_path: str | Path,
    model_size: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> List[Segment]:
    """Transcribe speech from a video or audio file into timestamped segments.

    Args:
        video_path: Path to the local video or audio file.
        model_size: Whisper model size (e.g. "tiny", "base", "small").
                    Defaults to settings.WHISPER_MODEL_SIZE.
        device: Device to run on ("cpu", "cuda", "auto").
                Defaults to settings.WHISPER_DEVICE (default "cpu").
        compute_type: Computation type (e.g. "int8", "default", "float16").

    Returns:
        List of Segment dicts: [{"start": 12.4, "end": 18.9, "text": "..."}]

    Raises:
        FileNotFoundError: If the video file does not exist.
        TranscriptionError: If faster-whisper fails to load or process audio.
    """
    input_file = Path(video_path).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Video/audio file not found at: {video_path}")
    if not input_file.is_file():
        raise TranscriptionError(f"Specified path is not a file: {video_path}")

    selected_model_size = (model_size or settings.WHISPER_MODEL_SIZE or "small").strip()
    selected_device = (device or settings.WHISPER_DEVICE or "cpu").strip().lower()
    selected_compute = compute_type or ("int8" if selected_device == "cpu" else "default")

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise TranscriptionError(
            "faster-whisper is required for transcription. Please install it with `pip install faster-whisper`."
        ) from e

    print(
        f"[transcribe] Loading faster-whisper model '{selected_model_size}' (device: {selected_device}, compute_type: {selected_compute})..."
    )
    start_load_time = time.time()
    try:
        model = WhisperModel(selected_model_size, device=selected_device, compute_type=selected_compute)
    except Exception as e:
        if selected_device != "cpu":
            print(f"[transcribe] Device '{selected_device}' failed ({e}). Falling back to CPU with int8...")
            selected_device = "cpu"
            selected_compute = "int8"
            model = WhisperModel(selected_model_size, device=selected_device, compute_type=selected_compute)
        else:
            raise TranscriptionError(
                f"Failed to load WhisperModel('{selected_model_size}'): {e}"
            ) from e

    print(
        f"[transcribe] Model loaded in {time.time() - start_load_time:.2f}s. "
        f"Transcribing '{input_file.name}'... This may take a few minutes for long videos."
    )

    start_transcribe_time = time.time()
    try:
        segments_gen, info = model.transcribe(str(input_file), beam_size=5)
    except Exception as e:
        if "cublas" in str(e).lower() or "cuda" in str(e).lower():
            print(f"[transcribe] CUDA runtime error ({e}). Retrying on CPU...")
            selected_device = "cpu"
            selected_compute = "int8"
            model = WhisperModel(selected_model_size, device=selected_device, compute_type=selected_compute)
            segments_gen, info = model.transcribe(str(input_file), beam_size=5)
        else:
            raise TranscriptionError(
                f"Transcription failed for '{input_file}': {e}"
            ) from e

    try:
        print(
            f"[transcribe] Detected language: '{getattr(info, 'language', 'unknown')}' "
            f"with probability {getattr(info, 'language_probability', 0.0):.2f}"
        )

        segments: List[Segment] = []
        for segment in segments_gen:
            clean_text = segment.text.strip()
            if not clean_text:
                continue
            seg_dict: Segment = {
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": clean_text,
            }
            segments.append(seg_dict)

    except Exception as e:
        if "cublas" in str(e).lower() or "cuda" in str(e).lower():
            print(f"[transcribe] CUDA execution failed during generation ({e}). Retrying on CPU...")
            model = WhisperModel(selected_model_size, device="cpu", compute_type="int8")
            segments_gen, info = model.transcribe(str(input_file), beam_size=5)
            segments = []
            for segment in segments_gen:
                clean_text = segment.text.strip()
                if not clean_text:
                    continue
                segments.append({
                    "start": round(float(segment.start), 2),
                    "end": round(float(segment.end), 2),
                    "text": clean_text,
                })
        else:
            raise TranscriptionError(
                f"Transcription failed for '{input_file}': {e}"
            ) from e

    duration = time.time() - start_transcribe_time
    print(
        f"[transcribe] Completed transcription: {len(segments)} segments in {duration:.2f}s."
    )

    # Derive lecture_id from filename without extension, sanitized for filesystem safety
    lecture_id = sanitize_lecture_id(input_file.stem)
    saved_path = save_transcript(lecture_id, segments)
    print(f"[transcribe] Saved transcript to: {saved_path}")

    return segments


def main() -> None:
    """CLI entry point for video transcription."""
    # Ensure Unicode output works on Windows cmd/PowerShell (cp1252 default)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Transcribe a video or audio file using faster-whisper."
    )
    parser.add_argument(
        "video_path",
        help="Path to the local video or audio file.",
    )
    parser.add_argument(
        "--model-size",
        default=None,
        help="Whisper model size (e.g. tiny, base, small, medium). Default from config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (e.g. cpu, cuda, auto). Default from config.",
    )
    args = parser.parse_args()

    try:
        segments = transcribe(args.video_path, model_size=args.model_size, device=args.device)
        print(f"\n--- Transcription Summary ---")
        print(f"Total segments: {len(segments)}")
        if segments:
            print(f"First segment: {segments[0]}")
            print(f"Last segment:  {segments[-1]}")
    except (FileNotFoundError, TranscriptionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
