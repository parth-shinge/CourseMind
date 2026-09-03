"""Video ingestion module.

Contract:
    download_video(url_or_path: str) -> str

Handles downloading videos/audio (via yt-dlp) or validating local video paths.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from app.config import settings


class VideoIngestError(Exception):
    """Raised when video ingestion or download fails."""


def _is_url(url_or_path: str) -> bool:
    """Check if the string is an HTTP/HTTPS URL."""
    parsed = urlparse(url_or_path.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _sanitize_title(title: str) -> str:
    """Sanitize video title for filesystem safety."""
    sanitized = re.sub(r'[^\w\-_.]', '_', title)
    return re.sub(r'_+', '_', sanitized).strip(' _')[:60]


def download_video(url_or_path: str) -> str:
    """Download video/audio from URL or validate local file path.

    Args:
        url_or_path: Web URL (e.g. YouTube) or local file path.

    Returns:
        Absolute string path to the local video/audio file.

    Raises:
        FileNotFoundError: If a local file path does not exist.
        VideoIngestError: If download fails, URL is invalid, or yt-dlp fails.
    """
    cleaned_input = (url_or_path or "").strip()
    if not cleaned_input:
        raise VideoIngestError("Video source input cannot be empty.")

    # 1. Local file path check
    if not _is_url(cleaned_input):
        local_path = Path(cleaned_input).resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Local video file not found at: {cleaned_input}")
        if not local_path.is_file():
            raise VideoIngestError(f"Provided path is not a file: {cleaned_input}")
        return str(local_path)

    # 2. Remote URL handling via yt-dlp
    try:
        import yt_dlp
    except ImportError as e:
        raise VideoIngestError(
            "yt-dlp is required for downloading URLs. Please install it with `pip install yt-dlp`."
        ) from e

    target_dir = settings.RAW_VIDEOS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Configure yt-dlp options (prefer audio-only to save time and space).
    # Use video ID only in filenames to avoid Unicode encoding issues on Windows
    # (video titles can contain characters like full-width colons ＝ that fail
    # with 'charmap' codec on non-UTF-8 Windows consoles).
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(target_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract metadata without downloading first to check for existing file
            info = ydl.extract_info(cleaned_input, download=False)
            if not info:
                raise VideoIngestError(f"Could not retrieve video info from: {cleaned_input}")

            video_id = info.get("id")
            if not video_id:
                raise VideoIngestError(f"Could not extract video ID from: {cleaned_input}")

            # Check if any file with this video ID already exists in data/raw_videos
            existing_files = list(target_dir.glob(f"*_{video_id}.*")) + list(
                target_dir.glob(f"{video_id}.*")
            )
            if existing_files:
                existing_path = existing_files[0].resolve()
                print(f"Video already present on disk: {existing_path}")
                return str(existing_path)

            # File not present, perform download
            print(f"Downloading audio/video from URL: {cleaned_input} ...")
            download_info = ydl.extract_info(cleaned_input, download=True)
            if not download_info:
                raise VideoIngestError(f"Download completed but no file info returned for: {cleaned_input}")

            # Find the downloaded file
            target_files = list(target_dir.glob(f"*_{video_id}.*")) + list(
                target_dir.glob(f"{video_id}.*")
            )
            if target_files:
                downloaded_path = target_files[0].resolve()
                print(f"Successfully downloaded to: {downloaded_path}")
                return str(downloaded_path)

            # Fallback to prepare_filename
            expected_file = Path(ydl.prepare_filename(download_info)).resolve()
            if expected_file.exists():
                return str(expected_file)

            raise VideoIngestError(
                f"Download succeeded but downloaded file could not be located in {target_dir}"
            )

    except yt_dlp.utils.YoutubeDLError as e:
        raise VideoIngestError(f"yt-dlp failed while processing '{cleaned_input}': {e}") from e
    except Exception as e:
        if isinstance(e, VideoIngestError):
            raise
        raise VideoIngestError(f"Unexpected error while ingesting '{cleaned_input}': {e}") from e


def main() -> None:
    """CLI entry point for video ingestion."""
    # Ensure Unicode output works on Windows cmd/PowerShell (cp1252 default)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Download video/audio from a URL or validate a local video path."
    )
    parser.add_argument(
        "url_or_path",
        help="URL (e.g., YouTube) or local filesystem path to the video file.",
    )
    args = parser.parse_args()

    try:
        resolved_path = download_video(args.url_or_path)
        print(f"Video ready: {resolved_path}")
    except (FileNotFoundError, VideoIngestError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
