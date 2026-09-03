"""Notes and transcript storage module.

Handles persisting transcripts and structured notes to disk according
to ARCHITECTURE.md data contracts (Section 6).

Transcript files:  data/transcripts/{lecture_id}.json  (list of Segments)
Notes files:       data/notes/{lecture_id}.md           (human-readable markdown)
                   data/notes/{lecture_id}.json         (lecture metadata + sections)

All writes use an atomic write pattern (write to temp file, then os.replace)
to prevent corrupt/partial files if the process crashes mid-write.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.config import settings
from app.utils import sanitize_lecture_id


class NotesStoreError(Exception):
    """Raised when notes or transcript storage operations fail."""


def _atomic_write_text(target_path: Path, content: str) -> None:
    """Write *content* to *target_path* atomically.

    Writes to a temporary file in the same directory, then uses
    os.replace() to move it into place.  This guarantees the target
    file is either the old version or the new version — never a
    partial write.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target_path.parent),
        suffix=".tmp",
        prefix=f".{target_path.stem}_",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, str(target_path))
    except BaseException:
        # Clean up the temp file if the rename failed
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_transcript(lecture_id: str, segments: List[Dict[str, Any]]) -> Path:
    """Save transcript segments list as JSON in data/transcripts/{lecture_id}.json.

    Uses atomic write to prevent corrupt files on crash.

    Args:
        lecture_id: Unique identifier (already sanitized by caller).
        segments: List of Segment dicts [{"start": float, "end": float, "text": str}].

    Returns:
        Path to the saved JSON transcript file.

    Raises:
        NotesStoreError: If lecture_id is empty.
    """
    if not lecture_id or not lecture_id.strip():
        raise NotesStoreError("lecture_id cannot be empty.")

    settings.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    transcript_path = settings.TRANSCRIPTS_DIR / f"{lecture_id}.json"
    content = json.dumps(segments, indent=2, ensure_ascii=False)
    _atomic_write_text(transcript_path, content)
    return transcript_path


def load_transcript(lecture_id: str) -> List[Dict[str, Any]]:
    """Load transcript segments from data/transcripts/{lecture_id}.json.

    Tries the sanitized lecture_id first, then falls back to the raw name
    for backwards compatibility with transcripts saved before sanitization
    was introduced.

    Args:
        lecture_id: Unique identifier for the lecture.

    Returns:
        List of Segment dicts.

    Raises:
        FileNotFoundError: If the transcript file does not exist.
    """
    sanitized = sanitize_lecture_id(lecture_id)
    transcript_path = settings.TRANSCRIPTS_DIR / f"{sanitized}.json"

    # Fallback: try the raw name if sanitized version doesn't exist
    if not transcript_path.exists():
        raw_path = settings.TRANSCRIPTS_DIR / f"{lecture_id}.json"
        if raw_path.exists():
            transcript_path = raw_path
        else:
            raise FileNotFoundError(
                f"Transcript not found. Tried:\n"
                f"  {transcript_path}\n"
                f"  {raw_path}"
            )

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_notes(
    lecture_id: str,
    notes_sections: List[Dict[str, Any]],
    notes_markdown: str,
    meta: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Save structured notes as markdown and JSON metadata files.

    Uses atomic writes to prevent corrupt/partial files on crash.

    Files created:
      - data/notes/{lecture_id}.md   (human-readable markdown)
      - data/notes/{lecture_id}.json (lecture metadata matching Section 6,
        extended with the sections list for Phase 3 chunking)

    Args:
        lecture_id: Sanitized unique identifier for the lecture.
        notes_sections: List of section dicts:
            [{"section_title": str, "timestamp_start": float,
              "timestamp_end": float, "text": str}]
        notes_markdown: Rendered markdown string for human reading.
        meta: Lecture metadata dict. Must include at least "title" and "source".

    Returns:
        Tuple of (notes_md_path, notes_json_path).

    Raises:
        NotesStoreError: If inputs are invalid (empty lecture_id, empty sections,
            or empty markdown).
    """
    if not lecture_id or not lecture_id.strip():
        raise NotesStoreError("lecture_id cannot be empty.")
    if not notes_sections:
        raise NotesStoreError(
            "notes_sections cannot be empty — nothing to save."
        )
    if not notes_markdown or not notes_markdown.strip():
        raise NotesStoreError(
            "notes_markdown cannot be empty — no rendered content to save."
        )

    settings.NOTES_DIR.mkdir(parents=True, exist_ok=True)

    md_path = settings.NOTES_DIR / f"{lecture_id}.md"
    json_path = settings.NOTES_DIR / f"{lecture_id}.json"

    # Build the metadata JSON matching ARCHITECTURE.md Section 6
    # plus the sections list that chunker.py will consume in Phase 3.
    metadata: Dict[str, Any] = {
        "lecture_id": lecture_id,
        "title": meta.get("title", "Untitled Lecture"),
        "source": meta.get("source", ""),
        "date": meta.get("date", ""),
        "transcript_path": f"data/transcripts/{lecture_id}.json",
        "notes_path": f"data/notes/{lecture_id}.md",
        "sections": notes_sections,
    }

    # Atomic write: markdown first, then JSON (JSON is the critical file;
    # if only the .md write completes before a crash, the notes can be
    # regenerated from the transcript).
    _atomic_write_text(md_path, notes_markdown)
    json_content = json.dumps(metadata, indent=2, ensure_ascii=False)
    _atomic_write_text(json_path, json_content)

    return md_path, json_path


def load_notes(lecture_id: str) -> Tuple[str, Dict[str, Any]]:
    """Load notes markdown and metadata for a lecture.

    Args:
        lecture_id: Unique identifier for the lecture.

    Returns:
        Tuple of (markdown_content, metadata_dict).

    Raises:
        FileNotFoundError: If the notes files do not exist.
    """
    sanitized = sanitize_lecture_id(lecture_id)
    md_path = settings.NOTES_DIR / f"{sanitized}.md"
    json_path = settings.NOTES_DIR / f"{sanitized}.json"

    # Fallback to raw name for backwards compatibility
    if not md_path.exists():
        raw_md = settings.NOTES_DIR / f"{lecture_id}.md"
        if raw_md.exists():
            md_path = raw_md
        else:
            raise FileNotFoundError(f"Notes markdown not found at {md_path}")

    if not json_path.exists():
        raw_json = settings.NOTES_DIR / f"{lecture_id}.json"
        if raw_json.exists():
            json_path = raw_json
        else:
            raise FileNotFoundError(f"Notes metadata not found at {json_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return markdown_content, metadata
