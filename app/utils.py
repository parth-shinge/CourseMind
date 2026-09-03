"""Shared utility functions used across the application."""

from __future__ import annotations

import re


def sanitize_lecture_id(raw_name: str) -> str:
    """Convert a raw filename stem into a clean, filesystem/JSON-safe lecture_id.

    Rules:
      - Lowercase
      - Replace spaces and non-alphanumeric characters (except hyphens) with underscores
      - Collapse consecutive underscores
      - Strip leading/trailing underscores
      - Truncate to 80 characters

    Examples:
        "Me at the zoo_jNQXAC9IVRw" -> "me_at_the_zoo_jnqxac9ivrw"
        "Unit 3 — Lec #4 (Newton)" -> "unit_3_lec_4_newton"

    Args:
        raw_name: Unsanitized string (typically Path.stem of a video file).

    Returns:
        A clean, lowercase, filesystem-safe identifier.
    """
    cleaned = raw_name.strip().lower()
    # Replace any character that isn't alphanumeric, hyphen, or underscore
    cleaned = re.sub(r"[^a-z0-9\-_]", "_", cleaned)
    # Collapse multiple consecutive underscores/hyphens
    cleaned = re.sub(r"[_\-]{2,}", "_", cleaned)
    # Strip leading/trailing underscores
    cleaned = cleaned.strip("_")
    # Truncate
    return cleaned[:80] if cleaned else "untitled"
