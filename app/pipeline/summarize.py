"""Summarization module using LLM.

Contract:
    generate_notes(segments, meta) -> Notes

Transforms raw transcript segments into structured notes with:
  - Topical sections derived from transcript content
  - Bullet-point key points per section
  - Preserved formulae/equations and key terms
  - Timestamp references from the original segments

Topic-segmentation heuristic (documented per user request):
    We use a two-pass approach:
    1. PASS 1 — LLM-based topic outlining: Feed the entire transcript (or
       large window) to the LLM and ask it to identify distinct topics, assign
       each a short title, and specify the start/end timestamps of each topic
       boundary. This is more robust than heuristics like "gap > N seconds"
       because lectures often have no natural pauses between topic shifts.
    2. PASS 2 — Per-section note generation: For each identified section,
       extract the relevant segments and ask the LLM to produce structured
       bullet notes strictly from those segments, preserving all formulae and
       key terms verbatim.

    Why this over pure timestamp-gap heuristics:
    - Timestamp gaps in Whisper output are often artifacts of silence detection,
      not actual topic boundaries.
    - LLM-based segmentation can detect semantic shifts ("Now let's move on to
      thermodynamics...") that no gap heuristic would catch.
    - The two-pass design keeps each LLM call focused and within token limits.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from typing import Any, Dict, List

from app.llm_client import LLMClient, LLMTruncationError, get_llm_client
from app.pipeline.notes_store import load_transcript, save_notes
from app.utils import sanitize_lecture_id

# Type aliases
Segment = Dict[str, Any]
LectureMeta = Dict[str, Any]
Notes = Dict[str, Any]


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS for display."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _segments_to_transcript_text(segments: List[Segment]) -> str:
    """Render segments as a timestamped transcript for the LLM."""
    lines = []
    for seg in segments:
        ts = _format_timestamp(seg["start"])
        lines.append(f"[{ts}] {seg['text']}")
    return "\n".join(lines)


# ---- PASS 1: Topic outlining ------------------------------------------------

_OUTLINE_SYSTEM_PROMPT = """\
You are an expert academic note-taker. You will receive a timestamped lecture \
transcript. Your task is to identify the distinct topics or sections discussed \
in the lecture.

For each topic, provide:
1. A short, descriptive section_title (5-10 words max)
2. The timestamp_start (in seconds, as a float) — the start time of the FIRST \
   segment that belongs to this topic
3. The timestamp_end (in seconds, as a float) — the end time of the LAST \
   segment that belongs to this topic

Rules:
- Stay STRICTLY faithful to the transcript content. Do not invent topics \
  that are not discussed.
- Timestamps must correspond exactly to segment boundaries present in the \
  transcript — never invent timestamps.
- If the entire transcript covers only one topic, return a single section.
- Return your answer as a JSON array and nothing else. No markdown fences, \
  no explanation text.

Example output format:
[
  {"section_title": "Introduction to Thermodynamics", "timestamp_start": 0.0, "timestamp_end": 180.5},
  {"section_title": "First Law of Thermodynamics", "timestamp_start": 181.0, "timestamp_end": 420.0}
]"""


import re


def _strip_json_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```) from LLM output.

    LLMs occasionally wrap JSON responses in markdown code blocks even when
    instructed not to. This helper removes those fences so json.loads() works.
    """
    stripped = text.strip()
    # Match ```json\n...\n``` or ```\n...\n``` (with optional language tag)
    fence_pattern = re.compile(
        r"^```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL
    )
    match = fence_pattern.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _compute_outline_max_tokens(segment_count: int) -> int:
    """Compute a reasonable max_tokens budget for topic outline generation.

    Each section in the JSON outline is roughly 30-50 tokens. A typical lecture
    produces ~1 section per 10-20 segments. We budget generously to avoid
    truncation even for long lectures with many topic shifts:
      - Base of 2000 tokens (covers short lectures easily)
      - Plus ~25 tokens per segment for proportional scaling
      - Ceiling of 16000 tokens (very long lectures)
    """
    budget = 2000 + segment_count * 25
    return min(budget, 16000)


def _identify_sections(
    segments: List[Segment], client: LLMClient
) -> List[Dict[str, Any]]:
    """PASS 1: Ask the LLM to identify topic boundaries in the transcript.

    Includes resilience against:
      - Truncated responses (LLMTruncationError): re-raises with context.
      - Markdown code fences wrapping the JSON.
      - Malformed JSON on first attempt: retries once with an explicit
        "respond with ONLY valid JSON" instruction.

    Uses response_mime_type="application/json" when supported (Gemini) to
    enable constrained decoding, which guarantees structurally valid JSON.
    """
    transcript_text = _segments_to_transcript_text(segments)
    max_tokens = _compute_outline_max_tokens(len(segments))

    base_user_prompt = (
        f"Here is the full timestamped transcript:\n\n{transcript_text}\n\n"
        f"Identify the distinct topics/sections. Return ONLY a JSON array."
    )

    # Attempt 1: use JSON mode (response_mime_type) for constrained output
    try:
        raw_response = client.generate(
            system_prompt=_OUTLINE_SYSTEM_PROMPT,
            user_prompt=base_user_prompt,
            max_tokens=max_tokens,
            response_mime_type="application/json",
        )
    except LLMTruncationError:
        raise RuntimeError(
            f"Topic outline response was truncated at max_tokens={max_tokens} "
            f"for {len(segments)} segments. The model ran out of output space. "
            f"Consider reducing transcript length or increasing the token budget."
        )

    # Try to parse (stripping markdown fences first, just in case)
    cleaned = _strip_json_fences(raw_response)
    try:
        sections = json.loads(cleaned)
    except json.JSONDecodeError as first_err:
        # Attempt 2: retry with explicit "only JSON" instruction
        print(
            f"[summarize]   JSON parse failed on first attempt ({first_err}), "
            f"retrying with stricter prompt..."
        )
        retry_user_prompt = (
            f"{base_user_prompt}\n\n"
            f"IMPORTANT: Respond with ONLY valid JSON. No markdown formatting, "
            f"no code fences, no explanation text. Just the raw JSON array."
        )
        try:
            raw_response = client.generate(
                system_prompt=_OUTLINE_SYSTEM_PROMPT,
                user_prompt=retry_user_prompt,
                max_tokens=max_tokens,
                response_mime_type="application/json",
            )
        except LLMTruncationError:
            raise RuntimeError(
                f"Topic outline response was truncated on retry at "
                f"max_tokens={max_tokens} for {len(segments)} segments."
            )

        cleaned = _strip_json_fences(raw_response)
        try:
            sections = json.loads(cleaned)
        except json.JSONDecodeError as second_err:
            raise RuntimeError(
                f"LLM returned invalid JSON for topic outline on both attempts.\n"
                f"Last response was:\n{raw_response}\n"
                f"Parse error: {second_err}"
            )

    if not isinstance(sections, list) or len(sections) == 0:
        raise RuntimeError(
            f"LLM returned empty or non-list topic outline: {raw_response}"
        )

    # Validate and clamp timestamps to actual segment boundaries
    all_starts = [s["start"] for s in segments]
    all_ends = [s["end"] for s in segments]
    min_start = min(all_starts) if all_starts else 0.0
    max_end = max(all_ends) if all_ends else 0.0

    validated = []
    for sec in sections:
        validated.append({
            "section_title": str(sec.get("section_title", "Untitled Section")),
            "timestamp_start": max(min_start, float(sec.get("timestamp_start", min_start))),
            "timestamp_end": min(max_end, float(sec.get("timestamp_end", max_end))),
        })

    return validated


# ---- PASS 2: Per-section note generation ------------------------------------

_NOTES_SYSTEM_PROMPT = """\
You are an expert academic note-taker. You will receive a section of a lecture \
transcript. Your task is to produce concise, structured bullet-point notes for \
this section.

Rules:
- Be STRICTLY faithful to the transcript. Do not add facts, context, or \
  explanations that are not present in the transcript.
- Preserve all key terms, definitions, and formulae EXACTLY as spoken — \
  do not paraphrase equations or technical terms.
- Use concise bullet points. Group related points logically.
- If the speaker mentions an example, include it briefly.
- Do not add introductory or concluding phrases like "In this section..." — \
  just the notes content.
- Return ONLY the bullet-point notes as plain text (using "- " for bullets). \
  No markdown headings, no JSON, no extra commentary."""


def _extract_section_segments(
    segments: List[Segment],
    timestamp_start: float,
    timestamp_end: float,
) -> List[Segment]:
    """Extract segments that fall within the given timestamp range."""
    result = []
    for seg in segments:
        # Include segment if it overlaps with the section range
        if seg["end"] > timestamp_start and seg["start"] < timestamp_end:
            result.append(seg)
    return result


def _generate_section_notes(
    section_title: str,
    section_segments: List[Segment],
    client: LLMClient,
) -> str:
    """PASS 2: Generate bullet-point notes for a single section.

    Uses a generous max_tokens budget (4000) to accommodate thinking-enabled
    models where thinking tokens consume part of the output budget. If the
    response is truncated (LLMTruncationError), retries once with a doubled
    budget before raising a clear error identifying the failed section.
    """
    if not section_segments:
        return "- (No transcript content available for this section)"

    transcript_text = _segments_to_transcript_text(section_segments)

    user_prompt = (
        f"Section topic: \"{section_title}\"\n\n"
        f"Transcript for this section:\n{transcript_text}\n\n"
        f"Produce concise, COMPLETE bullet-point notes for this section. "
        f"Do not stop mid-sentence. Finish every bullet point fully."
    )

    # Initial budget: 4000 tokens (generous for most sections)
    initial_max_tokens = 4000

    try:
        notes_text = client.generate(
            system_prompt=_NOTES_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=initial_max_tokens,
        )
    except LLMTruncationError:
        # Retry once with doubled budget
        retry_max_tokens = initial_max_tokens * 2
        print(
            f"[summarize]     Section \"{section_title}\" truncated at "
            f"{initial_max_tokens} tokens, retrying with {retry_max_tokens}..."
        )
        try:
            notes_text = client.generate(
                system_prompt=_NOTES_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=retry_max_tokens,
            )
        except LLMTruncationError as e:
            raise RuntimeError(
                f"Section \"{section_title}\" was truncated even at "
                f"{retry_max_tokens} max_tokens. Cannot generate complete notes "
                f"for this section. Error: {e}"
            ) from e

    return notes_text.strip()


# ---- Main orchestration -----------------------------------------------------

def generate_notes(
    segments: List[Segment],
    meta: LectureMeta,
    client: LLMClient | None = None,
) -> Notes:
    """Generate structured notes from timestamped transcript segments.

    Two-pass LLM approach:
      Pass 1: Identify topic sections and their timestamp boundaries.
      Pass 2: For each section, generate bullet-point notes.

    Args:
        segments: List of Segment dicts [{"start", "end", "text"}].
        meta: Lecture metadata dict (must include "title"; "source" optional).
        client: Optional pre-configured LLMClient. If None, one is created
                from settings.

    Returns:
        Notes dict with keys:
          - "sections": list of section dicts matching the shape the chunker
            expects: [{section_title, timestamp_start, timestamp_end, text}]
          - "markdown": rendered markdown string for human reading
          - "meta": the lecture metadata
    """
    if not segments:
        raise ValueError("Cannot generate notes from an empty segment list.")

    if client is None:
        client = get_llm_client()

    title = meta.get("title", "Untitled Lecture")
    print(f"[summarize] Generating notes for '{title}' ({len(segments)} segments)...")

    # ---- Pass 1: Identify sections ----
    print("[summarize] Pass 1: Identifying topic sections via LLM...")
    raw_sections = _identify_sections(segments, client)
    print(f"[summarize] Identified {len(raw_sections)} sections.")

    # ---- Pass 2: Generate notes per section ----
    print("[summarize] Pass 2: Generating notes for each section...")
    notes_sections: List[Dict[str, Any]] = []

    for i, sec in enumerate(raw_sections, 1):
        section_title = sec["section_title"]
        ts_start = sec["timestamp_start"]
        ts_end = sec["timestamp_end"]

        print(
            f"[summarize]   Section {i}/{len(raw_sections)}: "
            f"\"{section_title}\" ({_format_timestamp(ts_start)}-{_format_timestamp(ts_end)})"
        )

        section_segs = _extract_section_segments(segments, ts_start, ts_end)
        notes_text = _generate_section_notes(section_title, section_segs, client)

        notes_sections.append({
            "section_title": section_title,
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
            "text": notes_text,
        })

    # ---- Render markdown ----
    markdown = _render_markdown(title, meta, notes_sections)

    print(f"[summarize] Done. Generated {len(notes_sections)} sections of notes.")

    return {
        "sections": notes_sections,
        "markdown": markdown,
        "meta": meta,
    }


def _render_markdown(
    title: str,
    meta: LectureMeta,
    sections: List[Dict[str, Any]],
) -> str:
    """Render the notes as human-readable markdown."""
    lines = []
    lines.append(f"# Lecture: {title}")
    lines.append("")

    if meta.get("source"):
        lines.append(f"**Source:** {meta['source']}  ")
    if meta.get("date"):
        lines.append(f"**Date:** {meta['date']}  ")
    if meta.get("source") or meta.get("date"):
        lines.append("")
    lines.append("---")
    lines.append("")

    for sec in sections:
        ts_label = (
            f"{_format_timestamp(sec['timestamp_start'])}"
            f"–{_format_timestamp(sec['timestamp_end'])}"
        )
        lines.append(f"## {sec['section_title']} ({ts_label})")
        lines.append("")
        lines.append(sec["text"])
        lines.append("")

    return "\n".join(lines)


# ---- CLI entry point --------------------------------------------------------

def main() -> None:
    """CLI entry point: generate notes from a saved transcript."""
    # Ensure Unicode output works on Windows cmd/PowerShell (cp1252 default)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=(
            "Generate structured notes from a transcript. "
            "Reads data/transcripts/{lecture_id}.json, produces notes "
            "in data/notes/{lecture_id}.md and data/notes/{lecture_id}.json."
        )
    )
    parser.add_argument(
        "lecture_id",
        help="Lecture identifier (filename stem of the transcript JSON).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional lecture title. Defaults to the lecture_id.",
    )
    parser.add_argument(
        "--source",
        default="",
        help="Optional source URL or path.",
    )
    args = parser.parse_args()

    lecture_id = sanitize_lecture_id(args.lecture_id)

    try:
        segments = load_transcript(args.lecture_id)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    meta: LectureMeta = {
        "title": args.title or args.lecture_id,
        "source": args.source,
        "date": str(date.today()),
    }

    try:
        notes = generate_notes(segments, meta)
    except Exception as e:
        print(f"Error during summarization: {e}", file=sys.stderr)
        sys.exit(1)

    # Save to disk
    md_path, json_path = save_notes(
        lecture_id,
        notes["sections"],
        notes["markdown"],
        meta,
    )
    print(f"\n[summarize] Saved notes to:")
    print(f"  Markdown: {md_path}")
    print(f"  Metadata: {json_path}")

    # Print markdown to console
    print("\n" + "=" * 60)
    print(notes["markdown"])
    print("=" * 60)


if __name__ == "__main__":
    main()
