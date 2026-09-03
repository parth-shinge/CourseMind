"""Log summary script for pilot-study data analysis.

Reads data/logs/query_log.jsonl and data/logs/ingest_log.jsonl and prints
basic aggregate statistics useful for the evaluation section of the research
paper.

Usage:
    python -m app.utils.log_summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load all valid JSON lines from a .jsonl file. Skip malformed lines."""
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [log_summary] Skipping malformed line {lineno} in {path.name}: {exc}", file=sys.stderr)
    return records


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Public API (also callable from tests)
# ---------------------------------------------------------------------------

def summarize_query_log(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate stats from query log records."""
    total = len(records)
    if total == 0:
        return {
            "total_queries": 0,
            "pct_not_covered": 0.0,
            "avg_latency_seconds": 0.0,
            "avg_chunks_retrieved": 0.0,
        }

    not_covered = sum(1 for r in records if not r.get("used_llm", True))
    latencies = [r["latency_seconds"] for r in records if "latency_seconds" in r]
    chunk_counts = [r["retrieved_count"] for r in records if "retrieved_count" in r]

    return {
        "total_queries": total,
        "pct_not_covered": round(100.0 * not_covered / total, 1),
        "avg_latency_seconds": round(_mean(latencies), 3),
        "avg_chunks_retrieved": round(_mean(chunk_counts), 2),
    }


def summarize_ingest_log(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate stats from ingestion log records."""
    total = len(records)
    if total == 0:
        return {
            "total_ingestions": 0,
            "pct_success": 0.0,
            "avg_stage_durations_seconds": {},
            "avg_total_duration_seconds": 0.0,
        }

    successes = sum(1 for r in records if r.get("success", False))

    # Collect per-stage durations
    stage_buckets: Dict[str, List[float]] = {}
    total_durations: List[float] = []
    for r in records:
        for stage, dur in r.get("stage_durations_seconds", {}).items():
            stage_buckets.setdefault(stage, []).append(dur)
        if "total_duration_seconds" in r:
            total_durations.append(r["total_duration_seconds"])

    avg_stages = {k: round(_mean(v), 3) for k, v in stage_buckets.items()}

    return {
        "total_ingestions": total,
        "pct_success": round(100.0 * successes / total, 1),
        "avg_stage_durations_seconds": avg_stages,
        "avg_total_duration_seconds": round(_mean(total_durations), 3),
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    from app.config import settings

    # Ensure Unicode output works on Windows cmd/PowerShell (cp1252 default)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("CourseMind — Pilot Study Log Summary")
    print("=" * 60)

    # ---- Query log ----
    query_records = _load_jsonl(settings.LOGS_DIR / "query_log.jsonl")
    q_stats = summarize_query_log(query_records)

    print("\n[QUERIES]  data/logs/query_log.jsonl")
    print(f"  Total queries         : {q_stats['total_queries']}")
    print(f"  % 'not covered'       : {q_stats['pct_not_covered']}%")
    print(f"  Avg latency           : {q_stats['avg_latency_seconds']} s")
    print(f"  Avg chunks retrieved  : {q_stats['avg_chunks_retrieved']}")

    # ---- Ingest log ----
    ingest_records = _load_jsonl(settings.LOGS_DIR / "ingest_log.jsonl")
    i_stats = summarize_ingest_log(ingest_records)

    print("\n[INGEST]  data/logs/ingest_log.jsonl")
    print(f"  Total ingestions      : {i_stats['total_ingestions']}")
    print(f"  % succeeded           : {i_stats['pct_success']}%")
    print(f"  Avg total duration    : {i_stats['avg_total_duration_seconds']} s")
    if i_stats["avg_stage_durations_seconds"]:
        print("  Avg stage durations:")
        for stage, dur in i_stats["avg_stage_durations_seconds"].items():
            print(f"    {stage:<15} : {dur} s")

    if q_stats["total_queries"] == 0 and i_stats["total_ingestions"] == 0:
        print("\n  (no log data yet — run an ingestion and some queries first)")

    print()


if __name__ == "__main__":
    main()
