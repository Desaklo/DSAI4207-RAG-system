#!/usr/bin/env python3
"""Export a CSV marking sheet from results/runs.jsonl for claim-level annotation.

Marking rule (team applies against PASSED chunks only):
  - supported: claim is entailed by at least one passed chunk
  - unsupported: claim is not entailed by the passed chunks (even if "true" in the world)
  - abstention: model refused / said it does not know

Usage:
    python src/marking_template.py
    python src/marking_template.py --runs results/runs.jsonl --out results/marking_sheet.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "results" / "runs.jsonl"
DEFAULT_OUT = ROOT / "results" / "marking_sheet.csv"

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_claims(answer: str) -> list[str]:
    """Heuristic sentence split for manual marking rows."""
    answer = (answer or "").strip()
    if not answer:
        return []
    # Treat the canned abstention as one unit
    low = answer.lower()
    if "don't know based on the provided chunks" in low or "do not know based on the provided chunks" in low:
        # If the whole answer is mostly abstention, one row
        if len(answer) < 200:
            return [answer]
    parts = SENT_SPLIT.split(answer)
    return [p.strip() for p in parts if p.strip()]


def looks_like_abstention(text: str) -> bool:
    t = text.lower()
    return (
        "don't know based on the provided chunks" in t
        or "do not know based on the provided chunks" in t
        or t.strip() in {"i don't know.", "i do not know."}
    )


def export(runs_path: Path, out_path: Path) -> int:
    if not runs_path.exists():
        print(f"No runs file at {runs_path}. Run the experiment first.", file=sys.stderr)
        return 0

    rows: list[dict] = []
    with runs_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            answer = rec.get("answer", "")
            claims = split_claims(answer)
            if not claims:
                claims = [""]
            for ci, claim in enumerate(claims):
                suggested = ""
                if looks_like_abstention(claim) or (
                    len(claims) == 1 and looks_like_abstention(answer)
                ):
                    suggested = "abstention"
                rows.append(
                    {
                        "question_id": rec.get("question_id"),
                        "k": rec.get("k"),
                        "model": rec.get("model"),
                        "claim_index": ci,
                        "claim_text": claim,
                        "label": suggested,  # team fills: supported / unsupported / abstention
                        "notes": "",
                        "retrieved_chunk_ids": "|".join(rec.get("retrieved_chunk_ids") or []),
                        "retrieval": rec.get("retrieval", ""),
                        "chunk_size": rec.get("chunk_size") or "",
                        "junk": rec.get("junk", ""),
                        "timestamp": rec.get("timestamp", ""),
                    }
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "k",
        "model",
        "claim_index",
        "claim_text",
        "label",
        "notes",
        "retrieved_chunk_ids",
        "retrieval",
        "chunk_size",
        "junk",
        "timestamp",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} claim rows → {out_path}")
    print(
        "Fill column 'label' with: supported | unsupported | abstention "
        "(mark against PASSED chunks only)."
    )
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Export CSV marking template")
    p.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    n = export(args.runs, args.out)
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
