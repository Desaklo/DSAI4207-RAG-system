#!/usr/bin/env python3
"""Extract text from lecture PDFs and chunk into ~150-250 word passages.

Usage:
    python src/extract_and_chunk.py
    python src/extract_and_chunk.py --raw-dir data/raw --out-extracted data/extracted --out-chunks data/chunks/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_EXTRACTED = ROOT / "data" / "extracted"
DEFAULT_CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"

# Target chunk size in words (proposal: ~150-250)
MIN_WORDS = 150
MAX_WORDS = 250
TARGET_WORDS = 200


def extract_pdf(pdf_path: Path, txt_path: Path) -> str:
    """Extract text with pdftotext (poppler). Falls back to a clear error."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(txt_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            "ERROR: pdftotext not found. Install poppler-utils "
            "(e.g. apt install poppler-utils).",
            file=sys.stderr,
        )
        raise
    except subprocess.CalledProcessError as e:
        print(f"ERROR extracting {pdf_path.name}: {e.stderr}", file=sys.stderr)
        raise
    return txt_path.read_text(encoding="utf-8", errors="replace")


def clean_text(text: str) -> str:
    """Normalize whitespace while keeping paragraph breaks roughly intact."""
    # Drop form-feed page breaks
    text = text.replace("\x0c", "\n")
    # Collapse runs of spaces/tabs inside a line
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    # Join soft-wrapped empty lines
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def word_list(text: str) -> list[str]:
    return text.split()


def chunk_words(words: list[str], target: int = TARGET_WORDS, max_w: int = MAX_WORDS) -> list[str]:
    """Split a word list into chunks of roughly `target` words (cap at max_w)."""
    if not words:
        return []
    chunks: list[str] = []
    i = 0
    n = len(words)
    while i < n:
        end = min(i + target, n)
        # If leftover is tiny, absorb into previous or keep as short last chunk
        if n - end < MIN_WORDS // 2 and end < n:
            end = n
        # Prefer breaking at sentence-ish boundary near the end of the window
        window = words[i:end]
        if end < n and len(window) >= MIN_WORDS:
            # Look back up to ~40 words for a period/question/exclamation end
            lookback = min(40, len(window) - MIN_WORDS)
            for j in range(len(window) - 1, len(window) - 1 - lookback, -1):
                if window[j].endswith((".", "?", "!")):
                    window = window[: j + 1]
                    end = i + j + 1
                    break
        # Hard cap
        if len(window) > max_w:
            window = window[:max_w]
            end = i + max_w
        chunks.append(" ".join(window))
        i = end
    return chunks


def process_corpus(
    raw_dir: Path,
    extracted_dir: Path,
    chunks_path: Path,
    skip_extract: bool = False,
) -> int:
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {raw_dir}", file=sys.stderr)
        return 0

    extracted_dir.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    chunk_counter = 0

    for pdf in pdfs:
        stem = pdf.stem
        txt_path = extracted_dir / f"{stem}.txt"
        if skip_extract and txt_path.exists():
            text = txt_path.read_text(encoding="utf-8", errors="replace")
            print(f"[skip extract] {pdf.name}")
        else:
            print(f"[extract] {pdf.name}")
            text = extract_pdf(pdf, txt_path)
        text = clean_text(text)
        # Re-save cleaned text
        txt_path.write_text(text + "\n", encoding="utf-8")

        words = word_list(text)
        pieces = chunk_words(words)
        print(f"  → {len(words)} words → {len(pieces)} chunks")

        for local_i, piece in enumerate(pieces):
            wc = len(piece.split())
            # Skip tiny leftovers from odd PDFs
            if wc < 40:
                continue
            rec = {
                "chunk_id": f"{stem}__{local_i:04d}",
                "source_file": pdf.name,
                "source_stem": stem,
                "local_index": local_i,
                "word_count": wc,
                "text": piece,
            }
            all_chunks.append(rec)
            chunk_counter += 1

    with chunks_path.open("w", encoding="utf-8") as f:
        for rec in all_chunks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks → {chunks_path}")
    return len(all_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDFs and build chunks.jsonl")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-extracted", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--out-chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Reuse existing .txt under out-extracted",
    )
    args = parser.parse_args()
    n = process_corpus(args.raw_dir, args.out_extracted, args.out_chunks, args.skip_extract)
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
