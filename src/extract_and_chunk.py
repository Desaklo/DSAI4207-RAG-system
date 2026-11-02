#!/usr/bin/env python3
"""Extract text from lecture PDFs and chunk into word-bounded passages.

Usage:
    python src/extract_and_chunk.py
    python src/extract_and_chunk.py --raw-dir data/raw --out-extracted data/extracted --out-chunks data/chunks/chunks.jsonl

Rebuild the two demo corpora from extracted text (no PDFs required):

    python src/extract_and_chunk.py --from-extracted --preset small --out-chunks data/chunks/small/chunks.jsonl
    python src/extract_and_chunk.py --from-extracted --preset large --out-chunks data/chunks/large/chunks.jsonl
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

# Default chunk size in words (original pipeline: ~150-250).
MIN_WORDS = 150
MAX_WORDS = 250
TARGET_WORDS = 200

# Demo corpora. Targets are the usual band; a very short source file can be
# one chunk slightly outside the band so the text is not dropped.
PRESETS: dict[str, dict[str, int]] = {
    "default": {"min_words": 150, "target": 200, "max_words": 250},
    "small": {"min_words": 100, "target": 125, "max_words": 150},
    "large": {"min_words": 250, "target": 325, "max_words": 400},
}


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


def chunk_words(
    words: list[str],
    target: int = TARGET_WORDS,
    max_w: int = MAX_WORDS,
    min_w: int = MIN_WORDS,
) -> list[str]:
    """Split a word list into chunks of roughly `target` words, staying near max_w.

    A whole source shorter than about 1.15×max_w is kept as one chunk.
    A tail shorter than min_w is merged into the previous chunk.
    """
    if not words:
        return []
    stretch = int(max_w * 1.15)
    if len(words) <= stretch:
        return [" ".join(words)]

    pieces: list[list[str]] = []
    i = 0
    n = len(words)
    while i < n:
        remaining = n - i
        if remaining <= stretch:
            pieces.append(list(words[i:]))
            break
        take = min(target, max_w)
        if remaining - take < min_w:
            take = remaining - min_w
        take = max(min_w, min(take, max_w))
        window = list(words[i : i + take])
        lookback = min(40, len(window) - min_w)
        if lookback > 0:
            for j in range(len(window) - 1, len(window) - 1 - lookback, -1):
                if not window[j].endswith((".", "?", "!")):
                    continue
                new_len = j + 1
                leftover = remaining - new_len
                if new_len >= min_w and leftover >= min_w:
                    window = window[:new_len]
                    break
        if len(window) > max_w:
            window = window[:max_w]
        if not window:
            break
        pieces.append(window)
        i += len(window)

    merged: list[list[str]] = []
    for window in pieces:
        if merged and len(window) < min_w:
            merged[-1].extend(window)
        else:
            merged.append(window)
    return [" ".join(window) for window in merged if window]


def records_from_text(stem: str, source_file: str, text: str, preset: dict[str, int]) -> list[dict]:
    words = word_list(text)
    pieces = chunk_words(
        words,
        target=preset["target"],
        max_w=preset["max_words"],
        min_w=preset["min_words"],
    )
    print(f"  → {len(words)} words → {len(pieces)} chunks")
    recs = []
    for local_i, piece in enumerate(pieces):
        wc = len(piece.split())
        if wc < 40:
            continue
        recs.append(
            {
                "chunk_id": f"{stem}__{local_i:04d}",
                "source_file": source_file,
                "source_stem": stem,
                "local_index": local_i,
                "word_count": wc,
                "text": piece,
            }
        )
    return recs


def write_chunks(chunks_path: Path, records: list[dict]) -> None:
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} chunks → {chunks_path}")


def process_extracted(extracted_dir: Path, chunks_path: Path, preset_name: str) -> int:
    """Chunk already-extracted .txt files. Does not need the raw PDFs."""
    preset = PRESETS[preset_name]
    txts = sorted(extracted_dir.glob("*.txt"))
    if not txts:
        print(f"No .txt files in {extracted_dir}", file=sys.stderr)
        return 0
    print(
        f"Chunk preset {preset_name}: "
        f"~{preset['min_words']}–{preset['max_words']} words "
        f"(target {preset['target']})"
    )
    all_chunks: list[dict] = []
    for txt in txts:
        text = clean_text(txt.read_text(encoding="utf-8", errors="replace"))
        print(f"[chunk] {txt.name}")
        all_chunks.extend(records_from_text(txt.stem, txt.name, text, preset))
    write_chunks(chunks_path, all_chunks)
    if all_chunks:
        wcs = [c["word_count"] for c in all_chunks]
        print(
            f"Word counts: min={min(wcs)} median={sorted(wcs)[len(wcs)//2]} "
            f"max={max(wcs)} mean={sum(wcs)/len(wcs):.1f}"
        )
    return len(all_chunks)


def process_corpus(
    raw_dir: Path,
    extracted_dir: Path,
    chunks_path: Path,
    skip_extract: bool = False,
    preset_name: str = "default",
) -> int:
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {raw_dir}", file=sys.stderr)
        return 0

    preset = PRESETS[preset_name]
    extracted_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Chunk preset {preset_name}: "
        f"~{preset['min_words']}–{preset['max_words']} words "
        f"(target {preset['target']})"
    )

    all_chunks: list[dict] = []

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
        txt_path.write_text(text + "\n", encoding="utf-8")
        all_chunks.extend(records_from_text(stem, pdf.name, text, preset))

    write_chunks(chunks_path, all_chunks)
    return len(all_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDFs and build chunks.jsonl")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-extracted", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--out-chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="default",
        help="small ~100–150 words, large ~250–400, default ~150–250",
    )
    parser.add_argument(
        "--from-extracted",
        action="store_true",
        help="Chunk data/extracted/*.txt (or --out-extracted). No PDFs required.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Reuse existing .txt under out-extracted",
    )
    args = parser.parse_args()
    if args.from_extracted:
        n = process_extracted(args.out_extracted, args.out_chunks, args.preset)
    else:
        n = process_corpus(
            args.raw_dir,
            args.out_extracted,
            args.out_chunks,
            args.skip_extract,
            args.preset,
        )
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
