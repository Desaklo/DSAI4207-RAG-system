#!/usr/bin/env python3
"""Embed chunks with all-MiniLM-L6-v2 and build a FAISS (or numpy) index.

Usage:
    python src/build_index.py
    python src/build_index.py --chunks data/chunks/chunks.jsonl --index-dir data/index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
DEFAULT_INDEX = ROOT / "data" / "index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build(
    chunks_path: Path,
    index_dir: Path,
    model_name: str = EMBED_MODEL,
    chunk_size: str | None = None,
) -> None:
    chunks = load_chunks(chunks_path)
    if not chunks:
        print(f"No chunks in {chunks_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading embedding model: {model_name}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks on CPU…")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine via inner product
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    index_dir.mkdir(parents=True, exist_ok=True)
    emb_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "meta.jsonl"
    faiss_path = index_dir / "faiss.index"
    info_path = index_dir / "index_info.json"

    np.save(emb_path, embeddings)
    with meta_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            # Store metadata without full text duplication optional — keep text for retrieval display
            f.write(
                json.dumps(
                    {
                        "chunk_id": c["chunk_id"],
                        "source_file": c["source_file"],
                        "word_count": c["word_count"],
                        "text": c["text"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Prefer FAISS; fall back to numpy-only if import fails
    used_faiss = False
    try:
        import faiss

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on L2-normalized = cosine
        index.add(embeddings)
        faiss.write_index(index, str(faiss_path))
        used_faiss = True
        print(f"FAISS index written → {faiss_path} ({index.ntotal} vectors, dim={dim})")
    except Exception as e:
        print(f"FAISS unavailable ({e}); numpy cosine search will be used at query time.")
        if faiss_path.exists():
            faiss_path.unlink()

    word_counts = [int(c.get("word_count") or len(str(c.get("text", "")).split())) for c in chunks]
    info = {
        "embed_model": model_name,
        "num_chunks": len(chunks),
        "dim": int(embeddings.shape[1]),
        "normalized": True,
        "backend": "faiss" if used_faiss else "numpy",
        "embeddings_file": emb_path.name,
        "meta_file": meta_path.name,
        "faiss_file": faiss_path.name if used_faiss else None,
        "word_count_min": min(word_counts),
        "word_count_max": max(word_counts),
        "word_count_mean": round(sum(word_counts) / len(word_counts), 2),
    }
    if chunk_size:
        info["chunk_size"] = chunk_size
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved embeddings → {emb_path}")
    print(f"Saved meta → {meta_path}")
    print(f"Index info → {info_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embedding + FAISS index")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--model", type=str, default=EMBED_MODEL)
    parser.add_argument(
        "--chunk-size",
        choices=["small", "large"],
        default=None,
        help="Label stored in index_info.json (small or large demo corpus)",
    )
    args = parser.parse_args()
    build(args.chunks, args.index_dir, args.model, args.chunk_size)


if __name__ == "__main__":
    main()
