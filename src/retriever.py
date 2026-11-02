#!/usr/bin/env python3
"""Local top-k retriever over the cached embeddings / FAISS index."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data" / "index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, index_dir: Path = DEFAULT_INDEX, model_name: str | None = None):
        self.index_dir = Path(index_dir)
        info_path = self.index_dir / "index_info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"Missing {info_path}. Run: python src/build_index.py"
            )
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self.model_name = model_name or self.info.get("embed_model", EMBED_MODEL)

        self.embeddings = np.load(self.index_dir / self.info["embeddings_file"])
        self.meta: list[dict] = []
        with (self.index_dir / self.info["meta_file"]).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.meta.append(json.loads(line))
        assert len(self.meta) == self.embeddings.shape[0]

        self._faiss = None
        faiss_file = self.info.get("faiss_file")
        if faiss_file:
            faiss_path = self.index_dir / faiss_file
            if faiss_path.exists():
                import faiss

                self._faiss = faiss.read_index(str(faiss_path))

        self._model = None  # lazy

    def _encode(self, query: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vec = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vec, dtype=np.float32)

    def search(self, query: str, k: int) -> list[dict]:
        """Return top-k hits as list of meta dicts with score added."""
        if k <= 0:
            return []
        k = min(k, len(self.meta))
        q = self._encode(query)

        if self._faiss is not None:
            scores, idxs = self._faiss.search(q, k)
            scores = scores[0]
            idxs = idxs[0]
        else:
            # Cosine via dot product (vectors are L2-normalized)
            sims = (self.embeddings @ q[0].T).astype(np.float32)
            idxs = np.argsort(-sims)[:k]
            scores = sims[idxs]

        hits = []
        for idx, score in zip(idxs, scores):
            if int(idx) < 0:
                continue
            row = dict(self.meta[int(idx)])
            row["score"] = float(score)
            hits.append(row)
        return hits


def demo(query: str, k: int = 4, index_dir: Path = DEFAULT_INDEX) -> None:
    r = Retriever(index_dir)
    hits = r.search(query, k)
    print(f"Query: {query!r}  (k={k})")
    for i, h in enumerate(hits, 1):
        preview = h["text"][:160].replace("\n", " ")
        print(f"  {i}. {h['chunk_id']}  score={h['score']:.4f}  [{h['source_file']}]")
        print(f"     {preview}…")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    args = p.parse_args()
    demo(args.query, args.k, args.index_dir)
