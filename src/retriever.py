#!/usr/bin/env python3
"""Top-k retrieval over a chunk corpus: dense MiniLM/FAISS or lexical BM25.

Junk mode mixes low-ranked distractors into the k slots (same k, about half swapped).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data" / "index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZES = ("small", "large")
RETRIEVAL_MODES = ("dense", "bm25")
_TOKEN = re.compile(r"[a-z0-9]+")
_MODEL_CACHE: dict[str, object] = {}


def index_dir_for(chunk_size: str) -> Path:
    if chunk_size not in CHUNK_SIZES:
        raise ValueError(f"chunk_size must be one of {CHUNK_SIZES}")
    return ROOT / "data" / "index" / chunk_size


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over the chunk texts. Built in memory at load time."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tfs: list[dict[str, int]] = []
        self.doc_len: list[int] = []
        df: dict[str, int] = {}
        for text in texts:
            tf: dict[str, int] = {}
            for tok in tokenize(text):
                tf[tok] = tf.get(tok, 0) + 1
            self.tfs.append(tf)
            length = sum(tf.values())
            self.doc_len.append(length)
            for tok in tf:
                df[tok] = df.get(tok, 0) + 1
        self.n = len(self.tfs)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.idf = {
            tok: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0)
            for tok, freq in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if self.n == 0 or self.avgdl <= 0:
            return out
        seen: set[str] = set()
        terms: list[str] = []
        for tok in tokenize(query):
            if tok not in seen:
                seen.add(tok)
                terms.append(tok)
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tfmap in enumerate(self.tfs):
                tf = tfmap.get(term)
                if not tf:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * (tf * (self.k1 + 1.0)) / denom
        return out


def _mix_junk(ranked: list[dict], k: int) -> list[dict]:
    """Keep k slots. About half become the lowest-ranked other chunks, interleaved."""
    if k <= 0:
        return []
    k = min(k, len(ranked))
    n_junk = 1 if k == 1 else k // 2
    n_rel = k - n_junk
    relevant = []
    for hit in ranked[:n_rel]:
        row = dict(hit)
        row["irrelevant"] = False
        relevant.append(row)
    used = {row["chunk_id"] for row in relevant}
    distractors = []
    for hit in reversed(ranked):
        if hit["chunk_id"] in used:
            continue
        row = dict(hit)
        row["irrelevant"] = True
        distractors.append(row)
        if len(distractors) >= n_junk:
            break
    mixed: list[dict] = []
    ri = ji = 0
    while len(mixed) < k and (ri < len(relevant) or ji < len(distractors)):
        if ri < len(relevant):
            mixed.append(relevant[ri])
            ri += 1
        if len(mixed) >= k:
            break
        if ji < len(distractors):
            mixed.append(distractors[ji])
            ji += 1
    return mixed


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

        self._model = None  # lazy, shared by model name
        self._bm25 = BM25Index([row.get("text") or "" for row in self.meta])

    def _encode(self, query: str) -> np.ndarray:
        if self._model is None:
            cached = _MODEL_CACHE.get(self.model_name)
            if cached is None:
                from sentence_transformers import SentenceTransformer

                cached = SentenceTransformer(self.model_name)
                _MODEL_CACHE[self.model_name] = cached
            self._model = cached
        vec = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vec, dtype=np.float32)

    def _dense_scores(self, query: str) -> np.ndarray:
        q = self._encode(query)
        n = len(self.meta)
        if self._faiss is not None:
            scores, idxs = self._faiss.search(q, n)
            out = np.full(n, -np.inf, dtype=np.float32)
            for score, idx in zip(scores[0], idxs[0]):
                i = int(idx)
                if i >= 0:
                    out[i] = float(score)
            return out
        return (self.embeddings @ q[0].T).astype(np.float32)

    def search(
        self,
        query: str,
        k: int,
        *,
        retrieval: str = "dense",
        junk: bool = False,
    ) -> list[dict]:
        """Return k hits. retrieval is 'dense' or 'bm25'. junk mixes in distractors."""
        if retrieval not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval must be one of {RETRIEVAL_MODES}")
        if k <= 0 or not self.meta:
            return []
        scores = self._dense_scores(query) if retrieval == "dense" else self._bm25.scores(query)
        order = np.argsort(-scores, kind="mergesort")
        ranked: list[dict] = []
        for idx in order:
            i = int(idx)
            row = dict(self.meta[i])
            row["score"] = float(scores[i])
            row["retrieval"] = retrieval
            row["irrelevant"] = False
            ranked.append(row)
        k = min(k, len(ranked))
        if junk:
            return _mix_junk(ranked, k)
        return ranked[:k]


def resolve_index_dir(chunk_size: str | None, index_dir: Path | None) -> Path:
    if index_dir is not None:
        return index_dir
    if chunk_size is not None:
        return index_dir_for(chunk_size)
    return DEFAULT_INDEX


def demo(
    query: str,
    k: int = 4,
    index_dir: Path = DEFAULT_INDEX,
    retrieval: str = "dense",
    junk: bool = False,
) -> None:
    r = Retriever(index_dir)
    hits = r.search(query, k, retrieval=retrieval, junk=junk)
    size = r.info.get("chunk_size") or "custom"
    print(
        f"Query: {query!r}  (k={k}, retrieval={retrieval}, "
        f"chunk_size={size}, junk={junk})"
    )
    for i, h in enumerate(hits, 1):
        preview = h["text"][:160].replace("\n", " ")
        tag = "  JUNK" if h.get("irrelevant") else ""
        print(
            f"  {i}. {h['chunk_id']}  score={h['score']:.4f}  "
            f"[{h['source_file']}]  words={h.get('word_count', '?')}{tag}"
        )
        print(f"     {preview}…")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Query the lecture index")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--index-dir", type=Path, default=None)
    p.add_argument("--chunk-size", choices=CHUNK_SIZES, default=None)
    p.add_argument("--retrieval", choices=RETRIEVAL_MODES, default="dense")
    p.add_argument(
        "--junk",
        action="store_true",
        help="Mix low-ranked distractor chunks into the k slots",
    )
    args = p.parse_args()
    demo(
        args.query,
        args.k,
        resolve_index_dir(args.chunk_size, args.index_dir),
        args.retrieval,
        args.junk,
    )
