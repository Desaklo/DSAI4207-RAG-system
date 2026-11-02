#!/usr/bin/env python3
"""Local demo UI for DSAI 4207 RAG: ask questions, pick k, view curve charts.

Run from project root:
    source .venv/bin/activate
    uvicorn src.webapp:app --reload --host 127.0.0.1 --port 8000

Or:
    python -m uvicorn src.webapp:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

from retriever import (  # noqa: E402
    CHUNK_SIZES,
    RETRIEVAL_MODES,
    Retriever,
    index_dir_for,
)
from run_experiment import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_prompt,
    call_api,
    get_client,
    load_questions,
)

QUESTIONS_PATH = ROOT / "questions" / "questions.json"
DEMO_CURVE = ROOT / "results" / "demo_curve.json"
RUNS_PATH = ROOT / "results" / "runs.jsonl"
MARKING_PATH = ROOT / "results" / "marking_sheet.csv"
STATIC_DIR = ROOT / "web" / "static"
ALLOWED_K = {0, 1, 2, 4, 8}

app = FastAPI(title="DSAI 4207 RAG Demo", version="0.1.0")

_retrievers: dict[str, Retriever] = {}


def get_retriever(chunk_size: str) -> Retriever:
    if chunk_size not in CHUNK_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"chunk_size must be one of {list(CHUNK_SIZES)}",
        )
    cached = _retrievers.get(chunk_size)
    if cached is None:
        cached = Retriever(index_dir_for(chunk_size))
        _retrievers[chunk_size] = cached
    return cached


def _index_summary(chunk_size: str) -> dict:
    info_path = index_dir_for(chunk_size) / "index_info.json"
    if not info_path.exists():
        return {"ok": False, "error": f"missing {info_path}"}
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return {
        "ok": True,
        "num_chunks": info.get("num_chunks"),
        "word_count_min": info.get("word_count_min"),
        "word_count_max": info.get("word_count_max"),
        "word_count_mean": info.get("word_count_mean"),
        "embed_model": info.get("embed_model"),
    }


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(4, description="Number of chunks to retrieve")
    junk: bool = Field(False, description="Mix low-ranked distractors into the k slots")
    retrieval: Literal["dense", "bm25"] = "dense"
    chunk_size: Literal["small", "large"] = "small"


class AskResponse(BaseModel):
    question: str
    k: int
    retrieval: str
    chunk_size: str
    junk: bool
    answer: str | None
    generation_mode: str  # "openai" | "retrieval_only"
    message: str
    chunks: list[dict]
    model: str | None = None


def _metrics_from_marking(path: Path) -> dict | None:
    """Aggregate unsupported / support / abstention rates from marking CSV."""
    if not path.exists():
        return None
    by_k: dict[int, dict[str, int]] = defaultdict(
        lambda: {"supported": 0, "unsupported": 0, "abstention": 0, "other": 0}
    )
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    labeled = 0
    for row in rows:
        label = (row.get("label") or "").strip().lower()
        if not label:
            continue
        try:
            k = int(row["k"])
        except (KeyError, ValueError, TypeError):
            continue
        labeled += 1
        if label in ("supported", "unsupported", "abstention"):
            by_k[k][label] += 1
        else:
            by_k[k]["other"] += 1
    if labeled == 0:
        return None

    k_values = sorted(by_k.keys())
    if not k_values:
        return None
    unsupported_rates = []
    support_rates = []
    abstention_rates = []
    n_claims = {}
    for k in k_values:
        counts = by_k[k]
        total = sum(counts.values())
        n_claims[str(k)] = total
        if total == 0:
            unsupported_rates.append(None)
            support_rates.append(None)
            abstention_rates.append(None)
        else:
            unsupported_rates.append(round(counts["unsupported"] / total, 4))
            support_rates.append(round(counts["supported"] / total, 4))
            abstention_rates.append(round(counts["abstention"] / total, 4))

    return {
        "source": "marking",
        "label": "Real metrics from marking_sheet.csv",
        "note": "Computed from human labels in results/marking_sheet.csv.",
        "metric": "unsupported_claim_rate",
        "k_values": k_values,
        "unsupported_claim_rate": unsupported_rates,
        "support_rate": support_rates,
        "abstention_rate": abstention_rates,
        "n_claims_per_k": n_claims,
    }


def _placeholder_from_runs(path: Path) -> dict | None:
    """If runs exist but no labels yet, show a runs-count placeholder per k."""
    if not path.exists():
        return None
    counts: dict[int, int] = defaultdict(int)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            counts[int(rec["k"])] += 1
    if not counts:
        return None
    k_values = sorted(counts.keys())
    return {
        "source": "runs_placeholder",
        "label": "Runs logged — marking not filled yet (placeholder counts)",
        "note": (
            "results/runs.jsonl exists but marking labels are empty. "
            "Bars show answer counts per k, not unsupported-claim rate. "
            "Fill marking_sheet.csv for the real curve."
        ),
        "metric": "n_answers",
        "k_values": k_values,
        "n_answers": [counts[k] for k in k_values],
        "unsupported_claim_rate": None,
    }


def load_demo_curve() -> dict:
    if DEMO_CURVE.exists():
        return json.loads(DEMO_CURVE.read_text(encoding="utf-8"))
    return {
        "source": "demo",
        "label": "DEMO / EXAMPLE DATA — not real experiment results",
        "note": "Missing results/demo_curve.json; inline fallback.",
        "metric": "unsupported_claim_rate",
        "k_values": [0, 1, 2, 4, 8],
        "unsupported_claim_rate": [0.72, 0.58, 0.45, 0.31, 0.28],
        "support_rate": [0.10, 0.22, 0.35, 0.52, 0.55],
        "abstention_rate": [0.18, 0.20, 0.20, 0.17, 0.17],
    }


@app.get("/api/health")
def health():
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    indexes = {size: _index_summary(size) for size in CHUNK_SIZES}
    ok = all(item.get("ok") for item in indexes.values())
    small_n = indexes.get("small", {}).get("num_chunks")
    return {
        "ok": ok,
        "has_api_key": has_key,
        "num_chunks": small_n,
        "indexes": indexes,
        "retrieval_modes": list(RETRIEVAL_MODES),
        "chunk_sizes": list(CHUNK_SIZES),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    }


@app.get("/api/questions")
def api_questions():
    qs = load_questions(QUESTIONS_PATH)
    # Do not expose gold_hint_source to the UI (team-only)
    out = [{"id": q["id"], "question": q["question"]} for q in qs]
    return {"count": len(out), "questions": out}


@app.post("/api/ask", response_model=AskResponse)
def api_ask(body: AskRequest):
    if body.k not in ALLOWED_K:
        raise HTTPException(
            status_code=400,
            detail=f"k must be one of {sorted(ALLOWED_K)}",
        )
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is empty")
    if body.retrieval not in RETRIEVAL_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"retrieval must be one of {list(RETRIEVAL_MODES)}",
        )

    retriever = get_retriever(body.chunk_size)
    hits = (
        retriever.search(
            question,
            k=body.k,
            retrieval=body.retrieval,
            junk=body.junk,
        )
        if body.k > 0
        else []
    )
    chunks = [
        {
            "chunk_id": h["chunk_id"],
            "source_file": h["source_file"],
            "score": round(float(h["score"]), 6),
            "text": h["text"],
            "word_count": h.get("word_count"),
            "irrelevant": bool(h.get("irrelevant")),
            "retrieval": h.get("retrieval", body.retrieval),
        }
        for h in hits
    ]

    client = get_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if client is None:
        return AskResponse(
            question=question,
            k=body.k,
            retrieval=body.retrieval,
            chunk_size=body.chunk_size,
            junk=body.junk,
            answer=None,
            generation_mode="retrieval_only",
            message=(
                "OPENAI_API_KEY is not set — showing retrieval only. "
                "Set the key in .env to generate an answer (same prompt as run_experiment)."
            ),
            chunks=chunks,
            model=None,
        )

    user_prompt = build_user_prompt(question, hits)
    try:
        answer = call_api(client, model, user_prompt)
    except Exception as e:
        return AskResponse(
            question=question,
            k=body.k,
            retrieval=body.retrieval,
            chunk_size=body.chunk_size,
            junk=body.junk,
            answer=None,
            generation_mode="retrieval_only",
            message=f"API call failed ({e}). Retrieved chunks are still shown below.",
            chunks=chunks,
            model=model,
        )

    return AskResponse(
        question=question,
        k=body.k,
        retrieval=body.retrieval,
        chunk_size=body.chunk_size,
        junk=body.junk,
        answer=answer,
        generation_mode="openai",
        message="Answer generated with the experiment system prompt.",
        chunks=chunks,
        model=model,
    )


@app.get("/api/curve")
def api_curve(force_demo: bool = False):
    """Return real marking metrics if available, else demo_curve.json."""
    if force_demo:
        data = load_demo_curve()
        data["forced"] = True
        return data

    real = _metrics_from_marking(MARKING_PATH)
    if real is not None:
        return real

    runs_ph = _placeholder_from_runs(RUNS_PATH)
    if runs_ph is not None:
        # Prefer demo curve for the unsupported-rate chart shape, but note runs exist
        demo = load_demo_curve()
        demo["runs_note"] = runs_ph["note"]
        demo["runs_n_answers"] = {
            str(k): n for k, n in zip(runs_ph["k_values"], runs_ph["n_answers"])
        }
        demo["label"] = (
            "DEMO curve (runs exist but marking empty) — not real unsupported rates"
        )
        return demo

    return load_demo_curve()


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web/static/index.html missing")
    return FileResponse(index_path)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.webapp:app", host="127.0.0.1", port=8000, reload=True)
