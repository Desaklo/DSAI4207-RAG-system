#!/usr/bin/env python3
"""Run the k-curve experiment: for each frozen question and each k in {0,1,2,4,8}.

Usage:
    # Dry-run: retrieve only, print top chunks for sample questions (no API calls)
    python src/run_experiment.py --dry-run

    # Same check on a prebuilt corpus, with BM25 and junk distractors
    python src/run_experiment.py --dry-run --chunk-size small --retrieval bm25 --junk
    python src/run_experiment.py --dry-run --chunk-size large --retrieval dense

    # Full run (needs OPENAI_API_KEY)
    python src/run_experiment.py
    python src/run_experiment.py --chunk-size large --retrieval bm25 --junk

Env:
    OPENAI_API_KEY   required for full run
    OPENAI_BASE_URL  optional OpenAI-compatible endpoint
    OPENAI_MODEL     default gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retriever import (  # noqa: E402
    CHUNK_SIZES,
    RETRIEVAL_MODES,
    Retriever,
    resolve_index_dir,
)

HK = ZoneInfo("Asia/Hong_Kong")
DEFAULT_QUESTIONS = ROOT / "questions" / "questions.json"
DEFAULT_INDEX = ROOT / "data" / "index"
DEFAULT_OUT = ROOT / "results" / "runs.jsonl"
K_VALUES = [0, 1, 2, 4, 8]

SYSTEM_PROMPT = (
    "You are a careful course tutor. Answer ONLY using the lecture chunks provided "
    "in the user message. For every factual claim you make, quote a short supporting "
    "sentence from those chunks. If the chunks do not support an answer, say exactly: "
    '"I don\'t know based on the provided chunks." Do not use outside knowledge. '
    "Do not invent citations."
)


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return (
            "No lecture chunks were retrieved (k=0 baseline).\n\n"
            f"Question: {question}\n\n"
            "Answer only from the (empty) context above. If you cannot support the "
            'answer from provided chunks, say: "I don\'t know based on the provided chunks."'
        )
    parts = ["Retrieved lecture chunks:\n"]
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | id={c['chunk_id']} | source={c['source_file']}]\n{c['text']}\n"
        )
    parts.append(f"\nQuestion: {question}\n")
    parts.append(
        "\nInstructions: Answer ONLY from the chunks above. Quote a supporting "
        'sentence for each claim. If unsupported, say: "I don\'t know based on the provided chunks."'
    )
    return "\n".join(parts)


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    assert isinstance(data, list)
    return data


def get_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def call_api(client, model: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def now_iso_hk() -> str:
    return datetime.now(HK).isoformat(timespec="seconds")


def dry_run(
    retriever: Retriever,
    questions: list[dict],
    sample_n: int = 2,
    k: int = 4,
    retrieval: str = "dense",
    junk: bool = False,
) -> None:
    print(
        f"=== DRY-RUN retrieval={retrieval} junk={junk} k={k} "
        f"(no API calls) ==="
    )
    for q in questions[:sample_n]:
        print("\n" + "=" * 72)
        print(f"Q[{q['id']}]: {q['question']}")
        if q.get("gold_hint_source"):
            print(f"  gold_hint_source: {q['gold_hint_source']}")
        hits = retriever.search(q["question"], k=k, retrieval=retrieval, junk=junk)
        for i, h in enumerate(hits, 1):
            preview = h["text"][:220].replace("\n", " ")
            tag = "  JUNK" if h.get("irrelevant") else ""
            print(
                f"  top{i}: {h['chunk_id']}  score={h['score']:.4f}  "
                f"src={h['source_file']}{tag}"
            )
            print(f"         {preview}…")
    print("\nDry-run OK.")


def run_full(
    retriever: Retriever,
    questions: list[dict],
    out_path: Path,
    model: str,
    k_values: list[int],
    retrieval: str = "dense",
    junk: bool = False,
    chunk_size: str | None = None,
) -> None:
    client = get_client()
    if client is None:
        print(
            "ERROR: OPENAI_API_KEY is not set. Skipping full experiment.\n"
            "Copy .env.example to .env, set your key, then re-run.",
            file=sys.stderr,
        )
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append mode so partial runs are kept; team can truncate if needed
    n_written = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for q in questions:
            qid = q["id"]
            question = q["question"]
            for k in k_values:
                hits = (
                    retriever.search(question, k=k, retrieval=retrieval, junk=junk)
                    if k > 0
                    else []
                )
                user_prompt = build_user_prompt(question, hits)
                print(
                    f"[{now_iso_hk()}] q={qid} k={k} retrieval={retrieval} "
                    f"junk={junk} calling {model}…"
                )
                answer = call_api(client, model, user_prompt)
                record = {
                    "question_id": qid,
                    "question": question,
                    "k": k,
                    "retrieval": retrieval,
                    "chunk_size": chunk_size or retriever.info.get("chunk_size"),
                    "junk": junk,
                    "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
                    "irrelevant_chunk_ids": [
                        h["chunk_id"] for h in hits if h.get("irrelevant")
                    ],
                    "retrieved_scores": [round(h["score"], 6) for h in hits],
                    "prompt": {
                        "system": SYSTEM_PROMPT,
                        "user": user_prompt,
                    },
                    "answer": answer,
                    "model": model,
                    "temperature": 0,
                    "timestamp": now_iso_hk(),
                    "timestamp_utc": datetime.now(timezone.utc)
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z"),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                n_written += 1
    print(f"Wrote {n_written} rows → {out_path}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="DSAI4207 RAG k-curve experiment")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Index directory. Overrides --chunk-size. Default: data/index",
    )
    parser.add_argument(
        "--chunk-size",
        choices=CHUNK_SIZES,
        default=None,
        help="Use the prebuilt small (~100–150 words) or large (~250–400) index",
    )
    parser.add_argument(
        "--retrieval",
        choices=RETRIEVAL_MODES,
        default="dense",
        help="dense = MiniLM/FAISS, bm25 = lexical",
    )
    parser.add_argument(
        "--junk",
        action="store_true",
        help="Mix low-ranked irrelevant chunks into the retrieved context",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="Retrieve only; no API")
    parser.add_argument("--sample-n", type=int, default=2, help="Questions for dry-run")
    parser.add_argument("--dry-k", type=int, default=4, help="k for dry-run retrieval")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=K_VALUES,
        help="k values to evaluate (default: 0 1 2 4 8)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} frozen questions from {args.questions}")
    index_dir = resolve_index_dir(args.chunk_size, args.index_dir)
    retriever = Retriever(index_dir)
    print(
        f"Index: {index_dir}  chunks={retriever.info['num_chunks']}  "
        f"chunk_size={retriever.info.get('chunk_size', 'custom')}  "
        f"words={retriever.info.get('word_count_min', '?')}"
        f"–{retriever.info.get('word_count_max', '?')}  "
        f"backend={retriever.info['backend']}  "
        f"retrieval={args.retrieval}  junk={args.junk}"
    )

    if args.dry_run:
        dry_run(
            retriever,
            questions,
            sample_n=args.sample_n,
            k=args.dry_k,
            retrieval=args.retrieval,
            junk=args.junk,
        )
        return

    run_full(
        retriever,
        questions,
        args.out,
        args.model,
        list(args.k),
        retrieval=args.retrieval,
        junk=args.junk,
        chunk_size=args.chunk_size or retriever.info.get("chunk_size"),
    )


if __name__ == "__main__":
    main()
