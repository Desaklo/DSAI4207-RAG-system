# DSAI 4207 — RAG unsupported-claim curve / Кривая неподтверждённых утверждений

**EN** Measure whether the unsupported-claim rate of one hosted API model falls as retrieved chunk count `k ∈ {0,1,2,4,8}` on a fixed folder of DSAI 4207 lecture notes.

**RU** Измерить, снижается ли доля неподтверждённых утверждений одной hosted API-модели при росте числа извлечённых чанков `k ∈ {0,1,2,4,8}` на фиксированной папке лекций DSAI 4207.

Hypothesis / Гипотеза: unsupported-claim rate at **k=4** is **≥15 percentage points lower** than at **k=0**.

The main hypothesis still varies only **k**. No web search at answer time. Temperature = 0. Same model for every k.

You can also switch three retrieval settings in the demo and on the command line: junk context on/off, dense vs BM25, and small vs large chunks. Those settings are not a finished results table. Do not treat the demo chart as measured rates.

Extracted lecture text and three indexes are already in this repo: the original ~150–250 word index, plus prebuilt **small** and **large** corpora. Raw slide PDFs are omitted. Do not commit `.env`.

---

## Quick start / Быстрый старт

```bash
git clone https://github.com/Desaklo/DSAI4207-RAG-system.git
cd DSAI4207-RAG-system

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Create an API key at https://console.x.ai/ and paste it into `.env`, replacing `PASTE_YOUR_XAI_API_KEY_HERE`. Keep the other two lines (`OPENAI_BASE_URL` and `OPENAI_MODEL`). The variable name stays `OPENAI_API_KEY` because the client is OpenAI-compatible; the example points it at xAI.

RU: ключ вставляется только в файл `.env` в корне репозитория, строка `OPENAI_API_KEY=...`. Файл `.env` в git не попадает.

Retrieval-only check (no API calls; indexes are already built):

```bash
python src/run_experiment.py --dry-run
python src/retriever.py "What is LoRA?" -k 4 --chunk-size small --retrieval dense
python src/retriever.py "What is LoRA?" -k 4 --chunk-size large --retrieval bm25 --junk
```

Full experiment (uses the xAI key; writes `results/runs.jsonl`):

```bash
python src/run_experiment.py
python src/run_experiment.py --chunk-size large --retrieval bm25 --junk
```

Local demo:

```bash
uvicorn src.webapp:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/**. Without a key the UI still shows retrieved chunks and the demo curve; model answers appear after the key is in `.env`.

PDF re-extraction is optional and needs `poppler-utils` plus lecture PDFs you place in `data/raw/` yourself (`sudo apt install poppler-utils`, then `python src/extract_and_chunk.py`). Those PDFs are not in the clone.

---

## Retrieval controls

The demo page has the existing **k** chips and preset questions, plus:

| Control | Choices | What it does |
|---------|---------|----------------|
| Chunk size | Small / Large | Small chunks are about 100–150 words. Large chunks are about 250–400 words. A short file, or a short tail merged into the previous chunk, can sit a little outside that band. The page switches prebuilt indexes. It does not rebuild them. |
| Retrieval | Dense / BM25 | Dense is the MiniLM embedding index (FAISS). BM25 is lexical keyword search over the same chunks. |
| Junk context | Off / On | One switch. Off keeps the top k matches. On keeps k slots but replaces about half of them with the lowest-ranked other chunks from that same search, interleaved with the real hits. |

CLI flags match the page: `--chunk-size small|large`, `--retrieval dense|bm25`, and `--junk`.

```bash
python src/run_experiment.py --dry-run --chunk-size small --retrieval dense
python src/run_experiment.py --dry-run --chunk-size small --retrieval bm25 --junk
python src/run_experiment.py --dry-run --chunk-size large --retrieval dense --junk
python src/run_experiment.py --dry-run --chunk-size large --retrieval bm25
```

Leaving those flags off still uses `data/index` (the original ~150–250 word corpus) and dense search.

**RU:** В интерфейсе три переключателя: размер чанка (small/large), поиск dense или BM25, и один тумблер junk (подмешать нерелевантные куски). Те же флаги есть у `run_experiment.py`. График на сайте — демо, не результаты эксперимента.

### Rebuild the small and large indexes

Only needed if you change the extracted text. From the repo root, with the venv active:

```bash
python src/extract_and_chunk.py --from-extracted --preset small \
  --out-chunks data/chunks/small/chunks.jsonl
python src/extract_and_chunk.py --from-extracted --preset large \
  --out-chunks data/chunks/large/chunks.jsonl
python src/build_index.py --chunks data/chunks/small/chunks.jsonl \
  --index-dir data/index/small --chunk-size small
python src/build_index.py --chunks data/chunks/large/chunks.jsonl \
  --index-dir data/index/large --chunk-size large
```

`--from-extracted` reads `data/extracted/*.txt`. It does not need the raw PDFs.

---

## Pipeline / Конвейер

Steps 1–2 are already done in this clone (`data/extracted/`, `data/chunks/`, `data/index/`). Re-run them only if you add your own PDFs under `data/raw/`.

### 1. Extract + chunk

```bash
source .venv/bin/activate
python src/extract_and_chunk.py
```

- Reads `data/raw/*.pdf` via `pdftotext`
- Writes cleaned text → `data/extracted/*.txt`
- Writes ~150–250 word chunks → `data/chunks/chunks.jsonl`
  (fields: `chunk_id`, `source_file`, `word_count`, `text`, …)
- For the demo sizes, use `--from-extracted --preset small|large` (see Retrieval controls)

### 2. Build embedding index (CPU)

```bash
python src/build_index.py
```

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Outputs under `data/index/`: `embeddings.npy`, `meta.jsonl`, `faiss.index` (if faiss-cpu works), `index_info.json`

### 3. Dry-run retrieval (no API key needed)

```bash
python src/run_experiment.py --dry-run
# optional: --sample-n 2 --dry-k 4
```

Prints top chunks for the first sample questions.

Quick single query:

```bash
python src/retriever.py "What is LoRA?" -k 4 --chunk-size small
python src/retriever.py "What is LoRA?" -k 4 --chunk-size large --retrieval bm25 --junk
```

### 4. Full experiment (needs `OPENAI_API_KEY`)

```bash
python src/run_experiment.py
# logs → results/runs.jsonl
# each row: question_id, k, retrieval, chunk_size, junk, retrieved_chunk_ids, prompt, answer, model, timestamp
```

Expected call count ≈ `num_questions × 5` (for k = 0,1,2,4,8). With 35 questions → ~175 calls.

### 5. Marking sheet (manual labels)

After runs exist:

```bash
python src/marking_template.py
# → results/marking_sheet.csv
```

Team fills `label` with: **`supported` | `unsupported` | `abstention`**, judging each claim sentence **against the PASSED chunks only** (not against world knowledge / model memory).

Primary metric: **unsupported-claim rate** per k. Also track support rate and abstention rate.

---


---

## Demo website / Демо-сайт

Local single-page UI: type a question, pick **k ∈ {0,1,2,4,8}**, chunk size, dense or BM25, and the junk switch. Preset questions stay. Retrieved chunks and a Chart.js curve are shown (demo placeholder until marking exists). Junk chunks are marked in red. The model prompt does not say which chunks are junk.

```bash
source .venv/bin/activate
pip install -r requirements.txt   # adds fastapi + uvicorn if missing
uvicorn src.webapp:app --reload --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000/**

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Demo UI |
| `GET /api/questions` | Frozen questions (35) |
| `POST /api/ask` | `{ "question", "k", "retrieval", "chunk_size", "junk" }` → chunks (+ xAI answer if `OPENAI_API_KEY` is set in `.env`) |
| `GET /api/curve` | Real marking metrics if available, else `results/demo_curve.json` |
| `GET /api/curve?force_demo=true` | Always load the sample demo curve |
| `GET /api/health` | Index + API-key status |

**Without an API key** the UI still works: retrieval + chunk display + charts; generation is skipped with a clear message.

**RU tip:** Без ключа сайт всё равно показывает поиск по лекциям и демо-график; ответ модели появится после `OPENAI_API_KEY` в `.env`.

Layout additions: `src/webapp.py`, `web/static/index.html`, `results/demo_curve.json`.

## Project layout / Структура

```
.
  data/inventory.txt
  data/extracted/            # lecture text already extracted (no raw PDFs)
  data/chunks/chunks.jsonl   # original ~150–250 word chunks
  data/chunks/small/         # ~100–150 word chunks
  data/chunks/large/         # ~250–400 word chunks
  data/index/                # original MiniLM + FAISS index
  data/index/small/          # prebuilt small-chunk index
  data/index/large/          # prebuilt large-chunk index
  questions/questions.json   # frozen questions (≥30)
  results/                   # demo_curve.json; runs.jsonl after the experiment
  web/static/                # demo UI (index.html + Chart.js CDN)
  src/
    extract_and_chunk.py
    build_index.py
    retriever.py
    run_experiment.py
    marking_template.py
    webapp.py                # FastAPI demo server
  proposal.pdf
  requirements.txt
  .env.example               # copy to .env; .env is gitignored
  .gitignore
  README.md
```

---

## Notes / Замечания

- **EN** Support is marked against passed chunks, not the model’s private memory — course notes may already be in training data.
- **RU** Поддержка размечается по переданным чанкам, а не по «памяти» модели — материалы курса могли уже попасть в обучение.
- No fabricated experimental numbers. Numbers come only from real API runs + human marks.
- CPU-friendly; no GPU required for embeddings.

Team: Vsevolod Kevrukh, Choi Wai Lun, Ng Jun — DSAI 4207 course project.
