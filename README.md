# DSAI 4207 — RAG unsupported-claim curve / Кривая неподтверждённых утверждений

**EN** Measure whether the unsupported-claim rate of one hosted API model falls as retrieved chunk count `k ∈ {0,1,2,4,8}` on a fixed folder of DSAI 4207 lecture notes.

**RU** Измерить, снижается ли доля неподтверждённых утверждений одной hosted API-модели при росте числа извлечённых чанков `k ∈ {0,1,2,4,8}` на фиксированной папке лекций DSAI 4207.

Hypothesis / Гипотеза: unsupported-claim rate at **k=4** is **≥15 percentage points lower** than at **k=0**.

Only factor varied: **k**. No web search at answer time. Temperature = 0. Same model for every k.

Extracted lecture text, chunks, and the embedding index are already in this repo. Raw slide PDFs are omitted. Do not commit `.env`.

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

Retrieval-only check (no API calls; index is already built):

```bash
python src/run_experiment.py --dry-run
python src/retriever.py "What is LoRA?" -k 4
```

Full experiment (uses the xAI key; writes `results/runs.jsonl`):

```bash
python src/run_experiment.py
```

Local demo:

```bash
uvicorn src.webapp:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/**. Without a key the UI still shows retrieved chunks and the demo curve; model answers appear after the key is in `.env`.

PDF re-extraction is optional and needs `poppler-utils` plus lecture PDFs you place in `data/raw/` yourself (`sudo apt install poppler-utils`, then `python src/extract_and_chunk.py`). Those PDFs are not in the clone.

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
python src/retriever.py "What is LoRA?" -k 4
```

### 4. Full experiment (needs `OPENAI_API_KEY`)

```bash
python src/run_experiment.py
# logs → results/runs.jsonl
# each row: question_id, k, retrieved_chunk_ids, prompt, answer, model, timestamp
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

Local single-page UI: type a question, pick **k ∈ {0,1,2,4,8}**, click preset questions, see retrieved chunks, and a Chart.js curve (demo placeholder until marking exists).

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
| `POST /api/ask` | `{ "question", "k" }` → chunks (+ xAI answer if `OPENAI_API_KEY` is set in `.env`) |
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
  data/extracted/     # lecture text already extracted (no raw PDFs)
  data/chunks/        # chunks.jsonl
  data/index/         # embeddings.npy, faiss.index, meta.jsonl
  questions/questions.json   # frozen questions (≥30)
  results/            # demo_curve.json; runs.jsonl after the experiment
  web/static/         # demo UI (index.html + Chart.js CDN)
  src/
    extract_and_chunk.py
    build_index.py
    retriever.py
    run_experiment.py
    marking_template.py
    webapp.py         # FastAPI demo server
  proposal.pdf
  requirements.txt
  .env.example        # copy to .env; .env is gitignored
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
