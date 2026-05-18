# PDF Ingestion API (FastAPI)

Starter implementation for ingesting exam-style PDF documents.

## What this includes
- `POST /v1/ingest/pdf`: upload PDF and ingest text chunks
- `POST /v1/query`: retrieve top matching chunks from ingested manifests
- `POST /v1/agentic/query`: planner/retriever/synthesizer/critic orchestration with guardrails
- `GET /health`: health check
- PDF extraction with `pypdf`
- Layout-aware block parsing (heading/list/table/paragraph)
- Chunk generation with page-aware metadata
- Automatic metadata enrichment (months/topics/entities)
- True hybrid retrieval: BM25 + vector + reranker
- Embeddings + local Qdrant vector indexing (during ingestion)
- Local manifest output under `data/processed/<doc_id>.json`

## Quick start
1. Create virtual env and activate it.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Copy env template:
   `copy .env.example .env`
4. Run server:
   `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

## Example API calls
Ingest a PDF:
`curl -X POST "http://localhost:8000/v1/ingest/pdf" -F "file=@C:/path/doc.pdf"`

Duplicate protection:
- Uploading the same PDF content again is blocked using SHA-256 content hash comparison.
- API returns `409 Conflict` with `detail.error = "duplicate_file"` and existing document metadata.

Query ingested content:
`curl -X POST "http://localhost:8000/v1/query" -H "Content-Type: application/json" -d "{\"query\":\"RBI repo rate\",\"top_k\":2}"`

Query text + image evidence together (when multimodal ingest is enabled):
`curl -X POST "http://localhost:8000/v1/query" -H "Content-Type: application/json" -d "{\"query\":\"show chart trends\",\"top_k\":5,\"use_vector\":true,\"include_images\":true,\"response_mode\":\"full\"}"`

Agentic query:
`curl -X POST "http://localhost:8000/v1/agentic/query" -H "Content-Type: application/json" -d "{\"query\":\"What is route of Pune Metro Rail Phase-2\",\"top_k\":5,\"use_llm\":true,\"use_vector\":true}"`

Agentic text + image evidence:
`curl -X POST "http://localhost:8000/v1/agentic/query" -H "Content-Type: application/json" -d "{\"query\":\"explain figure on page 12\",\"top_k\":5,\"use_llm\":true,\"use_vector\":true,\"include_images\":true,\"response_mode\":\"full\"}"`

Precise response (default):
- Returns `snippet` (query-focused excerpt), not full noisy chunk text.
- Returns `matched_terms` to show why it matched.
- To include full chunk text only when needed, set `"include_full_text": true`.
- `response_mode` options:
  - `compact`: shorter snippets
  - `balanced`: richer snippets with key data preserved
  - `full`: include full chunk text in `text`
- `use_vector`: enable semantic vector retrieval
- `include_images`: include multimodal image collection hits (requires `ENABLE_MULTIMODAL_INGEST=true`)
- `vector_top_k`: number of vector candidates to pull from Qdrant
- Final ranking is reranked hybrid score from BM25 + vector + coverage/proximity signals
- Agentic endpoint includes:
  - input guardrails (prompt injection/jailbreak/PII)
  - retrieval guardrails (blocked sources/conflict warnings)
  - planner, synthesizer, critic with bounded self-correction
  - default output mode is `compact` (human-friendly single answer + citations)
  - use `"response_mode":"full"` to include evidence/steps/guardrail details
- Noise control knobs (env):
  - `SEARCH_MIN_SCORE`
  - `SEARCH_RELATIVE_SCORE_RATIO`
  - `SEARCH_MIN_KEYWORD_COVERAGE`
  - `LLM_MAX_CONTEXT_HITS`

Layout/OCR controls:
- `MIN_PAGE_TEXT_CHARS`: pages below this text threshold are treated as likely scanned.
- `ENABLE_OCR_FALLBACK`: when `true`, pipeline attempts OCR fallback per low-text page.
- OCR dependencies are optional and only needed when OCR fallback is enabled (`pymupdf`, `pillow`, `pytesseract`, plus local Tesseract binary).

LLM answer synthesis at end of `/v1/query`:
- Enabled by default with `"use_llm": true`.
- Uses retrieved hits as grounding context.
- Adds `answer`, `answer_status`, and `answer_model` in response.
- Default model config is `gpt-4o-mini` via `OPENAI_BASE_URL=https://api.openai.com/v1`.
- `OPENAI_API_KEY` is required for OpenAI-hosted APIs.

Vector setup (local):
- Run Qdrant locally on port `6333`.
- Ensure embedding endpoint is available at `${OPENAI_BASE_URL}/embeddings` for `EMBEDDING_MODEL` (default `text-embedding-3-small`).
- Ingestion writes `vector_index_summary` into each document manifest.
- Hybrid knobs:
  - `BM25_K1`, `BM25_B`
  - `VECTOR_QUERY_LIMIT`

Multimodal image retrieval (optional):
- `ENABLE_MULTIMODAL_INGEST=true` enables image extraction/indexing.
- `ENABLE_CLIP_IMAGE_VECTORS=true` adds CLIP image vectors in `MULTIMODAL_CLIP_QDRANT_COLLECTION`.
- `ENABLE_VLM_CAPTIONS=true` generates optional image captions via OpenAI Responses API.
- Keep all three flags `false` to preserve default text-only ingestion behavior.

## Next upgrades
- Add OCR fallback for scanned PDFs
- Add heading/table detection and metadata enrichment by section
- Push chunks to vector DB + BM25 index
- Add reranker in retrieval flow

## Evaluation (RAGAS + DeepEval)
Install eval dependencies:
`pip install -r requirements-eval.txt`

1. Build eval records from gold set:
`python scripts/eval_build_records.py --gold-file datasets/gold_qa_set_150.jsonl --output-file datasets/eval_records.jsonl --query-url http://localhost:8000/v1/query --top-k 5 --use-vector`

2. Run RAGAS analytics:
`python scripts/eval_ragas.py --records-file datasets/eval_records.jsonl --output-file datasets/ragas_report.json --llm-model qwen2.5:7b --embedding-model nomic-embed-text --base-url http://localhost:11434/v1`

3. Run DeepEval regression + CI gate:
`python scripts/eval_deepeval.py --records-file datasets/eval_records.jsonl --output-file datasets/deepeval_report.json --min-faithfulness 0.80 --min-answer-relevancy 0.75 --min-context-precision 0.70 --min-context-recall 0.70 --fail-on-gate`

Recommended usage:
- Use `RAGAS` reports for tuning loops and model/retriever diagnostics.
- Use `DeepEval` with `--fail-on-gate` in CI before deployment.
