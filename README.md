# PDF Ingestion API (FastAPI)

Starter implementation for ingesting exam-style PDF documents.

## What this includes
- `POST /v1/ingest/pdf`: upload PDF and ingest text chunks
- `POST /v1/query`: retrieve top matching chunks from Qdrant
- `POST /v1/agentic/query`: planner/retriever/synthesizer/critic orchestration with guardrails
- `GET /health`: health check
- PDF extraction with `pypdf`
- Layout-aware block parsing (heading/list/table/paragraph)
- Chunk generation with page-aware metadata
- Automatic metadata enrichment (months/topics/entities)
- True hybrid retrieval: BM25 + vector + reranker
- Embeddings + local Qdrant vector indexing (during ingestion)
- Qdrant is the single retrieval source for ingested chunks

## Quick start
1. Create virtual env and activate it.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Copy env template:
   `copy .env.example .env`
4. Run server:
   `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

5. (Async ingest) Run worker:
   `python -m app.workers.async_ingestion_worker --log-level INFO`

Async worker requirements:
- `ENABLE_ASYNC_INGESTION=true`
- `ENABLE_S3_UPLOAD=true`
- `AWS_REGION`, `S3_BUCKET_NAME`, `SQS_QUEUE_URL`, `DYNAMODB_JOBS_TABLE` configured

Optional fully automatic mode (no manual worker run):
- set `AUTO_START_ASYNC_WORKER=true`
- API process starts embedded SQS worker on startup

Optional end-to-end blocking upload response:
- set `ASYNC_WAIT_FOR_COMPLETION_DEFAULT=true` (default)
- or call `POST /v1/ingest/pdf?pipeline=hf&wait_for_completion=true`
- request waits for queued job completion and returns final ingestion response when ready

## AWS worker deployment (ECS/Fargate)
1. Create ECR repository (example: `rag-ingestion-worker`).
2. Build image from repo root:
   `docker build -f worker/Dockerfile -t rag-ingestion-worker:latest .`
3. Login to ECR:
   `aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com`
4. Tag and push image:
   `docker tag rag-ingestion-worker:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/rag-ingestion-worker:latest`
   `docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/rag-ingestion-worker:latest`
5. Create/attach task role policy from `infra/iam-task-role-policy.json` (replace placeholders).
6. Register task definition from `infra/ecs-task-definition.json` (replace placeholders).
7. Run ECS service on Fargate with desired count `1+`.

Result:
- `POST /v1/ingest/pdf?pipeline=hf` uploads to S3, enqueues SQS job.
- ECS worker consumes SQS, processes ingestion, updates DynamoDB, writes to Qdrant.
- Data becomes queryable through `POST /v1/agentic/query`.

## AWS API deployment (ECS/Fargate + ALB)
1. Create ECR repository (example: `rag-api-service`).
2. Build API image from repo root:
   `docker build -f api.Dockerfile -t rag-api-service:latest .`
   - `api.Dockerfile` installs from `requirements.api.txt` (lean runtime deps for smaller image size).
3. Login/tag/push:
   `aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com`
   `docker tag rag-api-service:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/rag-api-service:latest`
   `docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/rag-api-service:latest`
4. Create API task role and attach policy from `infra/iam-api-task-role-policy.json`.
5. Register task definition from `infra/ecs-api-task-definition.json`.
6. Create ALB target group for HTTP port `8000`.
7. Create ECS service using `infra/ecs-api-service.json` (replace placeholders for cluster/subnets/SG/target group).
8. Create ALB listener rule to forward API traffic to that target group.

After this, replace localhost endpoints with ALB DNS:
- `POST https://<ALB_DNS>/v1/ingest/pdf?pipeline=hf`
- `POST https://<ALB_DNS>/v1/agentic/query`

Recommended with cloud worker:
- API task env: `AUTO_START_ASYNC_WORKER=false`
- Worker remains separate ECS service (`rag-ingestion-worker`)

### Worker image size optimization
- The worker container intentionally excludes heavy local-ML packages (`sentence-transformers`, `torch`) to keep image size low.
- This is safe when embeddings/LLM calls are remote (OpenAI/Bedrock/SageMaker).
- If you require local HF/torch inference inside ECS worker, use a separate heavyweight image profile.

## Example API calls
Ingest a PDF:
`curl -X POST "http://localhost:8000/v1/ingest/pdf" -F "file=@C:/path/doc.pdf"`

Ingest with HF pipeline using async S3/SQS path:
`curl -X POST "http://localhost:8000/v1/ingest/pdf?pipeline=hf" -F "file=@C:/path/doc.pdf"`

Force async explicitly (any pipeline):
`curl -X POST "http://localhost:8000/v1/ingest/pdf?pipeline=hf&async_mode=true" -F "file=@C:/path/doc.pdf"`

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
- Ingestion writes `vector_index_summary` in API response and indexes chunks into Qdrant.
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
