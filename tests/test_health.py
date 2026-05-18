from fastapi.testclient import TestClient
import json

from app.core.config import settings
from app.main import app


def test_healthcheck() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_without_documents_returns_empty_hits(tmp_path) -> None:
    processed_dir = tmp_path / "processed-empty"
    processed_dir.mkdir(parents=True, exist_ok=True)

    original_dir = settings.processed_dir
    settings.processed_dir = processed_dir
    try:
        client = TestClient(app)
        response = client.post("/v1/query", json={"query": "RBI policy", "top_k": 3, "use_llm": False, "use_vector": False})
        assert response.status_code == 200
        assert response.json()["hits"] == []
    finally:
        settings.processed_dir = original_dir


def test_query_returns_snippet_without_full_text_by_default(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "doc_id": "doc-1",
        "source_file": "sample.pdf",
        "chunks": [
            {
                "chunk_id": "chunk_00001",
                "text": "Sahitya Akademi Yuva Puraskar 2025 winners include notable authors across languages. "
                "Assamese winner is Supraksam Bhuyan and Hindi winner is Parvati Tirkey.",
                "page_start": 10,
                "page_end": 10,
                "metadata": {"months": [], "topics": ["awards"], "entities": ["GA"]},
            }
        ],
    }
    (processed_dir / "doc-1.json").write_text(json.dumps(manifest), encoding="utf-8")

    original_dir = settings.processed_dir
    settings.processed_dir = processed_dir

    try:
        client = TestClient(app)
        response = client.post(
            "/v1/query",
            json={
                "query": "list of Sahitya Akademi Yuva Puraskar 2025 winners",
                "top_k": 3,
                "use_llm": False,
                "use_vector": False,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["hits"]) == 1
        assert "snippet" in payload["hits"][0]
        assert "text" not in payload["hits"][0]
        assert len(payload["hits"][0]["snippet"]) <= 2200
    finally:
        settings.processed_dir = original_dir


def test_query_filters_low_relevance_noise(tmp_path) -> None:
    processed_dir = tmp_path / "processed-noise"
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "doc_id": "doc-2",
        "source_file": "sample2.pdf",
        "chunks": [
            {
                "chunk_id": "chunk_good",
                "text": "Pune Metro Rail Phase-2 route includes Vanaz to Chandani Chowk corridor details.",
                "page_start": 20,
                "page_end": 20,
                "metadata": {"months": [], "topics": ["government_schemes"], "entities": ["PUNE"]},
            },
            {
                "chunk_id": "chunk_noise",
                "text": "Award ceremony phase announcement for sports event took place this week.",
                "page_start": 21,
                "page_end": 21,
                "metadata": {"months": [], "topics": ["awards"], "entities": ["GA"]},
            },
        ],
    }
    (processed_dir / "doc-2.json").write_text(json.dumps(manifest), encoding="utf-8")

    original_dir = settings.processed_dir
    settings.processed_dir = processed_dir
    try:
        client = TestClient(app)
        response = client.post(
            "/v1/query",
            json={
                "query": "What is route of Pune Metro Rail Phase-2",
                "top_k": 2,
                "use_llm": False,
                "use_vector": False,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["hits"]) == 1
        assert payload["hits"][0]["chunk_id"] == "chunk_good"
    finally:
        settings.processed_dir = original_dir


def test_hybrid_bm25_reranker_prioritizes_exact_route_chunk(tmp_path) -> None:
    processed_dir = tmp_path / "processed-hybrid"
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "doc_id": "doc-hybrid",
        "source_file": "metro.pdf",
        "chunks": [
            {
                "chunk_id": "chunk_route_exact",
                "text": "Pune Metro Rail Phase-2 route runs from Vanaz to Chandani Chowk with key interchange points.",
                "page_start": 12,
                "page_end": 12,
                "metadata": {"months": [], "topics": ["government_schemes"], "entities": ["PUNE"]},
            },
            {
                "chunk_id": "chunk_partial",
                "text": "Pune urban transport update mentions metro expansion and city infrastructure announcements.",
                "page_start": 13,
                "page_end": 13,
                "metadata": {"months": [], "topics": ["government_schemes"], "entities": ["PUNE"]},
            },
        ],
    }
    (processed_dir / "doc-hybrid.json").write_text(json.dumps(manifest), encoding="utf-8")

    original_dir = settings.processed_dir
    settings.processed_dir = processed_dir
    try:
        client = TestClient(app)
        response = client.post(
            "/v1/query",
            json={
                "query": "What is route of Pune Metro Rail Phase-2",
                "top_k": 2,
                "use_llm": False,
                "use_vector": False,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["hits"]) == 2
        assert payload["hits"][0]["chunk_id"] == "chunk_route_exact"
        assert payload["hits"][0]["score"] >= payload["hits"][1]["score"]
    finally:
        settings.processed_dir = original_dir
