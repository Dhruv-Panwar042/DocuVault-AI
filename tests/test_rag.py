import io
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure backend package can be found
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))

from app.main import app
from app.services.rag_service import rag_service
from app.services.report_service import report_service


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoints(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "DocuMind AI"

    res_v1 = client.get("/api/v1/health")
    assert res_v1.status_code == 200
    assert res_v1.json()["status"] == "healthy"


def test_frontend_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "DocuMind AI" in res.text
    assert "Vector Chunk Inspector" in res.text



def test_status_empty_initial(client):
    rag_service.clear()
    res = client.get("/api/v1/status")
    assert res.status_code == 200
    data = res.json()
    assert data["is_indexed"] is False
    assert data["document_count"] == 0
    assert data["total_chunks"] == 0


def test_upload_non_pdf_rejection(client):
    fake_txt = io.BytesIO(b"Hello world, this is a plain text file.")
    files = [("files", ("notes.txt", fake_txt, "text/plain"))]
    res = client.post("/api/v1/upload", files=files)
    assert res.status_code == 400
    assert "not a PDF" in res.json()["detail"]


def test_query_without_indexing_returns_400(client):
    rag_service.clear()
    res = client.post("/api/v1/query", json={"question": "What is in this document?"})
    assert res.status_code == 400
    assert "No documents have been indexed yet" in res.json()["detail"]


def test_summarize_without_indexing_returns_400(client):
    rag_service.clear()
    res = client.post("/api/v1/summarize")
    assert res.status_code == 400
    assert "No documents indexed" in res.json()["detail"]


def test_end_to_end_indexing_and_chunks(client):
    sample_pdf_path = BASE_DIR / "sample.pdf"
    if not sample_pdf_path.exists():
        pytest.skip("sample.pdf not found in workspace")

    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    files = [("files", ("sample.pdf", io.BytesIO(pdf_bytes), "application/pdf"))]
    res = client.post("/api/v1/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["total_documents"] >= 1
    assert data["total_chunks"] > 0
    assert len(data["suggested_prompts"]) > 0

    # Verify status reflects indexed state
    status_res = client.get("/api/v1/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["is_indexed"] is True
    assert status_data["total_chunks"] == data["total_chunks"]

    # Verify chunk inspector pagination
    chunks_res = client.get("/api/v1/chunks?limit=5&offset=0")
    assert chunks_res.status_code == 200
    chunks_data = chunks_res.json()
    assert chunks_data["total_chunks"] == data["total_chunks"]
    assert len(chunks_data["chunks"]) <= 5
    first_chunk = chunks_data["chunks"][0]
    assert "snippet" in first_chunk
    assert first_chunk["document_name"] == "sample.pdf"

    # Test PDF ReportLab generation
    export_res = client.post(
        "/api/v1/export/pdf",
        json={
            "summary": "This is a test executive briefing for verification.",
            "chat_history": [
                {
                    "question": "What does this document describe?",
                    "answer": "The document provides an overview of automated intelligence systems.",
                    "confidence_label": "🟢 High Confidence",
                    "sources": [{"document_name": "sample.pdf", "page_number": 1}],
                }
            ],
        },
    )
    assert export_res.status_code == 200
    assert export_res.headers["content-type"] == "application/pdf"
    # PDF magic byte signature
    assert export_res.content[:4] == b"%PDF"

    # Clean up index
    clear_res = client.delete("/api/v1/clear")
    assert clear_res.status_code == 200
    assert client.get("/api/v1/status").json()["is_indexed"] is False
