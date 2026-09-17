from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    document_name: str = Field(..., description="Source PDF filename")
    page_number: int = Field(..., description="1-indexed document page number")
    snippet: str = Field(..., description="Retrieved excerpt text")
    distance_score: float = Field(..., description="Raw L2 similarity distance from FAISS")
    relevance_score: float = Field(..., description="Normalized confidence rating between 0 and 1")


class ChunkItem(BaseModel):
    chunk_id: int = Field(..., description="Sequential index of the chunk")
    document_name: str = Field(..., description="Parent PDF filename")
    page_number: int = Field(..., description="1-indexed document page number")
    char_count: int = Field(..., description="Character length of chunk")
    snippet: str = Field(..., description="Text preview of chunk")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="User question about uploaded documents")
    chat_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Previous conversation turns")
    top_k: int = Field(default=3, ge=1, le=8, description="Number of vector chunks to retrieve")


class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence_score: float
    confidence_label: str
    sources: List[SourceCitation]
    latency_ms: float


class UploadResponse(BaseModel):
    total_documents: int
    total_pages: int
    total_chunks: int
    filenames: List[str]
    chunks_sample: List[ChunkItem]
    suggested_prompts: List[str]
    process_time_ms: float


class SummaryResponse(BaseModel):
    summary: str
    document_count: int
    total_pages: int
    latency_ms: float


class IndexStatusResponse(BaseModel):
    is_indexed: bool
    document_count: int
    documents: List[Dict[str, Any]]
    total_pages: int
    total_chunks: int
    embedding_model: str


class ExportPdfRequest(BaseModel):
    summary: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None


class ChunksResponse(BaseModel):
    total_chunks: int
    chunks: List[ChunkItem]
    limit: int
    offset: int

