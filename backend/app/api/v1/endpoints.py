import time
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Response, status

from app.core.config import settings
from app.schemas.rag import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    SummaryResponse,
    IndexStatusResponse,
    ExportPdfRequest,
    ChunksResponse,
)
from app.services.rag_service import rag_service
from app.services.gemini_service import gemini_rag_service
from app.services.report_service import report_service

router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check():
    """Liveness probe returning service and index state."""
    return {
        "status": "healthy",
        "service": "DocuVault AI RAG Engine",

        "version": "2.0.0",
        "is_indexed": rag_service.is_indexed,
        "embedding_model": settings.EMBEDDING_MODEL,
        "active_gemini_model": settings.GEMINI_MODEL,
    }


@router.get("/status", response_model=IndexStatusResponse, tags=["Document Management"])
def get_index_status():
    """Returns metadata about currently indexed documents and vector counts."""
    return rag_service.get_status()


@router.post("/upload", response_model=UploadResponse, tags=["Document Management"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or multiple PDF documents.
    Extracts text in-memory, performs recursive semantic chunking, generates local embeddings,
    and indexes chunks into FAISS vector space.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    file_tuples = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a PDF. Only PDF files are supported.",
            )
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file '{file.filename}' is empty (0 bytes).",
            )
        file_tuples.append((file.filename, content))

    try:
        response = rag_service.process_pdfs(file_tuples)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process and index PDF(s): {str(e)}",
        )


@router.post("/query", response_model=QueryResponse, tags=["RAG Reasoning"])
def execute_query(req: QueryRequest):
    """
    Executes dense similarity search against the indexed vector embeddings,
    retrieves top-k context passages with confidence scoring, and generates a grounded response.
    """
    if not rag_service.is_indexed:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload one or more PDFs first.",
        )

    start_time = time.perf_counter()
    try:
        docs, citations, top_score, conf_label = rag_service.similarity_search(
            query=req.question,
            top_k=req.top_k,
        )

        context_chunks = [
            f"[Source: {doc.metadata.get('source', 'Doc')} (Page {doc.metadata.get('page', 1)})]:\n{doc.page_content}"
            for doc in docs
        ]
        context_str = "\n\n---\n\n".join(context_chunks)

        answer = gemini_rag_service.answer_query(
            question=req.question,
            context=context_str,
            chat_history=req.chat_history,
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return QueryResponse(
            question=req.question,
            answer=answer,
            confidence_score=top_score,
            confidence_label=conf_label,
            sources=citations,
            latency_ms=elapsed_ms,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing grounded query: {str(e)}",
        )


@router.post("/summarize", response_model=SummaryResponse, tags=["RAG Reasoning"])
def generate_summary():
    """
    Synthesizes an executive briefing across all indexed documents.
    """
    if not rag_service.is_indexed:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Please upload PDF(s) before generating a summary.",
        )

    start_time = time.perf_counter()
    try:
        context_sample, doc_names = rag_service.get_summary_context(max_chars=6000)
        summary = gemini_rag_service.summarize_documents(
            context_sample=context_sample,
            doc_names=doc_names,
        )

        total_pages = sum(d.get("pages", 0) for d in rag_service._documents_meta)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return SummaryResponse(
            summary=summary,
            document_count=len(doc_names),
            total_pages=total_pages,
            latency_ms=elapsed_ms,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating executive briefing: {str(e)}",
        )


@router.get("/chunks", response_model=ChunksResponse, tags=["Vector Inspector"])
def inspect_chunks(
    limit: int = Query(default=50, ge=1, le=200, description="Max number of chunks to fetch"),
    offset: int = Query(default=0, ge=0, description="Chunk starting offset"),
):
    """
    Returns paginated raw vector chunks with parent document name, page number, and snippet.
    Used by the interactive Vector Chunk Inspector UI.
    """
    chunks = rag_service.get_all_chunks(limit=limit, offset=offset)
    return ChunksResponse(
        total_chunks=rag_service.total_chunks,
        chunks=chunks,
        limit=limit,
        offset=offset,
    )


@router.delete("/clear", tags=["Document Management"])
def clear_index():
    """
    Clears all vectors, cached chunks, and metadata from RAM.
    """
    rag_service.clear()
    return {
        "status": "cleared",
        "message": "Indexed documents and vector storage cleared successfully.",
    }


@router.post("/export/pdf", tags=["Export"])
def export_executive_pdf(req: ExportPdfRequest):
    """
    Generates and streams an executive PDF intelligence brief compiling corpus stats,
    executive synthesis, and interactive query history.
    """
    try:
        pdf_bytes = report_service.generate_executive_pdf(
            documents_meta=rag_service._documents_meta,
            summary_text=req.summary,
            qa_history=req.chat_history,
            embedding_model=settings.EMBEDDING_MODEL,
            total_chunks=rag_service.total_chunks,
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=DocuVault_Intelligence_Brief.pdf",

                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate executive PDF: {str(e)}",
        )
