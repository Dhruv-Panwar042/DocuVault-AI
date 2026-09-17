import os
import time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import settings
from app.api.v1.endpoints import router as v1_router
from app.services.rag_service import rag_service

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Decoupled enterprise RAG microservice with local all-MiniLM-L6-v2 vector embeddings and resilient Gemini reasoning.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware to support standalone web frontends or decoupled microfrontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(process_time)
    return response


# Include API router
app.include_router(v1_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
def health():
    """Global root health check for container liveness probe."""
    return {
        "status": "healthy",
        "service": "DocuVault AI",

        "version": "2.0.0",
        "is_indexed": rag_service.is_indexed,
        "embedding_model": settings.EMBEDDING_MODEL,
    }


# Path resolution for frontend assets
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", tags=["UI"])
def serve_index():
    """Serves the DocuVault AI modern light dashboard."""
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    return JSONResponse(
        content={
            "service": "DocuVault AI API",

            "message": "Frontend index.html is being prepared.",
            "docs": "/docs",
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
