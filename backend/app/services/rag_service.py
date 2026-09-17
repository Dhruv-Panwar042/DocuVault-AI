import io
import time
from typing import List, Dict, Tuple, Optional, Any
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

from app.core.config import settings
from app.schemas.rag import SourceCitation, ChunkItem, UploadResponse, IndexStatusResponse


class RAGService:
    _instance: Optional["RAGService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance._init_service()
        return cls._instance

    def _init_service(self):
        self._embeddings: Optional[FastEmbedEmbeddings] = None
        self._vectorstore: Optional[FAISS] = None
        self._chunks: List[Document] = []
        self._documents_meta: List[Dict[str, Any]] = []

    @property
    def embeddings(self) -> FastEmbedEmbeddings:
        if self._embeddings is None:
            print(f"Loading local ONNX embedding model: {settings.EMBEDDING_MODEL}...")
            self._embeddings = FastEmbedEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                threads=1,
            )
        return self._embeddings



    @property
    def is_indexed(self) -> bool:
        return self._vectorstore is not None and len(self._chunks) > 0

    def get_status(self) -> IndexStatusResponse:
        total_pages = sum(d.get("pages", 0) for d in self._documents_meta)
        return IndexStatusResponse(
            is_indexed=self.is_indexed,
            document_count=len(self._documents_meta),
            documents=self._documents_meta,
            total_pages=total_pages,
            total_chunks=len(self._chunks),
            embedding_model=settings.EMBEDDING_MODEL,
        )

    def clear(self):
        """Clears all indexed vectors and memory."""
        self._vectorstore = None
        self._chunks = []
        self._documents_meta = []

    def process_pdfs(self, file_tuples: List[Tuple[str, bytes]]) -> UploadResponse:
        """
        Parses PDF bytes directly in-memory via pypdf, chunks text with RecursiveCharacterTextSplitter,
        and constructs an in-memory FAISS vectorstore.
        file_tuples: list of (filename, file_bytes)
        """
        start_time = time.perf_counter()
        all_docs: List[Document] = []
        meta_list: List[Dict[str, Any]] = []
        total_pages = 0

        for filename, file_bytes in file_tuples:
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                num_pages = len(reader.pages)
                total_pages += num_pages
                file_doc_count = 0

                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    clean_text = text.strip()
                    if clean_text:
                        all_docs.append(
                            Document(
                                page_content=clean_text,
                                metadata={
                                    "source": filename,
                                    "page": page_idx + 1,  # 1-indexed for human readability
                                    "total_doc_pages": num_pages,
                                },
                            )
                        )
                        file_doc_count += 1

                meta_list.append({
                    "filename": filename,
                    "pages": num_pages,
                    "extracted_pages": file_doc_count,
                    "size_bytes": len(file_bytes),
                })
            except Exception as e:
                print(f"Error parsing PDF '{filename}': {e}")
                continue

        if not all_docs:
            raise ValueError("No extractable text found in uploaded PDF(s).")

        # Recursive semantic chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(all_docs)

        # Build in-memory FAISS index
        vectorstore = FAISS.from_documents(chunks, self.embeddings)

        self._vectorstore = vectorstore
        self._chunks = chunks
        self._documents_meta = meta_list

        # Extract sample chunks for inspector
        chunks_sample = [
            ChunkItem(
                chunk_id=idx + 1,
                document_name=chunk.metadata.get("source", "Document"),
                page_number=chunk.metadata.get("page", 1),
                char_count=len(chunk.page_content),
                snippet=chunk.page_content[:280] + ("..." if len(chunk.page_content) > 280 else ""),
            )
            for idx, chunk in enumerate(chunks[:8])
        ]

        # Generate grounded suggested starter questions based on initial content
        suggested_prompts = self._generate_starter_prompts(all_docs)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return UploadResponse(
            total_documents=len(meta_list),
            total_pages=total_pages,
            total_chunks=len(chunks),
            filenames=[m["filename"] for m in meta_list],
            chunks_sample=chunks_sample,
            suggested_prompts=suggested_prompts,
            process_time_ms=elapsed_ms,
        )

    def _generate_starter_prompts(self, docs: List[Document]) -> List[str]:
        """Generates dynamic starter prompts based on document text."""
        first_few = " ".join([d.page_content[:200] for d in docs[:3]]).lower()
        prompts = [
            "Provide a concise executive overview of the uploaded document(s).",
            "What are the primary findings, methodologies, or objectives discussed?",
        ]
        if "result" in first_few or "conclusion" in first_few or "data" in first_few:
            prompts.append("Summarize the key data, outcomes, and conclusions presented.")
        else:
            prompts.append("List the core topics and critical takeaways from this text.")
        return prompts[:3]

    def similarity_search(self, query: str, top_k: int = 3) -> Tuple[List[Document], List[SourceCitation], float, str]:
        """
        Runs dense similarity search against FAISS index.
        Returns (docs, citations, top_score, confidence_label).
        """
        if not self._vectorstore:
            raise ValueError("No documents have been indexed yet. Please upload a PDF first.")

        scored_docs = self._vectorstore.similarity_search_with_score(query, k=top_k)
        if not scored_docs:
            return [], [], 999.0, "Low Confidence"


        citations: List[SourceCitation] = []
        raw_scores = []

        for doc, score in scored_docs:
            raw_scores.append(score)
            rel_score = round(max(0.0, min(1.0, 1.0 - (score / 2.0))), 3)
            citations.append(
                SourceCitation(
                    document_name=doc.metadata.get("source", "Document"),
                    page_number=doc.metadata.get("page", 1),
                    snippet=doc.page_content[:350] + ("..." if len(doc.page_content) > 350 else ""),
                    distance_score=round(float(score), 4),
                    relevance_score=rel_score,
                )
            )

        top_score = raw_scores[0]
        if top_score <= 0.5:
            conf_label = "High Confidence"

        elif top_score <= 1.0:
            conf_label = "Medium Confidence"
        else:
            conf_label = "Low Confidence"


        docs = [doc for doc, _ in scored_docs]
        return docs, citations, round(float(top_score), 4), conf_label

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    def get_all_chunks(self, limit: int = 50, offset: int = 0) -> List[ChunkItem]:
        """Returns paginated chunk items for the Chunk Inspector UI."""
        return [
            ChunkItem(
                chunk_id=offset + idx + 1,
                document_name=chunk.metadata.get("source", "Document"),
                page_number=chunk.metadata.get("page", 1),
                char_count=len(chunk.page_content),
                snippet=chunk.page_content,
            )
            for idx, chunk in enumerate(self._chunks[offset : offset + limit])
        ]

    def get_summary_context(self, max_chars: int = 5000) -> Tuple[str, List[str]]:

        """Samples initial sections of each indexed document to form a representative summarization context."""
        if not self._chunks:
            return "", []

        doc_names = [d.get("filename", "Document") for d in self._documents_meta]
        selected_text = []
        current_chars = 0

        # Sample chunks evenly or from top chunks
        for chunk in self._chunks:
            content = f"[{chunk.metadata.get('source', 'Doc')} - p.{chunk.metadata.get('page', 1)}]:\n{chunk.page_content}"
            if current_chars + len(content) > max_chars:
                break
            selected_text.append(content)
            current_chars += len(content)

        return "\n\n---\n\n".join(selected_text), doc_names


rag_service = RAGService()

