from typing import Optional, List, Dict
import google.genai as genai
from app.core.config import settings


class GeminiRAGService:
    def __init__(self):
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not settings.GOOGLE_API_KEY:
                raise ValueError(
                    "GOOGLE_API_KEY is not configured. Please set GOOGLE_API_KEY in your .env or cloud environment."
                )
            self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        return self._client

    def _generate_with_failover(self, prompt: str) -> str:
        """Executes LLM call with transparent multi-model failover against 429 quota exhaustion."""
        candidate_models = [settings.GEMINI_MODEL, "gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash"]
        candidate_models = list(dict.fromkeys(candidate_models))

        last_error = None
        for model_name in candidate_models:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response.text:
                    return response.text
            except Exception as e:
                err_str = str(e)
                last_error = e
                # Intercept rate limit (429) or high demand (503) and cascade to next model
                if any(code in err_str for code in ["429", "RESOURCE_EXHAUSTED", "503", "404", "quota"]):
                    print(f"Gemini model '{model_name}' hit limit, falling back to next candidate...")
                    continue
                raise e

        raise last_error or RuntimeError("All Gemini model candidates exhausted.")

    def answer_query(self, question: str, context: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generates grounded answers based strictly on retrieved vector context and conversation turns."""
        history_lines = []
        if chat_history:
            for turn in chat_history[-4:]:  # Keep recent 4 turns to avoid context clutter
                role = "User" if turn.get("role") == "user" else "Assistant"
                history_lines.append(f"{role}: {turn.get('content', '')}")
        history_str = "\n".join(history_lines) if history_lines else "None"

        prompt = f"""You are DocuVault AI, an intelligent, objective multi-document analysis assistant.


Instructions:
1. Answer the user's question accurately using ONLY the provided Context and Conversation History.
2. If the answer cannot be directly determined from the provided Context, state:
   "I do not have sufficient information within the uploaded documents to answer this accurately."
3. Keep the tone professional, objective, and structured. Use bullet points where appropriate.

Conversation History:
{history_str}

Retrieved Context Excerpts:
{context}

Question:
{question}

Answer:"""
        return self._generate_with_failover(prompt)

    def summarize_documents(self, context_sample: str, doc_names: List[str]) -> str:
        """Generates a structured multi-document executive summary."""
        doc_list_str = ", ".join(doc_names)
        prompt = f"""You are an executive document analyst.

Synthesize the provided document excerpts from ({doc_list_str}) into a clear, structured executive briefing.

Structure your response with markdown headers:
### 📌 Executive Overview
(2-3 sentences explaining the overarching objective, scope, and premise of the text)

### 🔑 Core Themes & Methodologies
(Key concepts, procedures, or architectural points detailed in the content)

### 📊 Major Findings & Data Points
(Concrete metrics, results, conclusions, or critical data discussed)

### 💡 Strategic Takeaways
(High-level summary recommendations or notable conclusions)

Document Context:
{context_sample}

Executive Briefing:"""
        return self._generate_with_failover(prompt)


gemini_rag_service = GeminiRAGService()
