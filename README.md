# RAG-based Multi-Document Q&A Chatbot

A document chatbot that lets you upload one or more PDFs and ask questions about them. 
Built with LangChain, Google Gemini, HuggingFace embeddings, FAISS, and Streamlit.

---

## Why I built this

Most LLMs hallucinate when asked about specific documents. This project solves that by 
grounding every answer in the actual content of the uploaded PDFs — the model is 
explicitly instructed to say "I don't have enough information" if the answer isn't 
in the document. This makes it reliable for real use cases like research papers, 
textbooks, and technical documentation.

---

## Screenshots

![Upload and Process](screenshot1.png)
![Chat with Sources](screenshot2.png)
![Confidence Score](screenshot3.png)
![PDF Summary](screenshot4.png)

---

## Features

- **Multi-PDF support** — upload multiple PDFs at once and query across all of them
- **Source citation** — every answer shows the exact page number it came from
- **Confidence scoring** — color-coded labels (High / Medium / Low) based on FAISS similarity distances
- **Chat history** — follow-up questions are handled correctly because the full conversation is passed as context
- **PDF summarization** — one-click summary of all uploaded documents
- **No embedding API costs** — embeddings run locally using HuggingFace sentence-transformers; only Gemini is called for answer generation

---

## How it works

1. Uploaded PDFs are split into overlapping chunks (1000 chars, 200 overlap)
2. Each chunk is converted to a vector using `sentence-transformers/all-MiniLM-L6-v2` locally
3. Vectors are stored in a FAISS index for fast similarity search
4. When a question is asked, the top 3 most relevant chunks are retrieved
5. Those chunks + the conversation history are sent to Gemini as context
6. Gemini generates an answer strictly from that context

---

## Tech stack

| Tool | Purpose |
|---|---|
| LangChain | RAG pipeline and document loading |
| Google Gemini API | Answer generation |
| HuggingFace sentence-transformers | Local embedding generation |
| FAISS | Vector storage and similarity search |
| Streamlit | Web interface |
| PyPDF | PDF parsing |

---

## What I learned

- How RAG works in practice — chunking strategy and overlap size significantly affect answer quality
- Why local embeddings are a better choice for projects with limited API quota
- How to pass conversation history to an LLM for coherent multi-turn conversations
- How FAISS similarity scores can be used as a proxy for answer confidence

---

## Author

**Dhruv Panwar**   
[GitHub](https://github.com/Dhruv-Panwar042)  