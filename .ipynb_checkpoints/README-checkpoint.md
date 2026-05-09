# 🤖 RAG Chatbot using LangChain & Google Gemini

An AI-powered chatbot that answers questions from any PDF document using 
**Retrieval Augmented Generation (RAG)** — built with LangChain, Google Gemini, 
FAISS, and Streamlit.

---

## 📌 Project Overview

This project builds a document Q&A chatbot that:
- Accepts any PDF as input
- Splits it into chunks and converts them to embeddings
- Stores embeddings in a FAISS vector database
- Retrieves relevant chunks based on the user's question
- Uses Google Gemini to generate accurate, context-aware answers

---

## 🖥️ App Screenshots

![Screenshot 1](screenshot1.png)
![Screenshot 2](screenshot2.png)
![Screenshot 3](screenshot3.png)

---

## 🧠 What I Built

- Loaded and split PDFs into overlapping chunks using LangChain
- Generated embeddings using **Google Gemini Embedding Model**
- Stored and searched embeddings using **FAISS vector database**
- Built a RAG chain with **hallucination prevention** — the model only answers from the document context
- Deployed an interactive chat interface using **Streamlit**

---

## 📂 Project Structure

rag-chatbot-gemini/
│
├── RAG_Chatbot.ipynb   # Full notebook walkthrough
├── app.py              # Streamlit web app
├── requirements.txt    # Dependencies
└── README.md           # Project documentation

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| LangChain | RAG pipeline framework |
| Google Gemini API | Embeddings + LLM |
| FAISS | Vector database for semantic search |
| Streamlit | Interactive web app |
| PyPDF | PDF loading and parsing |

---

## 📊 How RAG Works

User Question
↓
Search FAISS for relevant chunks
↓
Send chunks + question to Gemini
↓
Gemini generates answer from context only
↓
Answer displayed to user


## 👤 Author

**Dhruv Panwar**  
[GitHub](https://github.com/Dhruv-Panwar042)