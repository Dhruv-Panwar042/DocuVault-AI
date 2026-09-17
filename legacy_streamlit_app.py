import os
import hashlib
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local embeddings (runs entirely on your machine)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Gemini is used only for answer generation
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ── Get API key ───────────────────────────────────────────────
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.write("Upload one or more PDFs and ask questions about them!")
st.divider()


# ── Initialize Models ─────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)


# ── Helper Functions ──────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(messages):
    history = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history.append(f"{role}: {msg['content']}")
    return "\n".join(history)


def get_confidence_label(score):
    if score <= 0.5:
        return "🟢 High Confidence"
    elif score <= 1.0:
        return "🟡 Medium Confidence"
    else:
        return "🔴 Low Confidence"


def get_files_signature(uploaded_files):
    signature_parts = []
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        signature_parts.append(f"{uploaded_file.name}_{file_hash}")
    return "_".join(signature_parts)


def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about the uploaded PDFs.

Use the conversation history and the retrieved context to answer the user's question.

If the answer is not found in the context, say:
"I don't have enough information to answer this."

Conversation History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
""")

    chain = (
        {
            "context": lambda x: format_docs(
                retriever.invoke(x["question"])
            ),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def create_vectorstore(chunks):
    return FAISS.from_documents(chunks, embeddings)


# ── PDF Upload ────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📄 Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    current_signature = get_files_signature(uploaded_files)

    if (
        "vectorstore" not in st.session_state
        or st.session_state.get("files_signature") != current_signature
    ):
        st.session_state.messages = []

        with st.spinner("Processing PDFs..."):
            all_docs = []
            total_pages = 0

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                total_pages += len(pages)
                all_docs.extend(pages)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(all_docs)
            vectorstore = create_vectorstore(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.total_pages = total_pages
            st.session_state.total_chunks = len(chunks)
            st.session_state.files_signature = current_signature

    vectorstore = st.session_state.vectorstore
    total_pages = st.session_state.total_pages
    total_chunks = st.session_state.total_chunks

    chain, retriever = build_chain(vectorstore)

    st.success(
        f"✅ {len(uploaded_files)} PDF(s) processed! "
        f"{total_pages} pages → {total_chunks} chunks"
    )

    # ── Summarize Button ──────────────────────────────────────
    if st.button("📝 Summarize Uploaded PDFs"):
        with st.spinner("Generating summary..."):
            summary = chain.invoke({
                "question": "Summarize the uploaded PDFs in a concise but informative way. Cover the main topics, key ideas, and important conclusions.",
                "chat_history": ""
            })
        st.subheader("📝 Document Summary")
        st.write(summary)

    st.divider()

    # ── Chat Interface ────────────────────────────────────────
    st.subheader("💬 Ask a Question")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask something about your PDFs...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_history = format_chat_history(st.session_state.messages)
                answer = chain.invoke({
                    "question": question,
                    "chat_history": chat_history
                })
                scored_docs = vectorstore.similarity_search_with_score(question, k=3)
                source_docs = [doc for doc, score in scored_docs]
                top_score = scored_docs[0][1]
                confidence_label = get_confidence_label(top_score)

            st.write(answer)
            st.markdown(f"**Confidence:** {confidence_label}")

            with st.expander("📄 View Sources"):
                for i, doc in enumerate(source_docs):
                    page = doc.metadata.get("page", "Unknown")
                    page_num = page + 1 if isinstance(page, int) else page
                    st.markdown(f"**Source {i+1} — Page {page_num}:**")
                    st.caption(doc.page_content[:300] + "...")
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👆 Please upload one or more PDFs to get started!")


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app uses **RAG (Retrieval Augmented Generation)** "
        "to answer questions from multiple PDFs using Google Gemini."
    )
    st.divider()
    st.header("⚙️ How it works")
    st.write("1. Upload one or more PDFs")
    st.write("2. PDFs are split into chunks")
    st.write("3. Chunks are embedded using HuggingFace")
    st.write("4. Your question retrieves relevant chunks")
    st.write("5. Gemini generates an answer from those chunks")
    st.divider()
    st.header("✨ Features")
    st.write("- 📄 Multi-PDF support")
    st.write("- 💬 Chat history awareness")
    st.write("- 📍 Source citation with page numbers")
    st.write("- 📊 Answer confidence scoring")
    st.write("- 📝 One-click PDF summarization")
    st.divider()
    st.header("🛠️ Tech Stack")
    st.write("- LangChain")
    st.write("- Google Gemini API")
    st.write("- HuggingFace Embeddings")
    st.write("- FAISS Vector Database")
    st.write("- Streamlit")