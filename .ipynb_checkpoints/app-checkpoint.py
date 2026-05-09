import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import tempfile

# ── Load API Key ──────────────────────────────────────────────
load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ── Title ─────────────────────────────────────────────────────
st.title("🤖 RAG Chatbot")
st.write("Upload a PDF and ask questions about it!")

st.divider()

# ── Initialize models ─────────────────────────────────────────
@st.cache_resource
def load_models():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    return embeddings, llm

embeddings, llm = load_models()

# ── Helper functions ──────────────────────────────────────────
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(pages)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(pages), len(chunks)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context provided below.
    If you don't know the answer from the context, say "I don't have enough information to answer this."

    Context: {context}

    Question: {question}

    Answer:
    """)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ── PDF Upload ────────────────────────────────────────────────
uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Process the PDF
        vectorstore, num_pages, num_chunks = process_pdf(tmp_path)
        chain = build_chain(vectorstore)

        st.success(f"✅ PDF processed! {num_pages} pages → {num_chunks} chunks")

    st.divider()

    # ── Chat Interface ────────────────────────────────────────
    st.subheader("💬 Ask a Question")

    # Store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    question = st.chat_input("Ask something about your PDF...")

    if question:
        # Show user message
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke(question)
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👆 Please upload a PDF to get started!")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app uses **RAG (Retrieval Augmented Generation)** "
        "to answer questions from your PDF using Google Gemini."
    )
    st.divider()
    st.header("⚙️ How it works")
    st.write("1. Upload a PDF")
    st.write("2. PDF is split into chunks")
    st.write("3. Chunks are converted to embeddings")
    st.write("4. Your question retrieves relevant chunks")
    st.write("5. Gemini generates an answer from those chunks")
    st.divider()
    st.header("🛠️ Tech Stack")
    st.write("- LangChain")
    st.write("- Google Gemini API")
    st.write("- FAISS Vector Database")
    st.write("- Streamlit")