import os
import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="RAG PDF Chatbot (Groq)", page_icon="📄", layout="wide")
st.title("📄 RAG-based PDF Chatbot (Groq + FAISS)")
st.caption("Upload a PDF, ask questions, get answers grounded in your document.")

# ----------------------------
# Sidebar: settings
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # Try to read from Streamlit secrets first, else let the user paste one
    default_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    groq_api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Get a free key at https://console.groq.com/keys. "
             "Better: add it as a Streamlit secret named GROQ_API_KEY so you don't have to paste it each time.",
    )

    model_name = st.selectbox(
        "Groq model (open-source)",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ],
        index=0,
        help="Check https://console.groq.com/docs/models for the current list of available models.",
    )

    chunk_size = st.slider("Chunk size (characters)", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 500, 200, 50)
    top_k = st.slider("Chunks to retrieve per question", 1, 8, 4, 1)

    st.divider()
    if st.button("🗑️ Reset session"):
        for key in ["chunks", "index", "pdf_name", "chat_history"]:
            st.session_state.pop(key, None)
        st.rerun()

# ----------------------------
# Session state init
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content)
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ----------------------------
# Cached resources / helpers
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    # Small, fast, open-source embedding model — good fit for free-tier deployments
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap  # slide window with overlap
    return [c for c in chunks if c.strip()]


def build_faiss_index(chunks, embed_model):
    embeddings = embed_model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)  # so inner product = cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_chunks(query, embed_model, index, chunks, k=4):
    query_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k)
    results = [chunks[i] for i in indices[0] if i != -1]
    return results


def generate_answer(query, context_chunks, api_key, model_name):
    client = Groq(api_key=api_key)
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided "
        "document context. If the answer is not in the context, say you don't know "
        "based on the document. Be concise and accurate."
    )
    user_prompt = f"Context from the document:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model_name,
    )
    return chat_completion.choices[0].message.content


# ----------------------------
# Main UI: upload + process
# ----------------------------
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
    with st.spinner("Extracting text, chunking, and building the vector index..."):
        raw_text = extract_text_from_pdf(uploaded_file)

        if not raw_text.strip():
            st.error("Couldn't extract any text from this PDF. It may be a scanned/image-only PDF.")
        else:
            embed_model = load_embedding_model()
            chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=chunk_overlap)
            index = build_faiss_index(chunks, embed_model)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chat_history = []

    st.success(f"'{uploaded_file.name}' processed into {len(st.session_state.chunks)} chunks. Ask away!")

if st.session_state.chunks is not None:
    st.info(f"📚 Active document: **{st.session_state.pdf_name}** "
             f"({len(st.session_state.chunks)} chunks indexed)")

# ----------------------------
# Chat interface
# ----------------------------
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Ask a question about the uploaded PDF...")

if question:
    if st.session_state.index is None:
        st.warning("Please upload and process a PDF first.")
    elif not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
    else:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                embed_model = load_embedding_model()
                relevant_chunks = retrieve_chunks(
                    question, embed_model, st.session_state.index,
                    st.session_state.chunks, k=top_k
                )
                try:
                    answer = generate_answer(question, relevant_chunks, groq_api_key, model_name)
                except Exception as e:
                    answer = f"⚠️ Error calling Groq API: {e}"

                st.markdown(answer)
                with st.expander("🔍 Retrieved context used for this answer"):
                    for i, c in enumerate(relevant_chunks, 1):
                        st.markdown(f"**Chunk {i}:** {c}")

        st.session_state.chat_history.append(("assistant", answer))
