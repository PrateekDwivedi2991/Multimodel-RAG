import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import RAGPipeline


# --- Page Config ---
st.set_page_config(
    page_title="HR Knowledge Assistant",
    page_icon="📋",
    layout="centered",
)

st.title("📋 HR Knowledge Assistant")
st.caption("V1: Vanilla RAG | Ask questions about HR policies and procedures")


# --- Initialize Pipeline ---
@st.cache_resource
def load_pipeline():
    """Load the RAG pipeline (cached so it only loads once)."""
    return RAGPipeline()


try:
    pipeline = load_pipeline()
except FileNotFoundError as e:
    st.error(
        "⚠️ FAISS index not found. Please run the setup steps first:\n\n"
        "```bash\n"
        "python src/generate_sample_docs.py\n"
        "python src/ingest.py\n"
        "```"
    )
    st.stop()


# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **{src['source']}** ({src['dept']}) "
                        f"— relevance: `{src['score']:.3f}`"
                    )


# --- Chat Input ---
if prompt := st.chat_input("Ask a question about HR policies..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching HR documents..."):
            result = pipeline.ask(prompt)

        st.markdown(result["answer"])

        # Show sources
        with st.expander("📎 Sources"):
            for src in result["sources"]:
                st.markdown(
                    f"- **{src['source']}** ({src['dept']}) "
                    f"— relevance: `{src['score']:.3f}`"
                )

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })


# --- Sidebar ---
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Version:** V1 — Vanilla RAG

        **Pipeline:**
        1. Load HR documents (text files)
        2. Chunk (500 chars, 50 overlap)
        3. Embed (OpenAI text-embedding-3-small)
        4. Index (FAISS)
        5. Retrieve (top-5 similarity search)
        6. Generate (GPT-4o-mini)

        **Department:** HR only (V2 adds more)

        ---

        **Sample questions:**
        - What is the annual leave entitlement?
        - How does the performance review work?
        - What are the expense approval limits?
        - Tell me about health insurance plans
        - What is the probationary period?
        - How do I file a grievance?
        """
    )

    st.header("📊 Stats")
    if pipeline.retriever.vectorstore:
        num_chunks = pipeline.retriever.vectorstore.index.ntotal
        st.metric("Indexed chunks", num_chunks)
        st.metric("Department", "HR")
        st.metric("Retrieval top-k", 5)
