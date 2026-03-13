import streamlit as st
import os
from pathlib import Path
from document_processor import process_documents
from vector_store import VectorStore
from llm_handler import GeminiHandler

# Page config
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot — Smart Document Q&A")
st.markdown("Upload PDFs and ask questions in natural language. Powered by **Google Gemini + FAISS**.")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Sidebar — API key + file upload
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password", help="Get yours at https://aistudio.google.com/")

    st.divider()
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files and api_key:
        if st.button("🔄 Process Documents", use_container_width=True):
            with st.spinner("Extracting text and building vector index..."):
                try:
                    # Save uploads to temp dir
                    tmp_dir = Path("tmp_uploads")
                    tmp_dir.mkdir(exist_ok=True)
                    file_paths = []
                    for f in uploaded_files:
                        fp = tmp_dir / f.name
                        fp.write_bytes(f.read())
                        file_paths.append(str(fp))

                    # Process documents → chunks
                    chunks = process_documents(file_paths)

                    # Build FAISS vector store
                    vs = VectorStore()
                    vs.build(chunks)
                    st.session_state.vector_store = vs
                    st.session_state.documents_loaded = True
                    st.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s)!")
                except Exception as e:
                    st.error(f"Error: {e}")
    elif uploaded_files and not api_key:
        st.warning("Please enter your Gemini API key first.")

    if st.session_state.documents_loaded:
        st.success("📚 Documents ready! Ask questions below.")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main chat area
if not st.session_state.documents_loaded:
    st.info("👈 Upload documents and enter your API key in the sidebar to get started.")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
        else:
            # Show user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Retrieve + generate
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Retrieve relevant chunks
                        relevant_chunks = st.session_state.vector_store.search(prompt, k=4)
                        context = "\n\n".join(relevant_chunks)

                        # Generate answer with Gemini
                        handler = GeminiHandler(api_key)
                        answer = handler.generate(prompt, context)

                        st.markdown(answer)

                        # Show source context in expander
                        with st.expander("📎 Retrieved Context"):
                            for i, chunk in enumerate(relevant_chunks, 1):
                                st.markdown(f"**Chunk {i}:**\n{chunk[:500]}...")

                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        err = f"Error generating answer: {e}"
                        st.error(err)
                        st.session_state.chat_history.append({"role": "assistant", "content": err})
