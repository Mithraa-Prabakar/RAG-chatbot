# RAG Chatbot — Intelligent Document Q&A

A production-ready RAG chatbot powered by **Google Gemini 1.5 Flash** + **FAISS** + **SentenceTransformers**.

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd rag-chatbot
pip install -r requirements.txt
```

### 2. Get a Gemini API Key
- Visit https://aistudio.google.com/
- Create a free API key

### 3. Run the App
```bash
streamlit run app.py
```

## 📁 Project Structure
```
rag-chatbot/
├── app.py                  # Streamlit UI — main entry point
├── document_processor.py   # PDF/TXT extraction + chunking
├── vector_store.py         # FAISS index build + similarity search
├── llm_handler.py          # Google Gemini API integration
├── requirements.txt        # Python dependencies
└── README.md
```

## 🏗️ Architecture

```
User uploads PDF/TXT
        ↓
document_processor.py → Extract text → Split into chunks
        ↓
vector_store.py → Embed chunks (SentenceTransformers) → Build FAISS index
        ↓
User asks a question
        ↓
vector_store.py → Embed query → Similarity search → Top-4 relevant chunks
        ↓
llm_handler.py → [Context + Question] → Gemini 1.5 Flash → Answer
        ↓
Streamlit UI displays answer + source context
```

## ☁️ Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your repo and set `app.py` as the entry point
4. Add your Gemini API key in the app's sidebar (no need to hardcode it)

## 🔧 Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 800 words | Words per text chunk |
| `overlap` | 150 words | Overlap between chunks |
| `k` (retrieval) | 4 | Number of chunks retrieved per query |
| Embedding model | `all-MiniLM-L6-v2` | SentenceTransformers model |
| LLM | `gemini-1.5-flash` | Google Gemini model |
