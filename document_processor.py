"""
document_processor.py
Extracts text from PDFs and TXTs, then splits into overlapping chunks.
"""

from pathlib import Path
from typing import List


def extract_text(file_path: str) -> str:
    """Extract raw text from a PDF or TXT file."""
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except ImportError:
            raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")
    elif path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks for better retrieval coverage."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap  # sliding window with overlap
    return chunks


def process_documents(file_paths: List[str]) -> List[str]:
    """Process multiple files and return a flat list of text chunks."""
    all_chunks = []
    for fp in file_paths:
        print(f"Processing: {fp}")
        text = extract_text(fp)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks extracted")
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks
