"""
llm_handler.py
Handles communication with Google Gemini 1.5 Flash API.
Constructs a RAG prompt using retrieved context and the user's question.
"""

import google.generativeai as genai


SYSTEM_PROMPT = """You are an expert document assistant. Your job is to answer the user's question 
accurately and concisely using ONLY the provided document context.

Rules:
- Answer strictly based on the given context.
- If the context does not contain enough information, say: "I couldn't find relevant information in the uploaded documents."
- Be precise, clear, and professional.
- Format your answer with bullet points or numbered lists where appropriate.
"""


class GeminiHandler:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT
        )

    def generate(self, question: str, context: str) -> str:
        """Generate an answer using Gemini with RAG context."""
        prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Please answer the question based strictly on the document context above."""

        response = self.model.generate_content(prompt)
        return response.text
