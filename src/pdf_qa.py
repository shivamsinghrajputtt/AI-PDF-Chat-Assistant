"""Core PDF extraction and lightweight retrieval utilities."""

from __future__ import annotations

import re
from typing import BinaryIO

from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file: BinaryIO) -> str:
    """Extract text from all readable pages in an uploaded PDF.

    Returns an empty string when the PDF cannot be parsed or contains no
    extractable text. UI-specific error handling belongs in the Streamlit app.
    """
    try:
        reader = PdfReader(uploaded_file)
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text.strip())
        return "\n\n".join(pages)
    except Exception:
        return ""


def _tokenize(text: str) -> list[str]:
    """Return normalized tokens, including boundaries inside CamelCase words."""
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return [token.lower() for token in re.findall(r"\w+", normalized, flags=re.UNICODE)]


def find_relevant_sentences(pdf_text: str, user_query: str, limit: int = 3) -> list[str]:
    """Return up to ``limit`` sentences ranked by query-token overlap."""
    if not pdf_text.strip() or not user_query.strip() or limit <= 0:
        return []

    query_words = set(_tokenize(user_query))
    if not query_words:
        return []

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", pdf_text)
        if sentence.strip()
    ]

    ranked: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_words = set(_tokenize(sentence))
        score = len(query_words & sentence_words)
        if score:
            ranked.append((score, -index, sentence))

    ranked.sort(reverse=True)
    return [sentence for _, _, sentence in ranked[:limit]]


def find_answer_from_text(pdf_text: str, user_query: str) -> str:
    """Build the current keyword-retrieval answer from relevant PDF sentences."""
    if not pdf_text.strip():
        return "No extractable text was found in this PDF. Scanned/image-only PDFs may require OCR."
    if not user_query.strip():
        return "Please enter a question."

    matches = find_relevant_sentences(pdf_text, user_query)
    if not matches:
        return "I couldn't find relevant text for this question in the PDF."
    return "\n\n".join(matches)
