"""Core PDF extraction and lightweight retrieval utilities.

This module intentionally contains no Streamlit UI code so the core behavior can
be tested independently and reused by future RAG implementations.
"""

from __future__ import annotations

import re
from typing import BinaryIO

from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file: BinaryIO) -> str:
    "