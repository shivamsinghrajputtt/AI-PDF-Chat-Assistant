"""Input validation helpers for the PDF application."""

from __future__ import annotations

MAX_PDF_SIZE_MB = 10
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024
PDF_MAGIC = b"%PDF-"


def validate_pdf_upload(data: bytes, *, max_bytes: int = MAX_PDF_SIZE_BYTES) -> tuple[bool, str]:
    """Validate an uploaded file before parsing or indexing it."""
    if not data:
        return False, "The uploaded file is empty."
    if len(data) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        return False, f"PDF is too large. Maximum supported size is {limit_mb:g} MB."
    if not data.startswith(PDF_MAGIC):
        return False, "The uploaded file does not appear to be a valid PDF."
    return True, ""
