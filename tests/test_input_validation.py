from src.input_validation import validate_pdf_upload


def test_validate_pdf_upload_accepts_pdf_header():
    valid, message = validate_pdf_upload(b"%PDF-1.7\ncontent")

    assert valid is True
    assert message == ""


def test_validate_pdf_upload_rejects_empty_file():
    valid, message = validate_pdf_upload(b"")

    assert valid is False
    assert "empty" in message.lower()


def test_validate_pdf_upload_rejects_non_pdf():
    valid, message = validate_pdf_upload(b"not a pdf")

    assert valid is False
    assert "valid pdf" in message.lower()


def test_validate_pdf_upload_rejects_oversized_file():
    valid, message = validate_pdf_upload(b"%PDF-" + b"x" * 10, max_bytes=5)

    assert valid is False
    assert "too large" in message.lower()
