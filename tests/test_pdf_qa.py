from src.pdf_qa import find_answer_from_text, find_relevant_sentences


def test_retrieval_ranks_matching_sentences():
    text = (
        "Python is a programming language. "
        "FastAPI is a Python framework for building APIs. "
        "Bananas are yellow."
    )

    matches = find_relevant_sentences(text, "Python API")

    assert matches
    assert "FastAPI" in matches[0]


def test_retrieval_returns_empty_for_blank_input():
    assert find_relevant_sentences("", "question") == []
    assert find_relevant_sentences("some text", "") == []


def test_answer_reports_no_match():
    answer = find_answer_from_text("Python is useful.", "database")

    assert "couldn't find relevant text" in answer


def test_answer_reports_missing_pdf_text():
    answer = find_answer_from_text("", "question")

    assert "No extractable text" in answer
