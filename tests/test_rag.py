import chromadb
import pytest

from src.rag import SemanticRetriever, chunk_pages, document_collection_name


class FakeEmbeddingFunction:
    """Small deterministic embedding function for fast retrieval tests."""

    def __call__(self, input):
        vectors = []
        for text in input:
            normalized = text.lower()
            if "cat" in normalized or "feline" in normalized:
                vectors.append([1.0, 0.0, 0.0])
            elif "finance" in normalized or "bank" in normalized:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    def name(self):
        """Match Chroma's embedding-function protocol for collection reuse."""
        return "default"


def test_chunk_pages_preserves_page_metadata_and_overlap():
    chunks = chunk_pages(
        ["one two three four five six", "page two content"],
        chunk_size=14,
        overlap=4,
    )

    assert chunks
    assert chunks[0].page == 1
    assert any(chunk.page == 2 for chunk in chunks)
    assert chunks[0].chunk_id


def test_chunk_pages_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_pages(["text"], chunk_size=10, overlap=10)


def test_document_collection_name_is_stable():
    pages = ["A page", "Another page"]
    assert document_collection_name(pages) == document_collection_name(pages)


def test_semantic_retriever_round_trip():
    retriever = SemanticRetriever(
        client=chromadb.EphemeralClient(),
        embedding_function=FakeEmbeddingFunction(),
    )
    collection_name = retriever.index_pages(
        [
            "Cats are common household pets and feline animals.",
            "Finance and banking involve money, markets, and institutions.",
        ],
        chunk_size=200,
        overlap=20,
    )

    results = retriever.retrieve(collection_name, "feline household animal", top_k=1)

    assert len(results) == 1
    assert "Cats" in results[0].text
    assert results[0].page == 1
    assert results[0].score >= 0.0
