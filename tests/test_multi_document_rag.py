import chromadb

from src.rag import SemanticRetriever


class FakeEmbeddingFunction:
    is_legacy = False

    @staticmethod
    def _embed(input):
        vectors = []
        for text in input:
            normalized = text.lower()
            if "python" in normalized:
                vectors.append([1.0, 0.0, 0.0])
            elif "database" in normalized:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    def embed_documents(self, input):
        return self._embed(input)

    def embed_query(self, input):
        return self._embed(input)

    def __call__(self, input):
        return self._embed(input)

    def name(self):
        return "default"


def test_retrieve_many_ranks_results_across_documents():
    retriever = SemanticRetriever(
        client=chromadb.EphemeralClient(),
        embedding_function=FakeEmbeddingFunction(),
    )
    first = retriever.index_pages(
        ["Python is a programming language."],
        document_id="one", document_name="python.pdf",
    )
    second = retriever.index_pages(
        ["Database systems store and query data."],
        document_id="two", document_name="database.pdf",
    )

    results = retriever.retrieve_many([first, second], "python programming", top_k=2)

    assert results
    assert results[0].document_name == "python.pdf"
    assert results[0].page == 1
