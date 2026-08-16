import pytest

from src.evaluation import RetrievalCase, evaluate_retrieval, filter_relevant_chunks
from src.rag import RetrievedChunk


def chunk(chunk_id: str, page: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=f"page {page}", page=page, score=score)


def test_filter_relevant_chunks_keeps_rank_order():
    results = filter_relevant_chunks(
        [chunk("a", 1, 0.91), chunk("b", 2, 0.42), chunk("c", 3, 0.75)],
        min_score=0.7,
    )
    assert [item.chunk_id for item in results] == ["a", "c"]


def test_filter_relevant_chunks_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        filter_relevant_chunks([], min_score=1.1)


def test_evaluate_retrieval_calculates_hit_rate_and_mrr():
    cases = [
        RetrievalCase("q1", frozenset({2})),
        RetrievalCase("q2", frozenset({5})),
        RetrievalCase("q3", frozenset({9})),
    ]
    results = [
        [chunk("a", 1, 0.9), chunk("b", 2, 0.8)],
        [chunk("c", 5, 0.9)],
        [chunk("d", 4, 0.9)],
    ]

    metrics = evaluate_retrieval(cases, results)

    assert metrics.cases == 3
    assert metrics.hit_rate == pytest.approx(2 / 3)
    assert metrics.mean_reciprocal_rank == pytest.approx((0.5 + 1.0) / 3)


def test_evaluate_retrieval_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        evaluate_retrieval([RetrievalCase("q", frozenset({1}))], [])


def test_evaluate_retrieval_empty_dataset_is_safe():
    metrics = evaluate_retrieval([], [])
    assert metrics.cases == 0
    assert metrics.hit_rate == 0.0
    assert metrics.mean_reciprocal_rank == 0.0
