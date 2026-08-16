"""Lightweight, zero-cost evaluation utilities for the PDF RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .rag import RetrievedChunk


@dataclass(frozen=True)
class RetrievalCase:
    """A question and the page(s) expected to contain its answer."""

    question: str
    relevant_pages: frozenset[int]


@dataclass(frozen=True)
class RetrievalMetrics:
    """Simple retrieval metrics that are easy to explain in a portfolio."""

    cases: int
    hit_rate: float
    mean_reciprocal_rank: float


def filter_relevant_chunks(
    chunks: Sequence[RetrievedChunk], *, min_score: float = 0.0
) -> list[RetrievedChunk]:
    """Drop weak retrievals while preserving Chroma ranking order."""
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    return [chunk for chunk in chunks if chunk.score >= min_score]


def evaluate_retrieval(
    cases: Sequence[RetrievalCase],
    results: Sequence[Sequence[RetrievedChunk]],
) -> RetrievalMetrics:
    """Calculate hit-rate and MRR without requiring a paid evaluation API."""
    if len(cases) != len(results):
        raise ValueError("cases and results must have the same length")
    if not cases:
        return RetrievalMetrics(cases=0, hit_rate=0.0, mean_reciprocal_rank=0.0)

    hits = 0
    reciprocal_rank_total = 0.0
    for case, retrieved in zip(cases, results):
        rank = next(
            (
                index
                for index, chunk in enumerate(retrieved, start=1)
                if chunk.page in case.relevant_pages
            ),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_rank_total += 1.0 / rank

    total = len(cases)
    return RetrievalMetrics(
        cases=total,
        hit_rate=hits / total,
        mean_reciprocal_rank=reciprocal_rank_total / total,
    )
