"""Semantic chunking and vector retrieval for PDF question answering."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chromadb


@dataclass(frozen=True)
class DocumentChunk:
    """A retrievable piece of a PDF with source metadata."""

    chunk_id: str
    text: str
    page: int
    chunk_index: int
    document_id: str = ""
    document_name: str = ""


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by semantic similarity search."""

    chunk_id: str
    text: str
    page: int
    score: float
    document_id: str = ""
    document_name: str = ""


def chunk_pages(
    pages: Sequence[str], *, chunk_size: int = 900, overlap: int = 150,
    document_id: str = "", document_name: str = "",
) -> list[DocumentChunk]:
    """Split page text into overlapping, page-aware chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    for page_number, raw_page in enumerate(pages, start=1):
        text = re.sub(r"\s+", " ", raw_page).strip()
        if not text:
            continue
        start = 0
        page_chunk_index = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start + chunk_size // 2:
                    end = boundary
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = hashlib.sha256(
                    f"{document_id}:{page_number}:{page_chunk_index}:{chunk_text}".encode("utf-8")
                ).hexdigest()[:24]
                chunks.append(DocumentChunk(
                    chunk_id, chunk_text, page_number, page_chunk_index,
                    document_id, document_name,
                ))
                page_chunk_index += 1
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks


def document_collection_name(pages: Sequence[str]) -> str:
    """Return a stable Chroma collection name for one PDF's content."""
    digest = hashlib.sha256("\n\n".join(pages).encode("utf-8")).hexdigest()[:16]
    return f"pdf_{digest}"


class SemanticRetriever:
    """Persistent Chroma-backed semantic retriever."""

    def __init__(self, storage_path: str | Path = ".chroma", *, client: Any | None = None,
                 embedding_function: Any | None = None) -> None:
        self.client = client or chromadb.PersistentClient(path=str(storage_path))
        self.embedding_function = embedding_function

    def _get_collection(self, name: str):
        kwargs: dict[str, Any] = {"name": name, "configuration": {"hnsw": {"space": "cosine"}}}
        if self.embedding_function is not None:
            kwargs["embedding_function"] = self.embedding_function
        return self.client.get_or_create_collection(**kwargs)

    def index_pages(self, pages: Sequence[str], *, chunk_size: int = 900, overlap: int = 150,
                    document_id: str = "", document_name: str = "") -> str:
        """Chunk and upsert a PDF into a dedicated semantic collection."""
        if not any(page.strip() for page in pages):
            raise ValueError("Cannot index a PDF with no extractable text")
        document_id = document_id or hashlib.sha256("\n".join(pages).encode()).hexdigest()[:16]
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap,
                             document_id=document_id, document_name=document_name)
        collection_name = document_collection_name(pages)
        collection = self._get_collection(collection_name)
        collection.upsert(
            ids=[c.chunk_id for c in chunks], documents=[c.text for c in chunks],
            metadatas=[{"page": c.page, "chunk_index": c.chunk_index,
                        "document_id": c.document_id, "document_name": c.document_name} for c in chunks],
        )
        return collection_name

    def retrieve(self, collection_name: str, query: str, *, top_k: int = 4) -> list[RetrievedChunk]:
        """Return nearest chunks from one document collection."""
        if not query.strip() or top_k <= 0:
            return []
        kwargs: dict[str, Any] = {"name": collection_name}
        if self.embedding_function is not None:
            kwargs["embedding_function"] = self.embedding_function
        collection = self.client.get_collection(**kwargs)
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(query_texts=[query], n_results=min(top_k, count),
                                   include=["documents", "metadatas", "distances"])
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]
        return [RetrievedChunk(
            chunk_id=chunk_id, text=text or "", page=int((metadata or {}).get("page", 0)),
            score=max(0.0, 1.0 - float(distance)),
            document_id=str((metadata or {}).get("document_id", "")),
            document_name=str((metadata or {}).get("document_name", "")),
        ) for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances)]

    def retrieve_many(self, collection_names: Sequence[str], query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve from multiple PDFs and return one globally ranked result list."""
        if not collection_names or not query.strip() or top_k <= 0:
            return []
        candidates: list[RetrievedChunk] = []
        per_document_k = max(top_k, 3)
        for name in collection_names:
            candidates.extend(self.retrieve(name, query, top_k=per_document_k))
        candidates.sort(key=lambda chunk: chunk.score, reverse=True)
        return candidates[:top_k]
