"""Grounded answer generation with a zero-cost local LLM option."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Protocol, Sequence

from .rag import RetrievedChunk


class LLMProvider(Protocol):
    """Minimal interface required by the grounded answer engine."""

    def generate(self, prompt: str) -> str:
        """Generate an answer from a fully grounded prompt."""


@dataclass(frozen=True)
class GroundedAnswer:
    """Answer plus the source pages used to build it."""

    text: str
    sources: tuple[int, ...]
    used_llm: bool


def build_grounded_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Build a strict context-only prompt for an LLM."""
    context = "\n\n".join(
        f"[Page {chunk.page}]\n{chunk.text}" for chunk in chunks
    )
    return (
        "You answer questions about a PDF. Use ONLY the supplied context. "
        "If the context does not contain enough information, say exactly that "
        "you could not find enough information in the PDF. Do not invent facts. "
        "Keep the answer concise and cite relevant page numbers like [Page 2].\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Context:\n{context}"
    )


def build_extractive_answer(
    question: str, chunks: Sequence[RetrievedChunk]
) -> GroundedAnswer:
    """Return a deterministic grounded response when no LLM is configured."""
    del question
    if not chunks:
        return GroundedAnswer(
            text="I couldn't find enough information in the PDF to answer this question.",
            sources=(),
            used_llm=False,
        )

    sources = tuple(dict.fromkeys(chunk.page for chunk in chunks if chunk.page > 0))
    excerpts = []
    for chunk in chunks[:3]:
        if chunk.text.strip():
            excerpts.append(f"[Page {chunk.page}] {chunk.text.strip()}")

    return GroundedAnswer(
        text="\n\n".join(excerpts),
        sources=sources,
        used_llm=False,
    )


class OllamaProvider:
    """Optional local Ollama provider; no API key or paid service required."""

    def __init__(
        self,
        model: str = "gemma3:1b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 90.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """Generate text through Ollama's local HTTP API."""
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("Local Ollama server is unavailable") from exc

        answer = str(data.get("response", "")).strip()
        if not answer:
            raise RuntimeError("Local LLM returned an empty response")
        return answer


class GroundedAnswerEngine:
    """Generate answers from retrieved chunks, with a deterministic fallback."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.llm = llm

    def answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> GroundedAnswer:
        """Generate a grounded answer without ever allowing an empty context."""
        if not question.strip():
            return GroundedAnswer("Please enter a question.", (), False)
        if not chunks:
            return build_extractive_answer(question, chunks)

        sources = tuple(dict.fromkeys(chunk.page for chunk in chunks if chunk.page > 0))
        if self.llm is None:
            return build_extractive_answer(question, chunks)

        try:
            generated = self.llm.generate(build_grounded_prompt(question, chunks))
        except RuntimeError:
            return build_extractive_answer(question, chunks)

        return GroundedAnswer(generated, sources, True)
