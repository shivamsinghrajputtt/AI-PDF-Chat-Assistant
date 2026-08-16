"""Small, deterministic conversation-memory layer for follow-up questions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class ConversationMemory:
    """Keep a bounded conversation history without external storage."""

    def __init__(self, max_messages: int = 10) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than zero")
        self.max_messages = max_messages
        self._messages: list[ChatMessage] = []

    @property
    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def add(self, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        if content.strip():
            self._messages.append(ChatMessage(role, content.strip()))
            self._messages = self._messages[-self.max_messages :]

    def clear(self) -> None:
        self._messages.clear()

    def contextual_query(self, question: str) -> str:
        """Build a lightweight query using recent turns, without an LLM call."""
        recent = self._messages[-4:]
        if not recent:
            return question.strip()
        context = " ".join(message.content for message in recent)
        return f"{context} {question.strip()}".strip()
