import pytest

from src.chat import ConversationMemory


def test_memory_keeps_recent_messages_only():
    memory = ConversationMemory(max_messages=2)
    memory.add("user", "first")
    memory.add("assistant", "second")
    memory.add("user", "third")

    assert [message.content for message in memory.messages] == ["second", "third"]


def test_contextual_query_includes_recent_context():
    memory = ConversationMemory(max_messages=4)
    memory.add("user", "What is the revenue?")
    memory.add("assistant", "The revenue is 10 million.")

    assert "revenue" in memory.contextual_query("What about the previous year?")


def test_memory_rejects_invalid_role():
    memory = ConversationMemory()
    with pytest.raises(ValueError):
        memory.add("system", "not allowed")
