from src.answer_engine import GroundedAnswerEngine, build_grounded_prompt
from src.rag import RetrievedChunk


class FakeLLM:
    def __init__(self):
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return "The answer is grounded in the PDF. [Page 2]"


class FailingLLM:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("unavailable")


def sample_chunks():
    return [
        RetrievedChunk(
            chunk_id="a",
            text="Python is used for data analysis.",
            page=2,
            score=0.91,
        ),
        RetrievedChunk(
            chunk_id="b",
            text="Pandas provides tabular data structures.",
            page=3,
            score=0.82,
        ),
    ]


def test_grounded_prompt_contains_only_retrieved_context():
    prompt = build_grounded_prompt("What is Python used for?", sample_chunks())

    assert "What is Python used for?" in prompt
    assert "Python is used for data analysis." in prompt
    assert "[Page 2]" in prompt
    assert "Pandas provides tabular data structures." in prompt


def test_engine_uses_llm_when_configured():
    llm = FakeLLM()
    result = GroundedAnswerEngine(llm).answer("What is Python used for?", sample_chunks())

    assert result.used_llm is True
    assert result.text == "The answer is grounded in the PDF. [Page 2]"
    assert result.sources == (2, 3)
    assert "Python is used for data analysis." in llm.prompt


def test_engine_has_zero_cost_extractive_fallback():
    result = GroundedAnswerEngine().answer("What is Python used for?", sample_chunks())

    assert result.used_llm is False
    assert "Python is used for data analysis." in result.text
    assert result.sources == (2, 3)


def test_engine_falls_back_when_local_llm_is_unavailable():
    result = GroundedAnswerEngine(FailingLLM()).answer(
        "What is Python used for?", sample_chunks()
    )

    assert result.used_llm is False
    assert "Python is used for data analysis." in result.text


def test_engine_rejects_empty_question_without_calling_llm():
    result = GroundedAnswerEngine(FakeLLM()).answer("  ", sample_chunks())

    assert result.text == "Please enter a question."
    assert result.sources == ()
    assert result.used_llm is False


def test_engine_handles_no_retrieval_results():
    result = GroundedAnswerEngine().answer("Unknown question", [])

    assert result.used_llm is False
    assert "couldn't find enough information" in result.text
    assert result.sources == ()
