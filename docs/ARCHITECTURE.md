# Architecture

## Request flow

```text
Streamlit UI
    |
    +--> PDF validation + fingerprinting
    |
    +--> pypdf extraction
    |
    +--> page-aware chunking
    |
    +--> ChromaDB persistent semantic index
    |
    +--> multi-document retrieval + score filtering
    |
    +--> bounded conversation context
    |
    +--> grounded answer engine
           |-- optional local Ollama
           `-- deterministic extractive fallback
    |
    `--> answer + document/page evidence
```

## Design goals

1. **Zero-cost first:** paid inference is never required.
2. **Grounding:** answers are derived from retrieved PDF evidence.
3. **Traceability:** retrieved evidence keeps document and page metadata.
4. **Bounded state:** conversation context is capped to prevent unbounded prompt growth.
5. **Testability:** retrieval and answer behavior are covered by automated regression tests.

## Deployment note

GitHub Pages hosts the static project showcase only. The Python/Streamlit application requires a Python-compatible runtime. A free Streamlit-compatible host can run the interactive application, while local Ollama provides optional inference without a paid API.
