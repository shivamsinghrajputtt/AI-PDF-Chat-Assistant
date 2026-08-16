# Performance Notes

The project is intentionally designed for a low-cost local/freetier environment.

## Main cost/latency drivers

1. PDF extraction scales with document size and page count.
2. Embedding/retrieval work scales with the number of indexed chunks.
3. Local Ollama generation depends on the machine and selected model.
4. Larger conversation context increases prompt processing time.

## Current controls

- Uploads are size-limited.
- Chunk retrieval is top-k bounded.
- Conversation memory is bounded.
- Weak semantic matches can be filtered by a configurable threshold.
- Deterministic extraction avoids model inference when Ollama is unavailable.

## Practical tuning

For constrained hardware, keep top-k small, use a lightweight Ollama model, and avoid indexing unnecessarily large PDFs. Measure retrieval quality before increasing top-k because more context is not automatically better.
