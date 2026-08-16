# 🤖 AI PDF Chat Assistant

A portfolio-grade **multi-document PDF RAG assistant** built with Python, Streamlit, ChromaDB and retrieval-grounded answering.

> **Zero-cost by design:** no paid AI API is required. Deterministic grounded excerpts work out of the box; Ollama is optional for local LLM generation.

## ✨ What it does

- 📄 Validates and extracts PDF text with `pypdf`
- 🧩 Page-aware overlapping chunking
- 📚 Indexes multiple PDFs in ChromaDB
- 🔎 Performs semantic retrieval with relevance filtering
- 🧠 Maintains bounded conversation context for follow-up questions
- 🎯 Produces grounded answers from retrieved evidence
- 🔗 Preserves document + page-level sources
- 🛡️ Falls back safely when there is no useful context or Ollama is unavailable
- 📊 Measures retrieval quality with Hit Rate and MRR
- 🧪 Runs automated regression tests in GitHub Actions
- 🌐 Includes a static GitHub Pages portfolio showcase

## 🧱 Architecture

```text
Streamlit UI
   ↓
PDF validation + fingerprinting
   ↓
pypdf extraction
   ↓
Page-aware chunking
   ↓
ChromaDB semantic index
   ↓
Multi-document retrieval + score filtering
   ↓
Bounded conversation context
   ↓
Grounded answer engine
   ├── Optional local Ollama
   └── Deterministic extractive fallback
   ↓
Answer + document/page evidence
```

Detailed design: `docs/ARCHITECTURE.md`

## 💰 Zero-cost deployment strategy

The core application does not require OpenAI, Gemini, Anthropic, or another paid inference API.

- **Default:** semantic retrieval + deterministic grounded excerpts
- **Optional:** Ollama running locally for LLM generation
- **Showcase:** GitHub Pages hosts the static portfolio page
- **Interactive runtime:** deploy the Streamlit app on a free Streamlit-compatible Python host, or run it locally

See `docs/DEPLOYMENT.md` for the deployment checklist.

## 🚀 Run locally

```bash
git clone https://github.com/shivamsinghrajputtt/AI-PDF-Chat-Assistant.git
cd AI-PDF-Chat-Assistant
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

### Optional local LLM

Install Ollama and pull a small model such as `gemma3:1b`. The application automatically falls back to deterministic extraction if Ollama is unavailable.

## 🧪 Tests

```bash
pytest -q
```

The current CI suite covers PDF processing, retrieval, grounded answering, evaluation, multi-document retrieval, and bounded conversation memory.

## 📊 Retrieval evaluation

- **Hit Rate:** percentage of evaluation questions where at least one expected source page was retrieved.
- **MRR:** rewards relevant evidence appearing near the top of the retrieval ranking.
- **Score thresholding:** filters weak semantic matches before answer generation.

No paid evaluation service is required.

## 🔐 Security & privacy

See `SECURITY.md`.

Never commit secrets or upload confidential documents to an uncontrolled public deployment.

## ⚡ Performance

The system bounds upload size, top-k retrieval and conversation memory. Deterministic extraction avoids model inference when Ollama is unavailable. See `docs/PERFORMANCE.md`.

## 🗺️ Roadmap

- [x] PDF extraction + page-aware chunking
- [x] Semantic retrieval + persistent ChromaDB
- [x] Grounded answers + page sources
- [x] Zero-cost fallback + optional local Ollama
- [x] Retrieval evaluation + regression CI
- [x] Multi-document RAG
- [x] Bounded conversation memory
- [x] Production validation + security documentation
- [x] Static portfolio showcase
- [ ] Optional reranking experiments
- [ ] Hosted interactive demo when a suitable free runtime is available

## 👨‍💻 Developer

**Shivam Kumar Singh**  
B.Tech CSE Student • GenAI & Full-Stack Development

[GitHub](https://github.com/shivamsinghrajputtt)
