# 🤖 AI PDF Chat Assistant

A portfolio-grade PDF question-answering system built with **Python, Streamlit, ChromaDB and RAG**.

> **Zero-cost by design:** no paid API is required. The app can return grounded excerpts locally, and users with Ollama installed can enable local LLM generation.

## ✨ Features

- 📄 Upload and extract text from PDFs with `pypdf`
- 🧩 Page-aware overlapping chunking
- 🔎 Semantic retrieval with persistent ChromaDB
- 🧠 Grounded answer generation with strict context-only prompting
- 📚 Page-level source tracking
- 🛡️ No-context and LLM-failure fallbacks
- 🆓 Deterministic extractive mode with no API key
- 🖥️ Optional local Ollama generation
- 📊 Retrieval evaluation with Hit Rate and MRR
- 🧪 Automated regression tests
- ⚙️ GitHub Actions CI
- 🌐 GitHub Pages project showcase

## 🧱 Architecture

```text
                    ┌──────────────────┐
                    │  Streamlit UI    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ PDF Extraction   │
                    │     pypdf        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Page-aware       │
                    │ Chunking         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ ChromaDB         │
                    │ Semantic Search  │
                    └────────┬─────────┘
                             │
                             ▼
                 ┌──────────────────────────┐
                 │ Grounded Answer Engine   │
                 └────────────┬─────────────┘
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
          Local Ollama             Extractive fallback
                  │                       │
                  └───────────┬───────────┘
                              ▼
                    Answer + Page Sources
```

## 💰 Zero-cost deployment strategy

The core application does not depend on OpenAI, Gemini, Anthropic, or another paid inference API.

- **Default:** retrieval + deterministic grounded excerpts
- **Optional:** Ollama running locally for natural-language generation
- **Hosting:** GitHub Pages is used for the public project showcase. The interactive Streamlit runtime can be deployed separately on a free Streamlit-compatible host.

## 🌐 Project Page

After GitHub Pages is enabled for this repository, the showcase is available at:

`https://shivamsinghrajputtt.github.io/AI-PDF-Chat-Assistant/`

The page provides the architecture, features, and source-code entry point so recruiters can quickly inspect the project.

## 🚀 Run Locally

```bash
git clone https://github.com/shivamsinghrajputtt/AI-PDF-Chat-Assistant.git
cd AI-PDF-Chat-Assistant

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

### Optional local LLM

Install [Ollama](https://ollama.com/), then pull a small model such as `gemma3:1b`. The application falls back automatically if Ollama is unavailable.

## 🧪 Tests

```bash
pytest -q
```

## 📊 Retrieval Evaluation

Phase 4 introduces a small reproducible evaluation layer:

- **Hit Rate:** percentage of questions where at least one expected source page was retrieved.
- **MRR:** rewards retrieving a relevant page near the top of the ranking.
- **Score thresholding:** allows weak semantic matches to be filtered before answer generation.

No paid evaluation API is required.

## 🗺️ Roadmap

- [x] Separate UI from core PDF processing
- [x] Page-aware chunking
- [x] Semantic retrieval + ChromaDB
- [x] Grounded answers + page sources
- [x] Zero-cost fallback + optional local Ollama
- [x] Retrieval evaluation metrics
- [x] Regression tests + CI
- [ ] Multi-document knowledge base
- [ ] Conversation memory with document scoping
- [ ] Retrieval/reranking experiments
- [ ] Production deployment

## 👨‍💻 Developer

**Shivam Kumar Singh**  
B.Tech CSE Student • GenAI & Full-Stack Development

GitHub: https://github.com/shivamsinghrajputtt
