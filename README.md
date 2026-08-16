# 📄 AI PDF Chat Assistant

A lightweight PDF question-answering application built with **Python** and **Streamlit**.

The current implementation extracts text from a PDF and ranks relevant sentences using local keyword overlap. The codebase is intentionally structured so a semantic **RAG + LLM** pipeline can be added as a later phase.

## ✨ Current Features

- Upload PDF documents through a Streamlit UI
- Extract text locally with `pypdf`
- Ask questions about the uploaded document
- Rank relevant sentences using deterministic keyword matching
- Keep extracted text in Streamlit session state
- Handle empty, unreadable, and unsupported scanned PDFs gracefully
- Automated unit tests with pytest
- GitHub Actions CI

## 🧱 Architecture

```text
Streamlit UI
     ↓
PDF extraction (pypdf)
     ↓
Text retrieval / ranking
     ↓
Relevant document sentences
```

The retrieval layer is isolated from the UI so it can be replaced with embeddings, a vector database, and an LLM without rewriting the application layer.

## 🛠️ Tech Stack

- Python 3.11+
- Streamlit
- pypdf
- pytest
- GitHub Actions

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

## 🧪 Run Tests

```bash
pytest -q
```

## 🔭 Roadmap

- [x] Separate UI from core PDF processing
- [x] Add deterministic retrieval tests
- [x] Add GitHub Actions CI
- [ ] Add document chunking
- [ ] Add embeddings and semantic retrieval
- [ ] Add vector database persistence
- [ ] Add LLM-based grounded answers with source references
- [ ] Add evaluation cases for retrieval quality

## 👨‍💻 Developer

**Shivam Kumar Singh**  
B.Tech CSE Student • GenAI & Full-Stack Development

GitHub: https://github.com/shivamsinghrajputtt
