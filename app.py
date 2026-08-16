import hashlib

import streamlit as st

from src.answer_engine import GroundedAnswerEngine, OllamaProvider
from src.chat import ConversationMemory
from src.evaluation import filter_relevant_chunks
from src.input_validation import validate_pdf_upload
from src.pdf_qa import extract_pages_from_pdf
from src.rag import SemanticRetriever


st.set_page_config(page_title="AI PDF Chat Assistant", page_icon="📄")

DEFAULT_MIN_RELEVANCE = 0.35
MAX_RETRIEVAL_RESULTS = 6


@st.cache_resource
def get_retriever() -> SemanticRetriever:
    """Reuse one Chroma client across Streamlit reruns."""
    return SemanticRetriever()


def main() -> None:
    st.title("📄 AI PDF Chat Assistant")
    st.caption("Ask grounded questions across multiple PDFs with local semantic retrieval.")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True,
        help="Upload multiple documents to search them together."
    )
    use_local_llm = st.checkbox(
        "Use local AI answer generation (Ollama)",
        help="Optional and free. Requires Ollama running locally with a model installed.",
    )
    min_relevance = st.slider(
        "Minimum retrieval relevance", 0.0, 0.9, DEFAULT_MIN_RELEVANCE, 0.05,
        help="Higher values reject weak semantic matches and reduce unsupported answers.",
    )

    if "documents" not in st.session_state:
        st.session_state["documents"] = {}
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationMemory(max_messages=10)

    retriever = get_retriever()
    documents = st.session_state["documents"]

    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_bytes = uploaded_file.getvalue()
            valid, message = validate_pdf_upload(pdf_bytes)
            pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
            if not valid:
                st.error(f"{uploaded_file.name}: {message}")
                continue
            if pdf_hash in documents:
                continue
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                pages = extract_pages_from_pdf(uploaded_file)
                if not pages:
                    st.error(f"{uploaded_file.name}: no extractable text found.")
                    continue
                collection_name = retriever.index_pages(
                    pages, document_id=pdf_hash[:16], document_name=uploaded_file.name
                )
                documents[pdf_hash] = {
                    "name": uploaded_file.name,
                    "collection": collection_name,
                    "pages": len(pages),
                }

    if documents:
        st.success(f"{len(documents)} PDF(s) ready for chat.")
        for document in documents.values():
            st.caption(f"📄 {document['name']} · {document['pages']} pages")

    for message in st.session_state["memory"].messages:
        with st.chat_message(message.role):
            st.write(message.content)

    user_query = st.chat_input("Ask a question about your PDFs")
    if user_query:
        if not documents:
            st.warning("Please upload at least one readable PDF first.")
            return

        memory = st.session_state["memory"]
        memory.add("user", user_query)
        search_query = memory.contextual_query(user_query)
        collection_names = [doc["collection"] for doc in documents.values()]

        with st.spinner("Searching across your PDFs..."):
            raw_chunks = retriever.retrieve_many(
                collection_names, search_query, top_k=MAX_RETRIEVAL_RESULTS
            )
            chunks = filter_relevant_chunks(raw_chunks, min_score=min_relevance)

        llm = OllamaProvider() if use_local_llm else None
        engine = GroundedAnswerEngine(llm=llm)
        with st.spinner("Generating grounded answer..."):
            result = engine.answer(user_query, chunks)

        memory.add("assistant", result.text)
        st.rerun()


if __name__ == "__main__":
    main()
