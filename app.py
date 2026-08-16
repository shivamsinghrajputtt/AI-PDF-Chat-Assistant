import hashlib
import time

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
    return SemanticRetriever()


def reset_chat() -> None:
    st.session_state["memory"] = ConversationMemory(max_messages=10)


def main() -> None:
    st.title("📄 AI PDF Chat Assistant")
    st.caption("Grounded multi-document RAG with local, zero-cost retrieval.")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True,
        help="Upload multiple documents to search them together.",
    )
    use_local_llm = st.checkbox(
        "Use local AI answer generation (Ollama)",
        help="Optional and free. Requires Ollama and a local model.",
    )
    min_relevance = st.slider(
        "Minimum retrieval relevance", 0.0, 0.9, DEFAULT_MIN_RELEVANCE, 0.05,
        help="Higher values reject weaker semantic matches.",
    )

    if "documents" not in st.session_state:
        st.session_state["documents"] = {}
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationMemory(max_messages=10)

    if st.button("Reset chat"):
        reset_chat()
        st.rerun()

    retriever = get_retriever()
    documents = st.session_state["documents"]

    if uploaded_files:
        active_hashes = set()
        for uploaded_file in uploaded_files:
            pdf_bytes = uploaded_file.getvalue()
            valid, message = validate_pdf_upload(pdf_bytes)
            pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
            active_hashes.add(pdf_hash)
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

        stale_hashes = set(documents) - active_hashes
        for stale_hash in stale_hashes:
            del documents[stale_hash]
        if stale_hashes:
            reset_chat()

    if documents:
        st.success(f"{len(documents)} PDF(s) ready for chat.")
        with st.expander("Documents"):
            for document in documents.values():
                st.write(f"📄 **{document['name']}** · {document['pages']} pages")

    for message in st.session_state["memory"].messages:
        with st.chat_message(message.role):
            st.write(message.content)

    user_query = st.chat_input("Ask a question about your PDFs")
    if not user_query:
        return
    if not documents:
        st.warning("Please upload at least one readable PDF first.")
        return

    memory = st.session_state["memory"]
    search_query = memory.contextual_query(user_query)
    collection_names = [doc["collection"] for doc in documents.values()]

    start = time.perf_counter()
    with st.spinner("Searching across your PDFs..."):
        raw_chunks = retriever.retrieve_many(
            collection_names, search_query, top_k=MAX_RETRIEVAL_RESULTS
        )
        chunks = filter_relevant_chunks(raw_chunks, min_score=min_relevance)
    retrieval_ms = (time.perf_counter() - start) * 1000

    llm = OllamaProvider() if use_local_llm else None
    engine = GroundedAnswerEngine(llm=llm)
    with st.spinner("Generating grounded answer..."):
        result = engine.answer(user_query, chunks)

    memory.add("user", user_query)
    memory.add("assistant", result.text)

    with st.chat_message("assistant"):
        st.write(result.text)
        if result.sources:
            st.caption("Sources: " + ", ".join(f"Page {page}" for page in result.sources))
        st.caption(f"Retrieval: {retrieval_ms:.0f} ms · {len(chunks)} evidence chunks")
        if chunks:
            with st.expander("Retrieved evidence"):
                for chunk in chunks:
                    source = chunk.document_name or "Document"
                    st.markdown(
                        f"**{source} · Page {chunk.page} · relevance {chunk.score:.2f}**"
                    )
                    st.write(chunk.text)
        else:
            st.warning("No passage met the relevance threshold. Try a more specific question.")

    if not result.used_llm:
        st.info("Zero-cost mode: retrieval + grounded extractive answer. Ollama is optional.")


if __name__ == "__main__":
    main()
