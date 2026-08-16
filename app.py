import streamlit as st

from src.answer_engine import GroundedAnswerEngine, OllamaProvider
from src.pdf_qa import extract_pages_from_pdf
from src.rag import SemanticRetriever


st.set_page_config(page_title="AI PDF Chat Assistant", page_icon="📄")


def main() -> None:
    """Render the Streamlit RAG application."""
    st.title("📄 AI PDF Chat Assistant")
    st.caption("Ask grounded questions about your PDF with local semantic retrieval.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    use_local_llm = st.checkbox(
        "Use local AI answer generation (Ollama)",
        help="Optional and free. Requires Ollama running locally with a model installed.",
    )

    if "collection_name" not in st.session_state:
        st.session_state["collection_name"] = None
    if "pdf_name" not in st.session_state:
        st.session_state["pdf_name"] = None

    if uploaded_file is not None:
        if st.session_state["pdf_name"] != uploaded_file.name:
            with st.spinner("Indexing PDF for semantic search..."):
                pages = extract_pages_from_pdf(uploaded_file)
                if pages:
                    retriever = SemanticRetriever()
                    collection_name = retriever.index_pages(pages)
                    st.session_state["collection_name"] = collection_name
                    st.session_state["pdf_name"] = uploaded_file.name
                    st.session_state["pdf_pages"] = len(pages)
                else:
                    st.session_state["collection_name"] = None
                    st.session_state["pdf_name"] = uploaded_file.name

        if st.session_state["collection_name"]:
            st.success(
                f"PDF indexed successfully ({st.session_state['pdf_pages']} pages)."
            )
        else:
            st.error(
                "No extractable text was found. Scanned/image-only PDFs are not "
                "supported yet."
            )

    user_query = st.text_input("Ask a question about the PDF")

    if st.button("Ask", type="primary"):
        collection_name = st.session_state.get("collection_name")
        if not collection_name:
            st.warning("Please upload a readable PDF first.")
        elif not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching the PDF..."):
                retriever = SemanticRetriever()
                chunks = retriever.retrieve(collection_name, user_query, top_k=4)

            llm = OllamaProvider() if use_local_llm else None
            engine = GroundedAnswerEngine(llm=llm)

            with st.spinner("Generating grounded answer..."):
                result = engine.answer(user_query, chunks)

            st.subheader("Answer")
            st.write(result.text)

            if result.sources:
                st.caption(
                    "Sources: " + ", ".join(f"Page {page}" for page in result.sources)
                )
            if not result.used_llm:
                st.info(
                    "Running in zero-cost retrieval mode. Enable local Ollama above "
                    "for natural-language answer generation."
                )


if __name__ == "__main__":
    main()
