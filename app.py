import streamlit as st

from src.pdf_qa import extract_text_from_pdf, find_answer_from_text


st.set_page_config(page_title="AI PDF Chat Assistant", page_icon="📄")


def main() -> None:
    """Render the Streamlit application."""
    st.title("📄 AI PDF Chat Assistant")
    st.caption("Upload a text-based PDF and ask questions about its contents.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if "pdf_text" not in st.session_state:
        st.session_state["pdf_text"] = ""

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            st.session_state["pdf_text"] = pdf_text

        if pdf_text:
            st.success("PDF loaded successfully. Ask a question below.")
        else:
            st.error(
                "No extractable text was found. Scanned/image-only PDFs are not "
                "supported yet."
            )

    user_query = st.text_input("Ask a question about the PDF")

    if st.button("Ask", type="primary"):
        if not st.session_state["pdf_text"]:
            st.warning("Please upload a PDF first.")
        elif not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Finding relevant content..."):
                answer = find_answer_from_text(
                    st.session_state["pdf_text"], user_query
                )
            st.subheader("Answer")
            st.write(answer)


if __name__ == "__main__":
    main()
