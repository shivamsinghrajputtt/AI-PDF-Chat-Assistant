import streamlit as st
import PyPDF2
import re

def extract_text_from_pdf(uploaded_file):
    """PDF se pura text nikaalta hai"""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return ""

def find_answer_from_text(pdf_text, user_query):
    """Simple 'AI' jo query ke according matching lines nikaalta hai"""
    if not pdf_text.strip():
        return "PDF se text nahi mil paya. Ho sakta hai PDF scanned image ho."

    # PDF ke text ko sentences me tod do
    sentences = re.split(r'(?<=[.!?])\s+', pdf_text)
    query_words = [w.lower() for w in re.findall(r'\w+', user_query)]

    scored_sentences = []
    for s in sentences:
        s_lower = s.lower()
        score = sum(1 for w in query_words if w in s_lower)
        if score > 0:
            scored_sentences.append((score, s))

    # agar kuch mila hi nahi
    if not scored_sentences:
        return "Mujhe is question se related kuch khaas nahi mila PDF me."

    # score ke hisaab se sort karo (zyada match upar)
    scored_sentences.sort(reverse=True, key=lambda x: x[0])

    # top 3 sentences le lo
    best_sentences = [s for score, s in scored_sentences[:3]]
    return "\n\n".join(best_sentences)

def main():
    st.title("📄 AI PDF Chat Assistant")
    st.write("PDF upload karo, phir question pucho, main PDF ke andar se relevant text dhoondh ke dunga 🙂")

    uploaded_file = st.file_uploader("Yahan PDF upload karo", type=["pdf"])

    # Session state me text store karenge taaki baar-baar na nikalna pade
    if "pdf_text" not in st.session_state:
        st.session_state["pdf_text"] = ""

    if uploaded_file is not None:
        with st.spinner("PDF se text nikaal raha hoon..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            st.session_state["pdf_text"] = pdf_text

        if pdf_text.strip():
            st.success("PDF load ho gaya ✅ Ab question pucho niche.")
        else:
            st.error("PDF se text nahi nikal paya. Shayad ye scanned PDF hai.")

    user_query = st.text_input("Apna question likho (English/Hindi dono chalega):")

    if st.button("Ask"):
        if not st.session_state["pdf_text"].strip():
            st.warning("Pehle koi PDF upload karo.")
        elif not user_query.strip():
            st.warning("Pehle question likho.")
        else:
            with st.spinner("Soch raha hoon..."):
                answer = find_answer_from_text(st.session_state["pdf_text"], user_query)
            st.subheader("Answer (PDF se related lines):")
            st.write(answer)

if __name__ == "__main__":
    main()
