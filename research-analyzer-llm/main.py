import streamlit as st
from document_parser import DocumentParser
from embeddings_indexer import EmbeddingsIndexer
from question_answering import QuestionAnsweringSystem
import os
llm_model = "meta/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf"

def main():
    st.title("Research Paper Analyzer")
    user_query = st.text_input("Enter your question", "")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with open("temp_uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        parsed_text = DocumentParser.extract_text_from_pdf(
            "temp_uploaded_file.pdf")
        structured_document = DocumentParser.structured_document(parsed_text)
        indexer = EmbeddingsIndexer()
        texts_to_index = [section['text']
                          for section in structured_document.values()]
        indexer.add_documents(texts_to_index)
        qa_system = QuestionAnsweringSystem(indexer, llm_model)

        if st.button("Analyze"):
            if user_query:
                response = qa_system.answer_query(user_query)
                st.write("Response:")
                st.json(response)
            else:
                st.write("Please enter a query.")
        os.remove("temp_uploaded_file.pdf")


if __name__ == "__main__":
    main()
