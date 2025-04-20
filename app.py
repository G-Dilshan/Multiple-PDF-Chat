import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
import streamlit as st
load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Read PDFs and Extracts Text
def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Text to Chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Chunks to Vectors
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')   # Save in the local folder as unreadable file for human

# Create conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in provided context just say, 'answer is not available in the context', don't provide the 
    wrong answer.
    Context: {context}?
    Question: {question}

    Answer:
    """
    llm_model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-001', temperature=0.9)

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(llm=llm_model, chain_type='stuff', prompt=prompt)
    return chain

# Implementation Function
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    new_db = FAISS.load_local('faiss_index', embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {'input_documents': docs, 'question': user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response['output_text'])

# Streamlit function
def main():
    st.set_page_config("Chat with Multiple PDF")
    st.header("Chat with Multiple PDF")

    user_question = st.text_input('Ask a Question from the PDF Files')

    if user_question:
        user_input(user_question=user_question)

    with st.sidebar:
        st.title("Menu:")
        # pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit button")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit button", accept_multiple_files=True)
        if (st.button("Submit & Process")) and (pdf_docs is not None and len(pdf_docs) > 0):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs=pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks=text_chunks)
                st.success("Done")
    
    st.markdown(
        """
        <hr style="margin-top: 2em;"/>
        <div style='text-align: center; color: gray; font-size: 0.9em;'>
            © 2025 <strong style="color: #3b82f6;">Built by GD</strong>. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
