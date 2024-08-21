import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_text):
    text=""
    for pdf in pdf_text: 
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template="""
    Answer the question  as detailed as possible from the provided  context, make sure to provide all the  details.if the answer is not in
    the context,just say,"answer is not available in the context", don't provide  the wrong answer\n\n
    Context:\n,{context}?\n
    Question:\n,{question}?\n

    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",temperature=0.3)

    prompt=PromptTemplate(template=prompt_template,input_variable=["context","question"])
    
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain= get_conversational_chain()

    response = chain(
        {"input_documents":docs,"question":user_question},
        return_only_outputs=True)
    
    print(response)
    st.write("Reply: ",response["output_text"])

def main():
    st.title("Chat with Multiple PDF using Gemini")
    
    # Step 1: Upload PDF
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        # Step 2: Extract text from PDF
        with st.spinner("Extracting text from PDFs..."):
            pdf_text = get_pdf_text(uploaded_files)
            text_chunks = get_text_chunks(pdf_text)
            get_vector_store(text_chunks)
            st.success("Text extracted and indexed successfully!")
        
        # Step 3: User input for questions
        user_question = st.text_input("Ask a question about the content of the PDFs:")
        
        if user_question:
            # Step 4: Process the user question and get the response
            with st.spinner("Generating response..."):
                user_input(user_question)
    
if __name__ == "__main__":
    main()













