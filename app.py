import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="📄 AI Q&A Copilot", layout="centered")
st.title("📄 AI Q&A Chatbot (Copilot Style)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and openai_api_key:
    st.info("Processing your document...")

    # Load and split PDF
    loader = PyPDFLoader(uploaded_file)
    pages = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    # Embed and store in FAISS
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Set up QA chain
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(api_key=openai_api_key),
        retriever=vectorstore.as_retriever()
    )

    # User question
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Finding answer..."):
            result = qa.run(question)
            st.success(result)
else:
    st.warning("Upload a PDF and ensure your API key is set.")
