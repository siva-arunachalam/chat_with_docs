from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings import CustomEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileIOLoader
import streamlit as st

def load_document(file: st.runtime.uploaded_file_manager.UploadedFile) -> list:
    """
    Load a document from an uploaded file.
    
    :param file: Uploaded file
    :return: List of loaded documents
    """
    try:
        loader = UnstructuredFileIOLoader(file)
        docs = loader.load()
        return docs
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return []

def process_document(doc: list) -> list:
    """
    Process a document by splitting it into chunks.
    
    :param doc: List of documents
    :return: List of processed text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(doc)
    return texts

def create_vectorstore(texts: list) -> FAISS:
    """
    Create a vector store from the processed text chunks.
    
    :param texts: List of processed text chunks
    :return: FAISS vector store
    """
    try:
        embeddings = CustomEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None
    
def process_uploaded_files(uploaded_files: list) -> tuple:
    """
    Process the uploaded files and create a vector store.
    
    :param uploaded_files: List of uploaded files
    :return: Tuple containing the list of uploaded files and the created vector store
    """
    uploaded_files_list = []
    vectorstore = None

    if uploaded_files:
        for file in uploaded_files:
            if file not in uploaded_files_list:
                uploaded_files_list.append(file)

        docs = []
        for file in uploaded_files_list:
            file_docs = load_document(file)
            docs.extend(file_docs)

        if docs:
            texts = process_document(docs)
            vectorstore = create_vectorstore(texts)

    return uploaded_files_list, vectorstore