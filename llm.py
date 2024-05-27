import streamlit as st
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import FAISS
from typing import Generator


def handle_user_query(query: str, vectorstore: FAISS) -> Generator[str, None, None]:
    """
    Handle a user query by retrieving relevant documents and generating a response using LLM.
    
    :param query: User query
    :param vectorstore: FAISS vector store
    :return: Updated chat history
    """
    if query and vectorstore:
        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        def format_docs(docs: list) -> str:
            """
            Format the retrieved documents into a string.
            
            :param docs: List of retrieved documents
            :return: Formatted string of documents
            """
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        for chunk in rag_chain.stream(query):
            yield chunk
