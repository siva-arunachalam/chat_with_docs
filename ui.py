import streamlit as st
from data import process_uploaded_files
from llm import handle_user_query


def display_uploaded_files() -> None:
    """Display the names of the uploaded files."""
    if st.session_state.uploaded_files:
        st.sidebar.subheader("Uploaded Documents")
        for file in st.session_state.uploaded_files:
            st.sidebar.write(file.name)

def display_chat() -> None:
    """Display the chat interface and handle user queries."""
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

def init_ui() -> None:
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def main() -> None:
    """Main function to run the Streamlit application."""

    st.set_page_config(page_title="Chat with Documents")
    st.title("Document Chat with LLM")

    # initialize variables
    init_ui()
    
    uploaded_files = st.sidebar.file_uploader("Choose your documents", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.uploaded_files, st.session_state.vectorstore = process_uploaded_files(uploaded_files)
        # display_uploaded_files()

        st.header("Chat with the Documents")
        display_chat()
        
        query = st.chat_input("Ask a question about the documents:")
        if query:
            message = {"role": "user", "content": query}
            st.session_state.chat_history.append(message)

            with st.chat_message("user"):
                st.markdown(query)
                            
            with st.chat_message("assistant"):
                response = st.write_stream(handle_user_query(query, st.session_state.vectorstore))
            assistant_message = {"role": "assistant", "content": response}
            st.session_state.chat_history.append(assistant_message)


if __name__ == "__main__":
    main()