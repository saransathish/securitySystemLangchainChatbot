import streamlit as st
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from streamlit_chat import message
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
from io import BytesIO

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_excel(file):
    """Process Excel file and create vector store"""
    df = pd.read_excel(file)
    # Convert DataFrame to text format
    documents = []
    for index, row in df.iterrows():
        # Convert each row to a string representation
        text = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
        documents.append(text)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.create_documents(documents)
    
    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# Page configuration
st.set_page_config(page_title="Excel Analysis Chatbot", page_icon="📊")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Streamlit UI
st.title("📊 Excel Analysis Chatbot")
st.subheader("Upload your Excel file and ask questions about your data")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])

if uploaded_file and "file_processed" not in st.session_state:
    with st.spinner("Processing Excel file..."):
        # Process the uploaded file
        vectorstore = process_excel(uploaded_file)
        st.session_state.vectorstore = vectorstore
        st.session_state.file_processed = True
        
        # Initialize Ollama with streaming capability
        llm = Ollama(
            model="llama2",
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.7
        )
        
        # Create conversational retrieval chain
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=True
        )
        
        st.success("File processed successfully!")

# Chat interface
if st.session_state.get("file_processed", False):
    # Chat container
    chat_container = st.container()

    # User input
    with st.container():
        user_input = st.text_input("Ask a question about your data:", key="user_input")

    # Handle user input
    if user_input:
        # Prepare the chat history
        chat_history = [(msg["role"], msg["content"]) for msg in st.session_state.messages]
        
        # Get response from the chain
        response = st.session_state.conversation({"question": user_input, "chat_history": chat_history})
        
        # Add messages to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    # Display chat history
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"user_msg_{i}")
            else:
                message(msg["content"], is_user=False, key=f"ai_msg_{i}")

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Display data analysis tools
    with st.expander("Data Analysis Tools"):
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            st.write("Basic Statistics:")
            st.write(df.describe())
            
            # Column selection for visualization
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column for analysis:", numeric_cols)
                st.write(f"Distribution of {selected_col}:")
                st.bar_chart(df[selected_col].value_counts())

else:
    st.info("Please upload an Excel file to begin analysis.")

# Footer
st.markdown("---")
st.markdown("Built with Langchain and Streamlit")
