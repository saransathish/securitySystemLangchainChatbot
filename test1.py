# Import required libraries
import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import json

# Initialize session state
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Load and prepare data
def load_data():
    # In practice, replace this with your actual Excel file
    survey_questions = pd.DataFrame({
        'Category': ['Presence'] * 5,
        'Item': ['Access Control', 'Alarm Monitoring (ARC)', 'ANPR', 'Audio Analytics', 'Bodycam'],
        'Question': [
            'Do you have access control systems in place?',
            'Do you currently benefit from alarm monitoring...',
            'Do you use Automatic Number Plate Recognition ...',
            'Do you currently have audio analytics installed...',
            'Do security staff or store colleagues wear bodycam...'
        ],
        'Answer Structure': ['Y/N'] * 5,
        'Type': ['Vulnerability'] * 5
    })
    
    likelihood_matrix = pd.DataFrame({
        'Category': ['Detailed Police Reporting'] * 5,
        'Risks Present': ['Fraud at POS', 'Arson', 'Arson', 'Criminal Damage', 'Criminal Damage'],
        'Question': [
            'Fraud offences recorded by Action Fraud',
            'Arson endangering life',
            'Arson not endangering life',
            'Criminal damage to a building other than a dwelling',
            'Criminal damage to a dwelling'
        ]
    })
    
    vulnerability_matrix = pd.DataFrame({
        'Category': ['Presence'] * 5,
        'Item': ['Access Control', 'Alarm Monitoring (ARC)', 'ANPR', 'Audio Analytics', 'Bodycam'],
        'Vulnerabilities Present': [
            'Physical Assault (w/ Weapon) - Colleague',
            'Physical Assault (w/ Weapon) - Colleague',
            'Physical Assault (w/ Weapon) - Colleague',
            'Physical Assault (w/ Weapon) - Colleague',
            'Physical Assault (w/ Weapon) - Colleague'
        ]
    })
    
    return {
        'Survey Questions': survey_questions,
        'Likelihood Matrix': likelihood_matrix,
        'Vulnerability Matrix': vulnerability_matrix
    }

# Initialize LLM and embedding model
def init_models():
    llm = Ollama(model="llama2")
    embeddings = HuggingFaceEmbeddings()
    return llm, embeddings

# Create vector store
def create_vector_store(data, embeddings):
    texts = []
    metadata = []
    
    # Convert DataFrames to text chunks with metadata
    for key, df in data.items():
        for _, row in df.iterrows():
            text = f"{key}: {' | '.join([f'{col}: {val}' for col, val in row.items()])}"
            texts.append(text)
            metadata.append({'source': key, 'row': row.to_dict()})
    
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadata)
    return vector_store

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
You are a security management expert assistant. Use the following pieces of context and chat history to answer the question at the end.

Context: {context}

Chat History: {history}

User Info:
Store Name: {store_name}
Address: {address}
Postcode: {postcode}

Question: {question}

Please provide a detailed and professional answer based on the security management context:
"""

# Create QA chain
def create_qa_chain(llm):
    prompt = PromptTemplate(
        input_variables=['context', 'history', 'store_name', 'address', 'postcode', 'question'],
        template=CUSTOM_PROMPT_TEMPLATE
    )
    
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="question"
    )
    
    chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        memory=memory,
        prompt=prompt
    )
    
    return chain

# Main UI
def main():
    st.title("Security Management Chatbot")
    
    # Initialize models and data if not already done
    if 'vector_store' not in st.session_state or st.session_state.vector_store is None:
        llm, embeddings = init_models()
        data = load_data()
        st.session_state.vector_store = create_vector_store(data, embeddings)
        st.session_state.qa_chain = create_qa_chain(llm)
    
    # User information form
    if st.session_state.user_info is None:
        with st.form("user_info_form"):
            store_name = st.text_input("Store Name")
            address = st.text_input("Address")
            postcode = st.text_input("Postcode")
            
            if st.form_submit_button("Submit"):
                st.session_state.user_info = {
                    "store_name": store_name,
                    "address": address,
                    "postcode": postcode
                }
                st.rerun()
    
    # Chat interface
    if st.session_state.user_info is not None:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about security management"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Get relevant documents from vector store
            docs = st.session_state.vector_store.similarity_search(prompt, k=3)
            
            # Generate response
            response = st.session_state.qa_chain({
                "input_documents": docs,
                "question": prompt,
                "store_name": st.session_state.user_info["store_name"],
                "address": st.session_state.user_info["address"],
                "postcode": st.session_state.user_info["postcode"]
            })
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response['output_text']})
            
            # Force streamlit to rerun to update chat history
            st.rerun()

if __name__ == "__main__":
    main()