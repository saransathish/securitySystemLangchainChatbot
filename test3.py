import streamlit as st
import pandas as pd
import numpy as np
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from typing import Dict, List
import re
import os
from pathlib import Path

# Custom prompt template for final response formatting
RESPONSE_TEMPLATE = """
Based on the provided context and tools, provide a clear and professional response to the user's question.
Please format the response in a clean, readable way without showing the internal thought process.

Context about the store:
Store: {store_name}
Address: {address}
Postcode: {postcode}

Question: {question}

Available Information:
{tool_output}

Please provide a clear, professional response:
"""

class SecurityDataTools:
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        self.data = data_dict
    
    def get_survey_questions(self, category: str = None) -> str:
        """Get survey questions, optionally filtered by category."""
        try:
            df = self.data['Survey Questions']
            if category and category.lower() != "all":
                df = df[df['Category'].str.contains(category, case=False, na=False)]
            result = df.to_dict('records')
            return "\n".join([f"Category: {r['Category']}, Question: {r['Question']}" for r in result])
        except Exception as e:
            return f"Error retrieving survey questions: {str(e)}"
    
    def get_likelihood_risks(self, category: str = None) -> str:
        """Get likelihood risks from the risk matrix."""
        try:
            df = self.data['Likelihood > Risk Matrix']
            if category and category.lower() != "all":
                df = df[df['Category'].str.contains(category, case=False, na=False)]
            result = df.to_dict('records')
            return "\n".join([f"Category: {r['Category']}, Risk: {r['Risks Present']}" for r in result])
        except Exception as e:
            return f"Error retrieving likelihood risks: {str(e)}"
    
    def get_vulnerabilities(self, item: str = None) -> str:
        """Get vulnerabilities, optionally filtered by item."""
        try:
            df = self.data['Vulnerability > Risk Matrix']
            if item and item.lower() != "all":
                df = df[df['Item'].str.contains(item, case=False, na=False)]
            result = df.to_dict('records')
            return "\n".join([f"Item: {r['Item']}, Vulnerabilities: {r['Vulnerabilities Present']}" for r in result])
        except Exception as e:
            return f"Error retrieving vulnerabilities: {str(e)}"

def create_vector_store(data: Dict[str, pd.DataFrame], embeddings) -> FAISS:
    """Create FAISS vector store from DataFrame."""
    texts = []
    metadatas = []
    
    for sheet_name, df in data.items():
        for idx, row in df.iterrows():
            text = f"Sheet: {sheet_name}\n" + \
                   "\n".join([f"{col}: {str(val)}" for col, val in row.items() if pd.notna(val)])
            texts.append(text)
            metadatas.append({
                "source": sheet_name,
                "row_index": idx,
                "original_data": row.to_dict()
            })
    
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

def clean_agent_output(output: str) -> str:
    """Clean the agent output to remove thought process and action steps."""
    cleaned = re.sub(r'Thought:.*?Action:', '', output, flags=re.DOTALL)
    cleaned = re.sub(r'Action:.*?Action Input:', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'Action Input:.*?Observation:', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'Observation:.*?Final Answer:', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'Final Answer:', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    return cleaned.strip()

def format_final_response(llm, store_info: dict, question: str, tool_output: str) -> str:
    """Format the final response using a template."""
    prompt = PromptTemplate(
        template=RESPONSE_TEMPLATE,
        input_variables=["store_name", "address", "postcode", "question", "tool_output"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(
        store_name=store_info['store_name'],
        address=store_info['address'],
        postcode=store_info['postcode'],
        question=question,
        tool_output=tool_output
    )
    
    return response

def setup_tools_and_agent(llm, data_tools, vector_store):
    """Setup tools and create agent with proper configuration."""
    tools = [
        Tool(
            name="SurveyQuestions",
            func=data_tools.get_survey_questions,
            description="Use this to get security survey questions. Input should be a category or 'all'"
        ),
        Tool(
            name="LikelihoodRisks",
            func=data_tools.get_likelihood_risks,
            description="Use this to get likelihood risks. Input should be a category or 'all'"
        ),
        Tool(
            name="Vulnerabilities",
            func=data_tools.get_vulnerabilities,
            description="Use this to get security vulnerabilities. Input should be an item or 'all'"
        )
    ]
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    
    tools.append(
        Tool(
            name="SecurityKnowledgeBase",
            func=qa_chain.run,
            description="Use this for general security management questions"
        )
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=3
    )

# Initialize Streamlit app
def main():
    st.title("Security Management RAG Chatbot")
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize system components
    if not st.session_state.initialized:
        with st.spinner("Initializing system..."):
            try:
                # Initialize LLM and embeddings
                llm = Ollama(model="llama2")
                embeddings = HuggingFaceEmbeddings()
                
                # Load Excel data
                excel_path = "assumption.xlsx"
                if not os.path.exists(excel_path):
                    st.error(f"Error: {excel_path} not found!")
                    return
                
                data = pd.read_excel(excel_path, sheet_name=None)
                
                # Create vector store
                vector_store = create_vector_store(data, embeddings)
                
                # Setup tools and agent
                data_tools = SecurityDataTools(data)
                agent = setup_tools_and_agent(llm, data_tools, vector_store)
                
                # Store in session state
                st.session_state.llm = llm
                st.session_state.agent = agent
                st.session_state.initialized = True
                st.success("System initialized successfully!")
            
            except Exception as e:
                st.error(f"Initialization error: {str(e)}")
                st.error("Please check your configuration and try again.")
                return
    
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
            
            try:
                with st.spinner("Thinking..."):
                    # Get raw agent response
                    raw_response = st.session_state.agent.run(prompt)
                    
                    # Clean and format the response
                    cleaned_output = clean_agent_output(raw_response)
                    final_response = format_final_response(
                        st.session_state.llm,
                        st.session_state.user_info,
                        prompt,
                        cleaned_output
                    )
                    
                    # Add formatted response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": final_response
                    })
                    
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "I apologize, but I encountered an error. Please try rephrasing your question."
                })

if __name__ == "__main__":
    main()