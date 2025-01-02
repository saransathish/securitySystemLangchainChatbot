import streamlit as st
import os
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub

import pandas as pd
from typing import List, Dict
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
import numpy as np

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

class SecurityVulnerabilityBot:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.sheets_data = {}
        self.vector_stores = {}
        self.load_excel_data()
        self.setup_model()
        self.setup_memory()
        self.setup_tools()
        self.setup_agent()

    def load_excel_data(self):
        """Load all sheets from Excel file into DataFrames"""
        excel_file = pd.ExcelFile(self.excel_path)
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            self.sheets_data[sheet_name] = df
            
            # For vulnerability data sheet
            if sheet_name == "Vulnerability Data":
                # Assuming the first column contains questions, regardless of its name
                question_column = df.columns[0]
                
                # Convert DataFrame to documents format
                documents = []
                for idx, row in df.iterrows():
                    content = row[question_column]
                    metadata = {"row_index": idx}
                    documents.append({
                        "page_content": content,
                        "metadata": metadata
                    })
                
                # Create embeddings and vector store
                embeddings = HuggingFaceEmbeddings()
                texts = [doc["page_content"] for doc in documents]
                metadatas = [doc["metadata"] for doc in documents]
                self.vector_stores[sheet_name] = FAISS.from_texts(
                    texts=texts,
                    embedding=embeddings,
                    metadatas=metadatas
                )

    def setup_model(self):
        """Initialize Mistral model"""
        # Option 1: Using HuggingFaceHub
        self.llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            model_kwargs={"temperature": 0.7, "max_new_tokens": 2000}
        )

    def setup_memory(self):
        """Setup conversation memory"""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=ChatMessageHistory()

        )

    def analyze_vulnerability(self, question: str, sheet_name: str) -> str:
        """Analyze vulnerability based on data in specified sheet"""
        df = self.sheets_data[sheet_name]
        
        # Find most relevant row based on question
        retriever = self.vector_stores[sheet_name].as_retriever()
        relevant_docs = retriever.get_relevant_documents(question)
        
        if not relevant_docs:
            return "No relevant information found."
            
        row_data = relevant_docs[0]
        question_text = row_data.page_content
        
        # Get corresponding row from DataFrame
        row = df[df['Question'] == question_text].iloc[0]
        
        # Analyze vulnerabilities
        vulnerabilities = []
        risks = []
        
        for col in df.columns[1:]:  # Skip 'Question' column
            if row[col] == 'X':
                vulnerabilities.append(f"Vulnerability found in {col}")
                risks.append(self.get_risk_level(col))
                
        return self.format_analysis(vulnerabilities, risks, question_text)

    def get_risk_level(self, category: str) -> str:
        """Get risk level from Risk Matrix sheet"""
        risk_matrix = self.sheets_data.get('Risk Matrix', pd.DataFrame())
        if not risk_matrix.empty:
            risk_row = risk_matrix[risk_matrix['Category'] == category]
            if not risk_row.empty:
                return risk_row.iloc[0]['Risk Level']
        return "Unknown"

    def format_analysis(self, vulnerabilities: List[str], risks: List[str], question: str) -> str:
        """Format the vulnerability analysis response"""
        response = f"Analysis for: {question}\n\n"
        response += "Identified Vulnerabilities:\n"
        
        for vuln, risk in zip(vulnerabilities, risks):
            response += f"- {vuln} (Risk Level: {risk})\n"
            
        if not vulnerabilities:
            response += "No immediate vulnerabilities detected. However, continuous monitoring is recommended.\n"
            
        # Add recommendations
        response += "\nRecommendations:\n"
        for vuln, risk in zip(vulnerabilities, risks):
            if risk == "High":
                response += f"- URGENT: Immediate action required for {vuln}\n"
            elif risk == "Medium":
                response += f"- IMPORTANT: Plan mitigation strategies for {vuln}\n"
            else:
                response += f"- MONITOR: Keep tracking {vuln}\n"
            
        return response

    def setup_tools(self):
        """Setup tools for the agent"""
        self.tools = [
            Tool(
                name="Vulnerability Analyzer",
                func=lambda q: self.analyze_vulnerability(q, "Vulnerability Data"),
                description="Analyzes security vulnerabilities based on the question"
            ),
            Tool(
                name="Risk Matrix Lookup",
                func=self.get_risk_level,
                description="Looks up risk levels for different categories"
            )
        ]

    def setup_agent(self):
        """Setup the ReAct agent"""
        prompt = PromptTemplate.from_template("""
        You are a security vulnerability analysis expert. Analyze the following question:
        {question}
        
        Use the available tools to:
        1. Identify vulnerabilities
        2. Assess risk levels
        3. Provide recommendations
        
        Previous conversation context:
        {chat_history}
        
        Let's approach this step by step:
        """)

        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

    def chat(self, user_input: str) -> str:
        """Process user input and return response"""
        try:
            response = self.agent_executor.run(user_input)
            return response
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Streamlit Interface Functions
def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'bot' not in st.session_state:
        st.session_state.bot = None
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False

def display_chat_history():
    """Display chat history with custom styling"""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f'👤 **You:** {message["content"]}')
        else:
            st.write(f'🤖 **Bot:** {message["content"]}')

def display_metrics():
    """Display security metrics and statistics"""
    if st.session_state.bot and hasattr(st.session_state.bot, 'sheets_data'):
        vuln_data = st.session_state.bot.sheets_data.get('Vulnerability Data', pd.DataFrame())
        if not vuln_data.empty:
            total_checks = len(vuln_data)
            implemented = vuln_data.iloc[:, 1:].apply(lambda x: x == 'X').sum().sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Security Checks", total_checks)
            with col2:
                st.metric("Implemented Measures", int(implemented))
            with col3:
                implementation_rate = (implemented / (total_checks * (len(vuln_data.columns) - 1))) * 100
                st.metric("Implementation Rate", f"{implementation_rate:.1f}%")

def main():
    st.set_page_config(
        page_title="Security Vulnerability Analysis",
        page_icon="🔒",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    # Sidebar
    st.sidebar.title("📊 Configuration")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File",
        type=['xlsx', 'xls']
    )

    # Main content
    st.title(" Security Vulnerability Analysis System")
    
    if uploaded_file:
        if not st.session_state.file_uploaded:
            # Save the uploaded file temporarily
            with open("temp_security_data.xlsx", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Initialize the bot
            try:
                st.session_state.bot = SecurityVulnerabilityBot("temp_security_data.xlsx")
                st.session_state.file_uploaded = True
                st.success("✅ Security data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return

        # Display metrics
        display_metrics()
        
        # Display file info
        st.sidebar.success("File loaded successfully!")
        
        # Display available sheets
        if st.session_state.bot and hasattr(st.session_state.bot, 'sheets_data'):
            st.sidebar.subheader("Available Data Sheets")
            for sheet_name in st.session_state.bot.sheets_data.keys():
                st.sidebar.info(f"📑 {sheet_name}")

        # Chat interface
        st.subheader("💬 Chat Interface")
        
        # Display chat history
        display_chat_history()
        
        # Chat input
        user_input = st.text_input("Ask about security vulnerabilities...", key="user_input")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            # Send button
            if st.button("Send", key="send"):
                if user_input:
                    # Add user message to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Get bot response
                    try:
                        response = st.session_state.bot.chat(user_input)
                        # Add bot response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response
                        })
                    except Exception as e:
                        st.error(f"Error getting response: {str(e)}")
                    
                    # Clear input
                    st.session_state.user_input = ""
                    
                    # Rerun to update chat display
                    st.experimental_rerun()
            
            # Clear chat button
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.experimental_rerun()

    else:
        # Display instructions when no file is uploaded
        st.info("""
         Welcome to the Security Vulnerability Analysis System!
        
        To get started:
        1. Upload your Excel file using the sidebar
        2. The file should contain:
           - Vulnerability Data sheet
           - Risk Matrix sheet
        3. Once uploaded, you can start asking questions about security vulnerabilities
        
        Example questions:
        - What are the vulnerabilities in our CCTV system?
        - Are there any risks in our access control?
        - What security measures are missing?
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p style='color: gray;'>Security Vulnerability Analysis System | Built with Streamlit & LangChain</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()