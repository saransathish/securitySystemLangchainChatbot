import streamlit as st
import pandas as pd
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import Dict, List, Any, Optional
import json
from pydantic import Field

class RiskAnalyzerTool(BaseTool):
    name: str = "risk_analyzer"
    description: str = "Analyzes security risks based on survey responses"
    data_processor: Any = Field(default=None)
    
    def _run(self, answers: str) -> List[str]:
        answers_dict = json.loads(answers)
        return self.data_processor.analyze_risks(answers_dict)
    
    async def _arun(self, answers: str) -> List[str]:
        raise NotImplementedError("Async not implemented")

class MitigationTool(BaseTool):
    name: str = "mitigation_finder"
    description: str = "Finds comprehensive mitigation steps for identified risks"
    data_processor: Any = Field(default=None)
    
    def _run(self, risk_type: str) -> Dict:
        return self.data_processor.get_mitigation_steps(risk_type)
    
    async def _arun(self, risk_type: str) -> Dict:
        raise NotImplementedError("Async not implemented")

class AssuranceMetricsTool(BaseTool):
    name: str = "assurance_metrics"
    description: str = "Gets detailed assurance metrics for security solutions"
    data_processor: Any = Field(default=None)
    
    def _run(self, solution: str) -> Dict:
        return self.data_processor.get_solution_details(solution)
    
    async def _arun(self, solution: str) -> Dict:
        raise NotImplementedError("Async not implemented")

class DataProcessor:
    def __init__(self):
        try:
            self.survey_data = pd.read_excel("assumption.xlsx", sheet_name="Survey Questions")
            self.risk_matrix = pd.read_excel("assumption.xlsx", sheet_name="Risk > Mitigation Matrix")
            self.assurance_matrix = pd.read_excel("assumption.xlsx", sheet_name="Assurance Metrics")
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
            raise

    def analyze_risks(self, answers: Dict[str, str]) -> List[str]:
        identified_risks = set()
        for question, answer in answers.items():
            if answer.lower() in ['no', 'n']:
                question_data = self.survey_data[
                    self.survey_data['Question'] == question
                ]
                if not question_data.empty and pd.notna(question_data.iloc[0]['Risk Present']):
                    risks = question_data.iloc[0]['Risk Present'].split(',')
                    identified_risks.update([risk.strip() for risk in risks])
        return list(identified_risks)

    def get_mitigation_steps(self, risk_type: str) -> Dict[str, Any]:
        try:
            risk_data = self.risk_matrix[self.risk_matrix['Risk Type'] == risk_type].iloc[0]
            mitigations = {
                'tech': [x.strip() for x in risk_data['Tech Mitigation'].split(',')] if pd.notna(risk_data['Tech Mitigation']) else [],
                'human': [x.strip() for x in risk_data['Human Mitigation'].split(',')] if pd.notna(risk_data['Human Mitigation']) else [],
                'tss': [x.strip() for x in risk_data['TSS Mitigation'].split(',')] if pd.notna(risk_data['TSS Mitigation']) else [],
                'analytics': [x.strip() for x in risk_data['Analytics Mitigation'].split(',')] if pd.notna(risk_data['Analytics Mitigation']) else [],
                'policy': [x.strip() for x in risk_data['Policy Mitigation'].split(',')] if pd.notna(risk_data['Policy Mitigation']) else []
            }
            
            # Get solution details for each mitigation
            solution_details = {}
            all_mitigations = []
            for mitigation_list in mitigations.values():
                all_mitigations.extend(mitigation_list)
            
            for mitigation in all_mitigations:
                solution_data = self.get_solution_details(mitigation)
                if solution_data:
                    solution_details[mitigation] = solution_data
            
            return {
                'mitigations': mitigations,
                'solution_details': solution_details
            }
        except Exception as e:
            st.error(f"Error getting mitigation steps: {str(e)}")
            return {
                'mitigations': {
                    'tech': [], 'human': [], 'tss': [], 
                    'analytics': [], 'policy': []
                },
                'solution_details': {}
            }

    def get_solution_details(self, solution_name: str) -> Dict[str, Any]:
        try:
            solution_data = self.assurance_matrix[
                self.assurance_matrix['Solution'] == solution_name
            ]
            
            if solution_data.empty:
                return None
                
            data = solution_data.iloc[0]
            return {
                'use_case': data['Use case'] if pd.notna(data['Use case']) else "",
                'links': data['Links to use case'] if pd.notna(data['Links to use case']) else "",
                'partners': data['Partner(s)'] if pd.notna(data['Partner(s)']) else "",
                'data_format': data['Data Format'] if pd.notna(data['Data Format']) else "",
                'immediate_actions': [x.strip() for x in str(data['Data type (Immediate Action)']).split(',')] if pd.notna(data['Data type (Immediate Action)']) else [],
                'data_collation': [x.strip() for x in str(data['Data type (Data Collation)']).split(',')] if pd.notna(data['Data type (Data Collation)']) else [],
                'dashboard': [x.strip() for x in str(data['Eco System outputs/results - Dashboard']).split(',')] if pd.notna(data['Eco System outputs/results - Dashboard']) else [],
                'wearable': [x.strip() for x in str(data['Eco System outputs/results - Wearable']).split(',')] if pd.notna(data['Eco System outputs/results - Wearable']) else [],
                'mobile': [x.strip() for x in str(data['Eco System outputs/results - Mobile']).split(',')] if pd.notna(data['Eco System outputs/results - Mobile']) else [],
                'soc': [x.strip() for x in str(data['Eco System outputs/results - SOC']).split(',')] if pd.notna(data['Eco System outputs/results - SOC']) else [],
                'audio_visual': [x.strip() for x in str(data['Eco System outputs/results - Audio/Visual']).split(',')] if pd.notna(data['Eco System outputs/results - Audio/Visual']) else []
            }
        except Exception as e:
            st.error(f"Error getting solution details: {str(e)}")
            return None

class RiskAssessmentChat:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.llm = Ollama(model="llama2")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.current_question_idx = 0
        self.answers = {}
        self.setup_tools()
        self.setup_agent()

    def setup_tools(self):
        self.risk_analyzer = RiskAnalyzerTool(data_processor=self.data_processor)
        self.mitigation_tool = MitigationTool(data_processor=self.data_processor)
        self.assurance_tool = AssuranceMetricsTool(data_processor=self.data_processor)
        
        self.tools = [self.risk_analyzer, self.mitigation_tool, self.assurance_tool]

    def setup_agent(self):
        prompt = PromptTemplate.from_template(
            """You are a security risk assessment expert. Use the available tools to analyze risks 
            and provide recommendations.

            Current conversation:
            {chat_history}

            Human: {input}
            Assistant: Let me help you with that analysis.

            Available Tools:
            {tools}

            {agent_scratchpad}

            Tool Names: {tool_names}
            """
        )

        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    def get_next_question(self) -> Optional[str]:
        if self.current_question_idx < len(self.data_processor.survey_data):
            return self.data_processor.survey_data.iloc[self.current_question_idx]['Question']
        return None

    def process_answer(self, answer: str) -> None:
        question = self.data_processor.survey_data.iloc[self.current_question_idx]['Question']
        self.answers[question] = answer
        self.current_question_idx += 1

    def generate_report(self) -> Dict[str, Any]:
        risks = self.data_processor.analyze_risks(self.answers)
        report = {
            'identified_risks': []
        }

        for risk in risks:
            risk_data = {
                'risk_type': risk,
                'mitigations': self.data_processor.get_mitigation_steps(risk)
            }
            report['identified_risks'].append(risk_data)

        return report

def main():
    st.set_page_config(page_title="Security Risk Assessment", layout="wide")
    st.title("Security Risk Assessment System")
    
    # Initialize session state
    if 'chat' not in st.session_state:
        st.session_state.chat = RiskAssessmentChat()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get next question or show report
    question = st.session_state.chat.get_next_question()

    if question:
        with st.chat_message("assistant"):
            st.markdown(f"**Question**: {question}")
            st.session_state.messages.append({"role": "assistant", "content": question})

        user_answer = st.chat_input("Your answer (Y/N):")

        if user_answer:
            if user_answer.upper() not in ['Y', 'N', 'YES', 'NO']:
                st.error("Please answer with Y or N")
                return

            with st.chat_message("user"):
                st.markdown(user_answer)
                st.session_state.messages.append({"role": "user", "content": user_answer})

            st.session_state.chat.process_answer(user_answer)
            st.rerun()

    elif not st.session_state.report_generated:
        report = st.session_state.chat.generate_report()
        
        with st.chat_message("assistant"):
            st.markdown("# Comprehensive Security Risk Assessment Report")

            if not report['identified_risks']:
                st.success("No security risks were identified based on your responses.")
            else:
                for risk_data in report['identified_risks']:
                    risk_type = risk_data['risk_type']
                    full_data = risk_data['mitigations']
                    mitigations = full_data['mitigations']
                    solution_details = full_data['solution_details']

                    st.markdown(f"## Risk: {risk_type}")
                    
                    tabs = st.tabs(["Mitigations", "Solution Details"])
                    
                    with tabs[0]:
                        cols = st.columns(3)
                        
                        with cols[0]:
                            if mitigations['tech']:
                                st.markdown("### Technical Mitigations")
                                for tech in mitigations['tech']:
                                    st.markdown(f"- {tech}")
                            
                            if mitigations['human']:
                                st.markdown("### Human Mitigations")
                                for human in mitigations['human']:
                                    st.markdown(f"- {human}")
                        
                        with cols[1]:
                            if mitigations['tss']:
                                st.markdown("### TSS Mitigations")
                                for tss in mitigations['tss']:
                                    st.markdown(f"- {tss}")
                            
                            if mitigations['analytics']:
                                st.markdown("### Analytics Mitigations")
                                for analytics in mitigations['analytics']:
                                    st.markdown(f"- {analytics}")
                        
                        with cols[2]:
                            if mitigations['policy']:
                                st.markdown("### Policy Mitigations")
                                for policy in mitigations['policy']:
                                    st.markdown(f"- {policy}")
                    
                    with tabs[1]:
                        for solution_name, details in solution_details.items():
                            if details:
                                st.markdown(f"### Solution: {solution_name}")
                                
                                if details['use_case']:
                                    st.markdown(f"**Use Case:** {details['use_case']}")
                                
                                if details['links']:
                                    st.markdown(f"**Reference Links:** {details['links']}")
                                
                                if details['partners']:
                                    st.markdown(f"**Partners:** {details['partners']}")
                                
                                cols = st.columns(2)
                                
                                with cols[0]:
                                    if details['immediate_actions']:
                                        st.markdown("#### Immediate Actions")
                                        for action in details['immediate_actions']:
                                            st.markdown(f"- {action}")
                                    
                                    if details['dashboard']:
                                        st.markdown("#### Dashboard Features")
                                        for feature in details['dashboard']:
                                            st.markdown(f"- {feature}")
                                    
                                    if details['mobile']:
                                        st.markdown("#### Mobile Features")
                                        for feature in details['mobile']:
                                            st.markdown(f"- {feature}")
                                
                                with cols[1]:
                                    if details['data_collation']:
                                        st.markdown("#### Data Collation")
                                        for data in details['data_collation']:
                                            st.markdown(f"- {data}")
                                    
                                    if details['soc']:
                                        st.markdown("#### SOC Features")
                                        for feature in details['soc']:
                                            st.markdown(f"- {feature}")
                                    
                                    if details['audio_visual']:
                                        st.markdown("#### Audio/Visual Features")
                                        for feature in details['audio_visual']:
                                            st.markdown(f"- {feature}")

        st.session_state.report_generated = True

if __name__ == "__main__":
    main()