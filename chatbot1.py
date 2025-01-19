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
        return self.data_processor.get_assurance_metrics(solution)
    
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
            if answer.lower() == 'no':
                question_data = self.survey_data[
                    self.survey_data['Question'] == question
                ]
                if not question_data.empty and pd.notna(question_data.iloc[0]['Risk Present']):
                    risks = question_data.iloc[0]['Risk Present'].split(',')
                    identified_risks.update([risk.strip() for risk in risks])
        return list(identified_risks)

    def get_mitigation_steps(self, risk_type: str) -> Dict[str, List[str]]:
        try:
            risk_data = self.risk_matrix[self.risk_matrix['Risk Type'] == risk_type].iloc[0]
            return {
                'tech': [x.strip() for x in risk_data['Tech Mitigation'].split(',')] if pd.notna(risk_data['Tech Mitigation']) else [],
                'human': [x.strip() for x in risk_data['Human Mitigation'].split(',')] if pd.notna(risk_data['Human Mitigation']) else [],
                'tss': [x.strip() for x in risk_data['TSS Mitigation'].split(',')] if pd.notna(risk_data['TSS Mitigation']) else [],
                'analytics': [x.strip() for x in risk_data['Analytics Mitigation'].split(',')] if pd.notna(risk_data['Analytics Mitigation']) else [],
                'policy': [x.strip() for x in risk_data['Policy Mitigation'].split(',')] if pd.notna(risk_data['Policy Mitigation']) else []
            }
        except Exception:
            return {
                'tech': [], 'human': [], 'tss': [], 
                'analytics': [], 'policy': []
            }

    def get_assurance_metrics(self, solution: str) -> Dict[str, Any]:
        try:
            solution_data = self.assurance_matrix[
                self.assurance_matrix['Solution'] == solution
            ].iloc[0]
            return {
                'data_format': solution_data['Data Format'] if pd.notna(solution_data['Data Format']) else "",
                'immediate_action': [x.strip() for x in solution_data['Data type (Immediate Action)'].split(',')] if pd.notna(solution_data['Data type (Immediate Action)']) else [],
                'data_collation': [x.strip() for x in solution_data['Data type (Data Collation)'].split(',')] if pd.notna(solution_data['Data type (Data Collation)']) else [],
                'dashboard': [x.strip() for x in solution_data['Eco System outputs/results - Dashboard'].split(',')] if pd.notna(solution_data['Eco System outputs/results - Dashboard']) else [],
                'mobile': [x.strip() for x in solution_data['Eco System outputs/results - Mobile'].split(',')] if pd.notna(solution_data['Eco System outputs/results - Mobile']) else [],
                'soc': [x.strip() for x in solution_data['Eco System outputs/results - SOC'].split(',')] if pd.notna(solution_data['Eco System outputs/results - SOC']) else []
            }
        except Exception:
            return {
                'data_format': "",
                'immediate_action': [],
                'data_collation': [],
                'dashboard': [],
                'mobile': [],
                'soc': []
            }

# [Previous imports and tool classes remain the same...]

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
        risk_analyzer = RiskAnalyzerTool(data_processor=self.data_processor)
        mitigation_tool = MitigationTool(data_processor=self.data_processor)
        assurance_tool = AssuranceMetricsTool(data_processor=self.data_processor)
        
        self.tools = [risk_analyzer, mitigation_tool, assurance_tool]

    def setup_agent(self):
        prompt = PromptTemplate.from_template(
            """You are a security risk assessment expert. Use the available tools to analyze risks 
            and provide recommendations.

            Current conversation:
            {chat_history}

            Human: {input}
            Assistant: Let me help you with that analysis.

            {agent_scratchpad}

            Tools:
            {tools}

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
            verbose=True
        )

    # [Rest of the RiskAssessmentChat class and main() function remain the same...]
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
            'identified_risks': [],
            'mitigation_steps': {},
            'assurance_metrics': {}
        }

        for risk in risks:
            risk_data = {
                'risk_type': risk,
                'mitigations': self.data_processor.get_mitigation_steps(risk)
            }
            report['identified_risks'].append(risk_data)

            risk_row = self.data_processor.risk_matrix[
                self.data_processor.risk_matrix['Risk Type'] == risk
            ]
            
            if not risk_row.empty and pd.notna(risk_row.iloc[0]['Tech Mitigation']):
                risk_solutions = risk_row.iloc[0]['Tech Mitigation'].split(',')
                for solution in risk_solutions:
                    solution = solution.strip()
                    if solution and solution not in report['assurance_metrics']:
                        report['assurance_metrics'][solution] = self.data_processor.get_assurance_metrics(solution)

        return report

def main():
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
        # Display question
        with st.chat_message("assistant"):
            st.markdown(f"**Question**: {question}")
            st.session_state.messages.append({"role": "assistant", "content": question})

        # Get user input
        user_answer = st.chat_input("Your answer (Y/N):")

        if user_answer:
            # Validate input
            if user_answer.upper() not in ['Y', 'N', 'YES', 'NO']:
                st.error("Please answer with Y or N")
                return

            # Display and process answer
            with st.chat_message("user"):
                st.markdown(user_answer)
                st.session_state.messages.append({"role": "user", "content": user_answer})

            st.session_state.chat.process_answer(user_answer)
            st.rerun()

    elif not st.session_state.report_generated:
        # Generate and display final report
        report = st.session_state.chat.generate_report()

        with st.chat_message("assistant"):
            st.markdown("# Security Risk Assessment Report")

            if not report['identified_risks']:
                st.success("No security risks were identified based on your responses.")
            else:
                # Display identified risks and mitigations
                for risk_data in report['identified_risks']:
                    risk_type = risk_data['risk_type']
                    mitigations = risk_data['mitigations']

                    st.markdown(f"## Risk: {risk_type}")
                    
                    tab1, tab2 = st.tabs(["Mitigations", "Assurance Metrics"])
                    
                    with tab1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if mitigations['tech']:
                                st.markdown("### Technical Mitigations")
                                for tech in mitigations['tech']:
                                    st.markdown(f"- {tech}")
                            
                            if mitigations['human']:
                                st.markdown("### Human Mitigations")
                                for human in mitigations['human']:
                                    st.markdown(f"- {human}")
                        
                        with col2:
                            if mitigations['tss']:
                                st.markdown("### TSS Mitigations")
                                for tss in mitigations['tss']:
                                    st.markdown(f"- {tss}")
                            
                            if mitigations['policy']:
                                st.markdown("### Policy Mitigations")
                                for policy in mitigations['policy']:
                                    st.markdown(f"- {policy}")

                    with tab2:
                        for solution, metrics in report['assurance_metrics'].items():
                            if solution:
                                st.markdown(f"### Solution: {solution}")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if metrics['immediate_action']:
                                        st.markdown("#### Immediate Actions")
                                        for action in metrics['immediate_action']:
                                            st.markdown(f"- {action}")
                                
                                with col2:
                                    if metrics['data_collation']:
                                        st.markdown("#### Data Collation")
                                        for data in metrics['data_collation']:
                                            st.markdown(f"- {data}")

        st.session_state.report_generated = True

if __name__ == "__main__":
    main()