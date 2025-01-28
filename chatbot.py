import streamlit as st
import pandas as pd
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import Dict, List, Any, Optional
import json
from pydantic import Field
from fpdf import FPDF
import base64
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()
# Define tools and data processor (unchanged)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
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
            risk_data = self.risk_matrix[self.risk_matrix['Risk Type'].str.strip() == risk_type.strip()]
            
            if risk_data.empty:
                # Skip warning and return empty mitigations
                return {
                    'mitigations': {
                        'tech': [], 'human': [], 'tss': [], 
                        'analytics': [], 'policy': []
                    },
                    'solution_details': {}
                }
            
            risk_row = risk_data.iloc[0]
            
            mitigations = {
                'tech': [x.strip() for x in str(risk_row['Tech Mitigation']).split(',')] if pd.notna(risk_row['Tech Mitigation']) else [],
                'human': [x.strip() for x in str(risk_row['Human Mitigation']).split(',')] if pd.notna(risk_row['Human Mitigation']) else [],
                'tss': [x.strip() for x in str(risk_row['TSS Mitigation']).split(',')] if pd.notna(risk_row['TSS Mitigation']) else [],
                'analytics': [x.strip() for x in str(risk_row['Analytics Mitigation']).split(',')] if pd.notna(risk_row['Analytics Mitigation']) else [],
                'policy': [x.strip() for x in str(risk_row['Policy Mitigation']).split(',')] if pd.notna(risk_row['Policy Mitigation']) else []
            }
            
            for key in mitigations:
                mitigations[key] = [item for item in mitigations[key] if item and item != 'nan']
            
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

# RiskAssessmentChat Class (previously missing)
class RiskAssessmentChat:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.llm = ChatMistralAI(
            mistral_api_key=MISTRAL_API_KEY,
            model="mistral-large")  # or "mistral-small" or "mistral-medium" based on your needs
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.current_question_idx = 0
        self.answers = {}
        self.store_info = StoreInformation()
        self.setup_tools()
        self.setup_agent()
        self.state = "store_info"

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

    def generate_quick_report(self) -> Dict[str, Any]:
        risks = self.data_processor.analyze_risks(self.answers)
        unique_solutions = set()
        unique_risks = set()
        
        report = {
            'identified_risks': [],
            'unique_solutions': [],
            'risk_summary': ''
        }
        
        for risk in risks:
            unique_risks.add(risk)
            risk_data = self.data_processor.get_mitigation_steps(risk)
            for category in risk_data['mitigations'].values():
                unique_solutions.update(category)
        
        report['identified_risks'] = list(unique_risks)
        report['unique_solutions'] = list(unique_solutions)
        report['risk_summary'] = f"Analysis identified {len(unique_risks)} risks with {len(unique_solutions)} possible solutions."
        
        return report

    def generate_detailed_report(self) -> Dict[str, Any]:
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
    
    def generate_pdf_report(self, report_type="detailed") -> str:
        if report_type == "detailed":
            report = self.generate_detailed_report()
        else:
            report = self.generate_quick_report()
            
        pdf_generator = PDFReport()
        pdf_generator.add_title("Security Risk Assessment Report")
        pdf_generator.add_content(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if report_type == "detailed":
            pdf_generator.add_detailed_report(report, self.store_info.store_data, self.answers)
        else:
            pdf_generator.add_quick_report(report, self.store_info.store_data, self.answers)
        
        return pdf_generator.get_download_link(report_type)

# StoreInformation Class
class StoreInformation:
    def __init__(self):
        self.store_fields = [
            {"field": "Store Name", "question": "What is your store name?"},
            {"field": "Store Identifier", "question": "What is your store identifier?"},
            {"field": "Address", "question": "What is the store address?"},
            {"field": "Postcode", "question": "What is the store postcode?"},
            {"field": "Store Format", "question": "What format is the store? (Superstore/Convenience store/Department store)"},
            {"field": "Location Footprint", "question": "What is the store's location footprint? (Retail park/Shopping centre/High street)"},
            {"field": "Selling Area Percentage", "question": "What percentage of the store size is selling area?"},
            {"field": "Service Checkout Counters", "question": "How many serviced checkout counters are available?"},
            {"field": "Self Service Checkouts", "question": "Do you have self service checkouts? (Yes/No)"},
            {"field": "High Risk Assets", "question": "What high-risk or high-value assets are present?"},
            {"field": "ATM Present", "question": "Do you have an ATM? (Yes/No)"},
            {"field": "ATM Type", "question": "What type of ATM do you have? (None/Freestanding/TTW/Both)"},
            {"field": "Customer Toilet", "question": "Do you have a customer toilet? (Yes/No)"},
            {"field": "Fitting Rooms", "question": "Do you have fitting rooms? (Yes/No)"}
        ]
        self.store_data = {}
        self.current_field_idx = 0

    def get_next_question(self):
        if self.current_field_idx < len(self.store_fields):
            return self.store_fields[self.current_field_idx]
        return None

    def process_answer(self, answer):
        current_field = self.store_fields[self.current_field_idx]
        self.store_data[current_field["field"]] = answer
        self.current_field_idx += 1

    def is_complete(self):
        return self.current_field_idx >= len(self.store_fields)

class PDFReport:
    def __init__(self):
        self.pdf = FPDF(orientation='P', unit='mm', format='A4')
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=10)  # Reduced font size
        self.pdf.set_left_margin(10)
        self.pdf.set_right_margin(10)
        self.pdf.set_auto_page_break(auto=True, margin=10)  # Reduced margin
    
    def add_title(self, title):
        self.pdf.set_font('Arial', 'B', 16)  # Bold and larger title
        self.pdf.set_text_color(0, 0, 128)  # Blue color for title
        self.pdf.cell(190, 8, txt=title, ln=True, align='C')  # Reduced height
        self.pdf.ln(5)  # Reduced spacing
    
    def add_section(self, title):
        self.pdf.set_font('Arial', 'B', 12)  # Bold section title
        self.pdf.set_text_color(0, 0, 0)  # Black color for section title
        self.pdf.cell(190, 8, txt=title, ln=True)  # Reduced height
        self.pdf.ln(2)  # Minimal spacing
    
    def add_content(self, content):
        self.pdf.set_font('Arial', '', 10)  # Normal font for content
        self.pdf.set_text_color(0, 0, 0)  # Black color for content
        self.pdf.multi_cell(190, 6, txt=content)  # Reduced line height
        self.pdf.ln(2)  # Minimal spacing
    
    def add_store_info(self, store_data):
        self.add_section("Store Information")
        for field, value in store_data.items():
            self.add_content(f"{field}: {value}")
    
    def add_survey_responses(self, answers):
        self.add_section("Survey Responses")
        for question, answer in answers.items():
            self.add_content(f"Q: {question}")
            self.add_content(f"A: {answer}")
            self.pdf.ln(1)  # Minimal spacing
    
    def add_quick_report(self, report, store_data, answers):
        self.add_store_info(store_data)
        self.add_survey_responses(answers)
        
        self.add_section("Quick Analysis Summary")
        self.add_content(report['risk_summary'])
        
        self.add_section("Identified Risks")
        risks_text = ", ".join(report['identified_risks'])  # Convert list to sentence
        self.add_content(risks_text)
        
        self.add_section("Available Solutions")
        solutions_text = ", ".join(report['unique_solutions'])  # Convert list to sentence
        self.add_content(solutions_text)
    
    def add_detailed_report(self, report, store_data, answers):
        self.add_store_info(store_data)
        self.add_survey_responses(answers)
        
        for risk_data in report['identified_risks']:
            self.add_section(f"Risk: {risk_data['risk_type']}")
            
            mitigations = risk_data['mitigations']['mitigations']
            solution_details = risk_data['mitigations']['solution_details']
            
            # Add Mitigations
            self.add_section("Mitigation Steps")
            for category, items in mitigations.items():
                if items:
                    category_text = f"{category.title()}: {', '.join(items)}"  # Convert list to sentence
                    self.add_content(category_text)
            
            # Add Implementation Details
            self.add_section("Implementation Details")
            for solution_name, details in solution_details.items():
                if details:
                    self.add_content(f"**Solution: {solution_name}**")
                    if details['use_case']:
                        self.add_content(f"Use Case: {details['use_case']}")
                    if details['links']:
                        self.add_content(f"Reference Links: {details['links']}")
                    if details['partners']:
                        self.add_content(f"Partners: {details['partners']}")
                    if details['data_format']:
                        self.add_content(f"Data Format: {details['data_format']}")
                    if details['immediate_actions']:
                        self.add_content(f"Immediate Actions: {', '.join(details['immediate_actions'])}")
                    if details['data_collation']:
                        self.add_content(f"Data Collation: {', '.join(details['data_collation'])}")
                    if details['dashboard']:
                        self.add_content(f"Dashboard Features: {', '.join(details['dashboard'])}")
                    if details['wearable']:
                        self.add_content(f"Wearable Features: {', '.join(details['wearable'])}")
                    if details['mobile']:
                        self.add_content(f"Mobile Features: {', '.join(details['mobile'])}")
                    if details['soc']:
                        self.add_content(f"SOC Features: {', '.join(details['soc'])}")
                    if details['audio_visual']:
                        self.add_content(f"Audio/Visual Features: {', '.join(details['audio_visual'])}")
                    self.pdf.ln(2)  # Minimal spacing
    
    def get_download_link(self, report_type):
        try:
            temp_file = f"security_assessment_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            self.pdf.output(temp_file)
            
            with open(temp_file, "rb") as f:
                pdf_data = f.read()
            base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
            
            href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="{temp_file}">Download {report_type.title()} Report</a>'
            
            import os
            os.remove(temp_file)
            
            return href
        except Exception as e:
            st.error(f"Error generating download link: {str(e)}")
            return "Error generating PDF download link"
        
# Main Application
def main():
    st.set_page_config(page_title="Security Risk Assessment", layout="wide")
    st.title("Security Risk Survey")
    
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

    if st.session_state.chat.state == "store_info":
        field_info = st.session_state.chat.store_info.get_next_question()
        
        if field_info:
            with st.chat_message("assistant"):
                message = field_info['question']
                st.markdown(message)
                st.session_state.messages.append({"role": "assistant", "content": message})

            user_answer = st.chat_input("Your answer:")
            
            if user_answer:
                with st.chat_message("user"):
                    st.markdown(user_answer)
                    st.session_state.messages.append({"role": "user", "content": user_answer})
                
                st.session_state.chat.store_info.process_answer(user_answer)
                
                if st.session_state.chat.store_info.is_complete():
                    st.session_state.chat.state = "survey"
                st.rerun()
                
    if st.session_state.chat.state == "survey":
        question = st.session_state.chat.get_next_question()
        
        if question:
            with st.chat_message("assistant"):
                st.markdown(f"**Survey Question**: {question}")
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
        else:
            st.session_state.chat.state = "report"
            st.rerun()
            
    elif st.session_state.chat.state == "report" and not st.session_state.report_generated:
        detailed_report = st.session_state.chat.generate_detailed_report()
        quick_report = st.session_state.chat.generate_quick_report()
        
        with st.chat_message("assistant"):
            tabs = st.tabs(["Quick Analysis", "Detailed Analysis"])
            
            with tabs[0]:
                st.markdown("## Quick Security Risk Assessment Summary")
                st.markdown(quick_report['risk_summary'])
                
                st.markdown("### Identified Risks")
                for risk in quick_report['identified_risks']:
                    st.markdown(f"- {risk}")
                
                st.markdown("### Available Solutions")
                solutions_text = ", ".join(quick_report['unique_solutions'])
                st.markdown(solutions_text)
                
                st.markdown("### Download Quick Report")
                quick_pdf_link = st.session_state.chat.generate_pdf_report("quick")
                st.markdown(quick_pdf_link, unsafe_allow_html=True)
            
            with tabs[1]:
                st.markdown("## Detailed Security Risk Assessment")

                if not detailed_report['identified_risks']:
                    st.success("No security risks were identified based on your responses.")
                else:
                    for risk_data in detailed_report['identified_risks']:
                        risk_type = risk_data['risk_type']
                        full_data = risk_data['mitigations']
                        mitigations = full_data['mitigations']
                        solution_details = full_data['solution_details']

                        st.markdown(f"### Risk: {risk_type}")
                        
                        solution_tabs = st.tabs(["Solutions", "Implementation Details"])
                        
                        with solution_tabs[0]:
                            cols = st.columns(3)
                            
                            with cols[0]:
                                if mitigations['tech']:
                                    st.markdown("#### Technical Solutions")
                                    tech_text = ", ".join(mitigations['tech'])
                                    st.markdown(tech_text)
                                
                                if mitigations['human']:
                                    st.markdown("#### Human Solutions")
                                    human_text = ", ".join(mitigations['human'])
                                    st.markdown(human_text)
                            
                            with cols[1]:
                                if mitigations['tss']:
                                    st.markdown("#### TSS Solutions")
                                    tss_text = ", ".join(mitigations['tss'])
                                    st.markdown(tss_text)
                                
                                if mitigations['analytics']:
                                    st.markdown("#### Analytics Solutions")
                                    analytics_text = ", ".join(mitigations['analytics'])
                                    st.markdown(analytics_text)
                            
                            with cols[2]:
                                if mitigations['policy']:
                                    st.markdown("#### Policy Solutions")
                                    policy_text = ", ".join(mitigations['policy'])
                                    st.markdown(policy_text)
                        
                        with solution_tabs[1]:
                            for solution_name, details in solution_details.items():
                                if details:
                                    st.markdown(f"#### Solution: {solution_name}")
                                    
                                    if details['use_case']:
                                        st.markdown(f"**Use Case:** {details['use_case']}")
                                    
                                    if details['links']:
                                        st.markdown(f"**Reference Links:** {details['links']}")
                                    
                                    if details['partners']:
                                        st.markdown(f"**Partners:** {details['partners']}")
                                    
                                    cols = st.columns(2)
                                    
                                    with cols[0]:
                                        if details['immediate_actions']:
                                            st.markdown("##### Actions")
                                            actions_text = ", ".join(details['immediate_actions'])
                                            st.markdown(actions_text)
                                        
                                        if details['dashboard']:
                                            st.markdown("##### Dashboard Features")
                                            dashboard_text = ", ".join(details['dashboard'])
                                            st.markdown(dashboard_text)
                                        
                                        if details['mobile']:
                                            st.markdown("##### Mobile Features")
                                            mobile_text = ", ".join(details['mobile'])
                                            st.markdown(mobile_text)
                                    
                                    with cols[1]:
                                        if details['data_collation']:
                                            st.markdown("##### Data Collection")
                                            data_text = ", ".join(details['data_collation'])
                                            st.markdown(data_text)
                                        
                                        if details['soc']:
                                            st.markdown("##### SOC Features")
                                            soc_text = ", ".join(details['soc'])
                                            st.markdown(soc_text)
                                        
                                        if details['audio_visual']:
                                            st.markdown("##### Audio/Visual Features")
                                            av_text = ", ".join(details['audio_visual'])
                                            st.markdown(av_text)
                
                st.markdown("### Download Detailed Report")
                detailed_pdf_link = st.session_state.chat.generate_pdf_report("detailed")
                st.markdown(detailed_pdf_link, unsafe_allow_html=True)
        
        st.session_state.report_generated = True

if __name__ == "__main__":
    main()