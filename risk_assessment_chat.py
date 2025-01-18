import streamlit as st
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.agents import AgentType, initialize_agent
from typing import Dict, List, Any, Optional
from pydantic import Field

class RiskAnalyzerTool(BaseTool):
    name = "analyze_risks"
    description = "Analyzes security risks based on given answers"
    
    def __init__(self, data_processor) -> None:
        super().__init__()
        self.data_processor = data_processor
        
    def _run(self, answers: str) -> List[str]:
        return self.data_processor.analyze_risks(eval(answers))
        
    async def _arun(self, answers: str) -> List[str]:
        raise NotImplementedError("Async not implemented")

class MitigationTool(BaseTool):
    name = "suggest_mitigations"
    description = "Suggests mitigation steps for a given risk type"
    
    def __init__(self, data_processor) -> None:
        super().__init__()
        self.data_processor = data_processor
        
    def _run(self, risk_type: str) -> Dict:
        return self.data_processor.get_mitigation_steps(risk_type)
        
    async def _arun(self, risk_type: str) -> Dict:
        raise NotImplementedError("Async not implemented")

class AssuranceMetricsTool(BaseTool):
    name = "evaluate_metrics"
    description = "Evaluates assurance metrics for a given solution"
    
    def __init__(self, data_processor) -> None:
        super().__init__()
        self.data_processor = data_processor
        
    def _run(self, solution: str) -> Dict:
        return self.data_processor.get_assurance_metrics(solution)
        
    async def _arun(self, solution: str) -> Dict:
        raise NotImplementedError("Async not implemented")

class DataProcessor:
    def __init__(self):
        pass
        
    def analyze_risks(self, answers: Dict[str, str]) -> List[str]:
        risks = []
        # Add risk analysis logic here
        for question, answer in answers.items():
            if "password" in question.lower() and "yes" not in answer.lower():
                risks.append("Weak password management")
            if "encryption" in question.lower() and "no" in answer.lower():
                risks.append("Lack of encryption")
        return risks

    def get_mitigation_steps(self, risk_type: str) -> Dict[str, List[str]]:
        mitigations = {
            "Weak password management": [
                "Implement password complexity requirements",
                "Enable two-factor authentication",
                "Regular password rotation"
            ],
            "Lack of encryption": [
                "Implement end-to-end encryption",
                "Use strong encryption algorithms",
                "Regular key rotation"
            ]
        }
        return {risk_type: mitigations.get(risk_type, ["No specific mitigations found"])}

    def get_assurance_metrics(self, solution: str) -> Dict[str, Any]:
        # Dummy metrics for demonstration
        return {
            "implementation_time": "2-3 weeks",
            "cost_estimate": "medium",
            "effectiveness": 0.85
        }

class RiskAssessmentChat:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.llm = Ollama(model="llama2")
        self.memory = ConversationBufferMemory()
        self.current_question_idx = 0
        self.answers = {}
        self.setup_tools()
        self.setup_agent()

    def setup_tools(self):
        self.tools = [
            RiskAnalyzerTool(self.data_processor),
            MitigationTool(self.data_processor),
            AssuranceMetricsTool(self.data_processor)
        ]

    def setup_agent(self):
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True
        )

    def get_next_question(self) -> Optional[str]:
        questions = [
            "Do you use strong password requirements?",
            "Is sensitive data encrypted?",
            "Are there regular security audits?"
        ]
        if self.current_question_idx < len(questions):
            return questions[self.current_question_idx]
        return None

    def process_answer(self, answer: str) -> None:
        question = self.get_next_question()
        if question:
            self.answers[question] = answer
            self.current_question_idx += 1

    def generate_report(self) -> Dict[str, Any]:
        try:
            response = self.agent_executor.run(
                f"Analyze these security assessment answers and provide recommendations: {str(self.answers)}"
            )
            return {"analysis": response}
        except Exception as e:
            return {"error": str(e)}

def main():
    st.title("Security Risk Assessment Chatbot")
    st.write("Answer the questions to get a security risk assessment.")

    if "chat" not in st.session_state:
        st.session_state.chat = RiskAssessmentChat()
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.session_state.chat.get_next_question()
    if question:
        with st.chat_message("assistant"):
            st.markdown(question)

        user_answer = st.chat_input("Type your answer...")
        if user_answer:
            with st.chat_message("user"):
                st.markdown(user_answer)
            st.session_state.messages.append({"role": "user", "content": user_answer})
            st.session_state.chat.process_answer(user_answer)
            st.experimental_rerun()
    else:
        if not st.session_state.get("report_generated", False):
            with st.chat_message("assistant"):
                st.markdown("Generating security assessment report...")
                report = st.session_state.chat.generate_report()
                
                if "error" in report:
                    st.error(f"Error generating report: {report['error']}")
                else:
                    st.session_state.report_generated = True
                    
                    tabs = st.tabs(["Analysis", "Detailed Metrics"])
                    
                    with tabs[0]:
                        st.markdown("### Security Analysis")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Identified Risks")
                            risks = st.session_state.chat.data_processor.analyze_risks(
                                st.session_state.chat.answers
                            )
                            for risk in risks:
                                st.markdown(f"- {risk}")
                        
                        with col2:
                            st.markdown("#### Recommended Actions")
                            for risk in risks:
                                mitigations = st.session_state.chat.data_processor.get_mitigation_steps(risk)
                                st.markdown(f"**For {risk}:**")
                                for step in mitigations[risk]:
                                    st.markdown(f"- {step}")
                    
                    with tabs[1]:
                        st.markdown("### Implementation Metrics")
                        for risk in risks:
                            st.markdown(f"#### {risk}")
                            metrics = st.session_state.chat.data_processor.get_assurance_metrics(risk)
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Implementation Time", metrics["implementation_time"])
                                st.metric("Cost Estimate", metrics["cost_estimate"])
                            
                            with col2:
                                st.metric("Effectiveness", f"{metrics['effectiveness']*100}%")

if __name__ == "__main__":
    main()