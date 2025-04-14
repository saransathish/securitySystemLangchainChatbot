from datetime import datetime
import json
from area import DemographicAnalyzer, EnvironmentAnalyzer, CrimeAnalyzer, LocationAnalyzer
from chatbot import RiskAssessmentChat, PDFReport

class AnalysisService:
    def __init__(self, api_key):
        self.demographic_analyzer = DemographicAnalyzer(api_key)
        self.environment_analyzer = EnvironmentAnalyzer(api_key)
        self.crime_analyzer = CrimeAnalyzer()
        self.location_analyzer = LocationAnalyzer(api_key)
        self.risk_assessment = RiskAssessmentChat()

    def analyze_area(self, address, post_code):
        # Get location coordinates
        location = self.location_analyzer.get_location_coordinates(f"{address}, {post_code}")
        if not location:
            return None

        lat, lng = location

        # Perform various analyses
        demographic_data = self.demographic_analyzer.get_demographic_data(address)
        environment_data = self.environment_analyzer.analyze_environment(lat, lng)
        crimes_data = self.crime_analyzer.get_crimes_data(lat, lng)
        crime_analysis = self.crime_analyzer.analyze_crimes(crimes_data)
        nearby_places = self.location_analyzer.get_nearby_places(location, "store", 1000)
        motorway_data = self.location_analyzer.analyze_motorway_junctions(lat, lng)

        return {
            "demographic_data": demographic_data,
            "environment_data": environment_data,
            "crime_analysis": crime_analysis,
            "nearby_places": nearby_places,
            "motorway_data": motorway_data
        }

    def generate_report(self, user_data, area_analysis, survey_responses, report_type="quick"):
        pdf_report = PDFReport()
        
        # Add basic information
        pdf_report.add_title(f"Store Analysis Report - {report_type.capitalize()}")
        pdf_report.add_store_info(user_data)
        
        # Add area analysis
        pdf_report.add_section("Area Analysis")
        pdf_report.add_content(json.dumps(area_analysis, indent=2))
        
        # Add survey responses
        pdf_report.add_section("Security Survey Responses")
        pdf_report.add_survey_responses(survey_responses)
        
        if report_type == "detailed":
            # Add detailed analysis
            report_data = self.risk_assessment.generate_detailed_report()
            pdf_report.add_detailed_report(report_data, user_data, survey_responses)
        else:
            # Add quick analysis
            report_data = self.risk_assessment.generate_quick_report()
            pdf_report.add_quick_report(report_data, user_data, survey_responses)
        
        return pdf_report.get_download_link(report_type)