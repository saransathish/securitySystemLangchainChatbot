from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import os
from analysis import AnalysisService

app = Flask(__name__)
CORS(app)

# PostgreSQL configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://username:password@localhost:5432/store_analysis'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Initialize analysis service
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'your_api_key_here')
analysis_service = AnalysisService(GOOGLE_API_KEY)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    store_address = db.Column(db.String(200), nullable=False)
    post_code = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Survey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.String(500), nullable=False)
    answer = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Routes
@app.route('/api/user', methods=['POST'])
def create_user():
    try:
        data = request.json
        new_user = User(
            name=data['name'],
            store_address=data['store_address'],
            post_code=data['post_code']
        )
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': True, 'user_id': new_user.id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/survey', methods=['POST'])
def save_survey():
    try:
        data = request.json
        new_survey = Survey(
            user_id=data['user_id'],
            question=data['question'],
            answer=data['answer']
        )
        db.session.add(new_survey)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/analyze-area', methods=['POST'])
def analyze_area():
    try:
        data = request.json
        analysis_result = analysis_service.analyze_area(
            data['store_address'],
            data['post_code']
        )
        return jsonify({'success': True, 'analysis': analysis_result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/survey/question', methods=['GET'])
def get_next_question():
    try:
        chat = analysis_service.risk_assessment
        question = chat.get_next_question()
        return jsonify({'success': True, 'question': question})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/report/<report_type>', methods=['GET'])
def generate_report(report_type):
    try:
        user_id = request.args.get('user_id')
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        # Get area analysis
        area_analysis = analysis_service.analyze_area(user.store_address, user.post_code)
        
        # Get survey responses
        survey_responses = Survey.query.filter_by(user_id=user_id).all()
        responses_dict = {r.question: r.answer for r in survey_responses}
        
        # Generate report
        report_url = analysis_service.generate_report(
            {
                'name': user.name,
                'store_address': user.store_address,
                'post_code': user.post_code
            },
            area_analysis,
            responses_dict,
            report_type
        )
        
        return jsonify({'success': True, 'report_url': report_url})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)