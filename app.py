import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import FileParser, ResumeValidator, DatabaseManager
from inference import RobustATSInference
from job_recommender import EnhancedJobRecommender
from gemini_integration import get_gemini_client
import os
import re
import time
import uuid

# Page configuration
st.set_page_config(
    page_title="Advanced ATS System",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.portal-header {
    background: linear-gradient(45deg, #1f77b4, #ff7f0e);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
}
.score-card {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    text-align: center;
}
.poor-fit { background-color: #f8d7da; color: #721c24; }
.weak-match { background-color: #fff3cd; color: #856404; }
.good-match { background-color: #d4edda; color: #155724; }
.excellent-match { background-color: #d1ecf1; color: #0c5460; }
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.user-message { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
.assistant-message { background-color: #f3e5f5; border-left: 4px solid #9c27b0; }
.chat-container {
    max-height: 400px;
    overflow-y: auto;
    margin-bottom: 1rem;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
}
.job-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.real-job { border-left: 4px solid #28a745; }
.generated-job { border-left: 4px solid #ffc107; }
.candidate-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}
.candidate-card:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
.shortlisted {
    border-left: 4px solid #28a745;
    background-color: #f8fff9;
}
.rejected {
    border-left: 4px solid #dc3545;
    background-color: #fff8f8;
}
.pending {
    border-left: 4px solid #ffc107;
}
</style>
""", unsafe_allow_html=True)

class AdvancedATSApp:
    def __init__(self):
        self.file_parser = FileParser()
        self.resume_validator = ResumeValidator()
        self.db_manager = DatabaseManager()
        self.ats_inference = RobustATSInference()
        self.job_recommender = EnhancedJobRecommender()
        self.gemini = get_gemini_client()
        
        # Session state
        if 'current_resume' not in st.session_state:
            st.session_state.current_resume = None
        if 'current_jd' not in st.session_state:
            st.session_state.current_jd = None
        if 'job_seeker_chat_history' not in st.session_state:
            st.session_state.job_seeker_chat_history = []
        if 'recruiter_resumes' not in st.session_state:
            st.session_state.recruiter_resumes = []
        if 'shortlisted_candidates' not in st.session_state:
            st.session_state.shortlisted_candidates = []
        if 'selected_jd' not in st.session_state:
            st.session_state.selected_jd = None
    
    def setup_sidebar(self):
        """Setup sidebar configuration"""
        st.sidebar.title("🔧 Configuration")
        
        # Model status
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 Model Status")
        model_status = self.ats_inference.get_model_status()
        
        if model_status['transformer'] == 'Base':
            st.sidebar.success("✅ Transformer: Loaded")
        else:
            st.sidebar.error("❌ Transformer: Error")
        
        if model_status['ml_model'] == 'Trained':
            st.sidebar.success("✅ ATS Model: Trained")
        else:
            st.sidebar.info("ℹ️ ATS Model: Enhanced Scoring")
            
        # API status
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔑 API Status")
        if os.getenv('GEMINI_API_KEY'):
            st.sidebar.success("✅ Gemini: Configured")
        else:
            st.sidebar.error("❌ Gemini: Not Configured")
            
        if os.getenv('SERPAPI_KEY'):
            st.sidebar.success("✅ SerpAPI: Configured")
        else:
            st.sidebar.warning("⚠️ SerpAPI: Not Configured")

        # Quick actions
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Quick Actions")
        
        if st.sidebar.button("🔄 Clear Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        if st.sidebar.button("📊 View Stats", use_container_width=True):
            self.show_database_stats()

    def show_database_stats(self):
        """Show database statistics"""
        resumes = self.db_manager.get_all_resumes()
        jds = self.db_manager.get_all_jds()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 Database Stats")
        st.sidebar.write(f"📄 Resumes: {len(resumes)}")
        st.sidebar.write(f"📋 Job Descriptions: {len(jds)}")
        st.sidebar.write(f"👥 Recruiter Uploads: {len(st.session_state.recruiter_resumes)}")
        st.sidebar.write(f"⭐ Shortlisted: {len(st.session_state.shortlisted_candidates)}")

    def job_seeker_portal(self):
        """Job Seeker Portal"""
        st.markdown('<div class="portal-header"><h1>🎯 Job Seeker Portal</h1><p>Upload your resume and compare with job descriptions</p></div>', 
                   unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload Resume", "🔍 Compare with JD", "💼 Job Recommendations", "🤖 CareerGPT Assistant"])
        
        with tab1:
            self.upload_resume_section()
        
        with tab2:
            self.compare_with_jd_section()
        
        with tab3:
            self.job_recommendations_section()
        
        with tab4:
            self.virtual_assistant_section()
    
    def upload_resume_section(self):
        """Section for uploading resume"""
        st.subheader("📄 Upload Your Resume")
        
        resume_file = st.file_uploader(
            "Choose your resume file (PDF, DOCX, PNG, JPG, JPEG)",
            type=['pdf', 'docx', 'png', 'jpg', 'jpeg'],
            key="resume_upload"
        )
        
        if resume_file:
            # Parse resume
            with st.spinner("🔄 Parsing your resume..."):
                file_bytes = resume_file.getvalue()
                resume_text = self.file_parser.parse_file(file_bytes, resume_file.name)
            
            if resume_text:
                st.session_state.current_resume = {
                    'text': resume_text,
                    'bytes': file_bytes,
                    'filename': resume_file.name
                }
                
                # Validate resume and provide ATS feedback
                is_valid, missing_sections = self.resume_validator.validate_resume(resume_text)
                ats_feedback = self.analyze_ats_friendliness(resume_text)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if is_valid:
                        st.success("✅ Resume is valid!")
                    else:
                        st.error(f"❌ Resume missing sections: {', '.join(missing_sections)}")
                
                with col2:
                    st.info(f"📊 ATS Friendliness: {ats_feedback['score']}/100")
                
                # Show detailed ATS feedback
                with st.expander("🔍 Detailed ATS Analysis"):
                    for category, feedback in ats_feedback['details'].items():
                        st.write(f"**{category}:** {feedback}")
                
                # Personal info form
                with st.form("resume_info"):
                    st.subheader("👤 Personal Information")
                    name = st.text_input("Full Name*")
                    email = st.text_input("Email*")
                    
                    if st.form_submit_button("💾 Save Resume Profile"):
                        if name and email:
                            success = self.db_manager.store_resume(
                                name, email, resume_text, file_bytes, resume_file.name
                            )
                            if success:
                                st.success("✅ Resume saved successfully!")
                        else:
                            st.error("Please fill in all required fields")
                
                # Show parsed text
                with st.expander("👀 View Parsed Resume Text"):
                    st.text_area("Resume Content", resume_text, height=200, key="resume_content_display")
            else:
                st.error("❌ Could not extract text from the file. Please try another file.")
    
    def analyze_ats_friendliness(self, resume_text):
        """Analyze resume for ATS friendliness"""
        # Calculate basic metrics
        word_count = len(resume_text.split())
        sections_count = self._count_sections(resume_text)
        
        # Score calculation
        length_score = min(word_count / 8, 100)  # Optimal ~800 words = 100
        section_score = min(sections_count * 20, 100)  # 5 sections = 100
        
        # Check for ATS-friendly elements
        has_quantifiable_achievements = bool(re.search(r'\d+%|\$\d+|\d+\+', resume_text))
        has_action_verbs = bool(re.search(r'managed|developed|created|implemented|led', resume_text, re.I))
        has_contact_info = bool(re.search(r'@|\d{10}', resume_text))
        
        feature_score = 0
        if has_quantifiable_achievements:
            feature_score += 25
        if has_action_verbs:
            feature_score += 25
        if has_contact_info:
            feature_score += 25
        
        total_score = (length_score * 0.3 + section_score * 0.4 + feature_score * 0.3)
        
        feedback_details = {
            "Length": f"{word_count} words - {'Good' if 400 <= word_count <= 1200 else 'Needs adjustment'}",
            "Sections": f"{sections_count} detected - {'Complete' if sections_count >= 4 else 'Incomplete'}",
            "Quantifiable Achievements": "✅ Present" if has_quantifiable_achievements else "❌ Missing",
            "Action Verbs": "✅ Present" if has_action_verbs else "❌ Missing",
            "Contact Info": "✅ Present" if has_contact_info else "❌ Missing"
        }
        
        return {'score': round(total_score), 'details': feedback_details}
    
    def _count_sections(self, text):
        """Count number of important sections in resume"""
        sections = ['education', 'experience', 'skills', 'work', 'projects', 'certifications', 'summary', 'objective']
        text_lower = text.lower()
        return sum(1 for section in sections if section in text_lower)
    
    def compare_with_jd_section(self):
        """Enhanced section for comparing resume with job description"""
        st.subheader("🔍 Compare Resume with Job Description")
        
        if not st.session_state.current_resume:
            st.warning("⚠️ Please upload your resume first in the 'Upload Resume' tab")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("📋 Your Resume is Ready")
            resume_preview = st.session_state.current_resume['text'][:500] + "..." if len(st.session_state.current_resume['text']) > 500 else st.session_state.current_resume['text']
            st.text_area("Current Resume Preview", resume_preview, height=150, key="resume_display", disabled=True)
        
        with col2:
            st.subheader("Upload Job Description")
            jd_file = st.file_uploader(
                "Upload Job Description",
                type=['pdf', 'docx', 'png', 'jpg', 'jpeg', 'txt'],
                key="jd_upload"
            )
            
            jd_text = st.text_area("Or paste job description", height=150,
                                 placeholder="Paste the job description here...",
                                 key="jd_text")
            
            if jd_file:
                with st.spinner("🔄 Parsing job description..."):
                    jd_bytes = jd_file.getvalue()
                    jd_text = self.file_parser.parse_file(jd_bytes, jd_file.name)
                    st.session_state.current_jd = {
                        'text': jd_text,
                        'bytes': jd_bytes,
                        'filename': jd_file.name
                    }
                    st.success("✅ Job description parsed!")
            
            if jd_text and not jd_file:
                st.session_state.current_jd = {'text': jd_text}
            
            if st.session_state.current_jd and st.button("🚀 Analyze Compatibility", use_container_width=True):
                self.perform_ats_analysis()
    
    def perform_ats_analysis(self):
        """Perform ATS analysis between resume and JD"""
        resume_text = st.session_state.current_resume['text']
        jd_text = st.session_state.current_jd['text']
        
        with st.spinner("🔄 Analyzing compatibility..."):
            # Extract features with error handling
            try:
                features, resume_emb, jd_emb = self.ats_inference.extract_features(resume_text, jd_text)
                
                if features is None:
                    st.error("❌ Error extracting features. Please check your inputs.")
                    return
                
                # Calculate scores
                ats_score = self.ats_inference.predict_ats_score(features)
                ats_category = self.ats_inference.get_ats_category(ats_score)
                similarity = self.ats_inference.calculate_similarity_percentage(resume_emb, jd_emb)
                
                # Display results
                self.display_analysis_results(ats_score, ats_category, similarity, features)
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")

    def display_analysis_results(self, ats_score, ats_category, similarity, features):
        """Display ATS analysis results"""
        st.subheader("📊 Analysis Results")
        
        # Score cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ATS Score", f"{ats_score:.1f}/100")
        
        with col2:
            category_class = ats_category.lower().replace(" ", "-")
            st.markdown(f'<div class="score-card {category_class}"><h3>{ats_category}</h3></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            st.metric("Semantic Similarity", f"{similarity:.1f}%")
        
        # Fit assessment
        fit_assessment = self.get_fit_assessment(ats_score)
        st.info(f"🎯 **Fit Assessment:** {fit_assessment}")
        
        # Progress bar for score visualization
        st.subheader("Score Breakdown")
        st.progress(ats_score/100)
        
        if ats_score < 50:
            st.warning("⚠️ Your resume needs significant improvements to pass ATS screening")
        elif ats_score < 70:
            st.info("ℹ️ Good foundation, but optimization can improve your chances")
        else:
            st.success("🎉 Excellent! Your resume is well-optimized for ATS")

        # Features breakdown
        st.subheader("🔍 Feature Breakdown")
        
        if features:
            features_data = {
                'Feature': ['Keyword Match', 'Skill Overlap', 'Resume Quality', 'Sections', 'Semantic Match'],
                'Score': [
                    features[1],  # keyword_match
                    features[2],  # skill_overlap
                    features[3],  # resume_quality
                    features[4],  # sections_count
                    features[0]   # semantic_similarity
                ]
            }
            
            fig = px.bar(features_data, x='Feature', y='Score', title="Feature Scores",
                        range_y=[0, 100], color='Score',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        st.subheader("📈 Feature Importance")
        importance = self.ats_inference.get_feature_importance()
        
        fig = go.Figure(data=[
            go.Bar(x=list(importance.keys()), y=list(importance.values()))
        ])
        fig.update_layout(title="How Features Affect Your ATS Score")
        st.plotly_chart(fig, use_container_width=True)

        # AI Feedback - with fallback
        st.subheader("🤖 AI Feedback & Suggestions")
        try:
            feedback = self.gemini.generate_resume_feedback(
                st.session_state.current_resume['text'],
                st.session_state.current_jd['text'],
                ats_score,
                features
            )
            st.markdown(feedback)
        except Exception as e:
            st.warning("⚠️ Using enhanced fallback feedback (Gemini API not available)")
            fallback_feedback = self.generate_fallback_feedback(ats_score, features)
            st.markdown(fallback_feedback)
    
    def get_fit_assessment(self, ats_score):
        """Get fit assessment based on dataset patterns"""
        if ats_score >= 75:
            return "🎉 Excellent Fit - High probability of passing ATS"
        elif ats_score >= 60:
            return "👍 Good Fit - Strong match with job requirements"
        elif ats_score >= 40:
            return "🤔 Potential Fit - Some alignment, needs optimization"
        else:
            return "❌ Poor Fit - Significant improvements needed"
    
    def generate_fallback_feedback(self, ats_score, features):
        """Generate meaningful fallback feedback"""
        feedback = []
        
        if ats_score >= 75:
            feedback.append("## 🎉 Excellent Match!")
            feedback.append("Your resume is well-optimized for this position.")
            feedback.append("**Maintain:** Continue using quantifiable achievements and relevant keywords")
            
        elif ats_score >= 60:
            feedback.append("## 👍 Good Match")
            feedback.append("Your resume has good alignment but can be improved.")
            feedback.append("**Improve:**")
            feedback.append("- Increase keyword density by 15-20%")
            feedback.append("- Add more specific technical skills from the job description")
            feedback.append("- Include more metrics and numbers in achievements")
            
        elif ats_score >= 40:
            feedback.append("## 🤔 Potential Fit")
            feedback.append("Some alignment found, but significant improvements needed.")
            feedback.append("**Critical Actions:**")
            feedback.append("- Match 50% more keywords from the job description")
            feedback.append("- Restructure resume sections for better ATS parsing")
            feedback.append("- Add missing technical skills and certifications")
            
        else:
            feedback.append("## ❌ Poor Fit")
            feedback.append("Major improvements needed to pass ATS screening.")
            feedback.append("**Immediate Actions:**")
            feedback.append("- Completely restructure resume to match job requirements")
            feedback.append("- Add all missing key skills and technologies")
            feedback.append("- Increase resume length to 400-800 words")
            feedback.append("- Ensure all major sections are present")
        
        # Add feature-specific feedback
        feedback.append("\n## 🔍 Specific Recommendations:")
        
        if features[1] < 60:  # keyword_match
            feedback.append("- **Keyword Match Low:** Add more job-specific keywords in first 1/3 of resume")
        
        if features[2] < 50:  # skill_overlap
            feedback.append("- **Skill Gap:** Identify and add missing technical skills from job description")
        
        if features[4] < 4:  # sections_count
            feedback.append("- **Sections Missing:** Ensure Experience, Education, Skills, and Summary sections are present")
        
        return "\n\n".join(feedback)
    
    def job_recommendations_section(self):
        """Enhanced job recommendations based on resume"""
        st.subheader("💼 Job Recommendations")
        
        if not st.session_state.current_resume:
            st.warning("⚠️ Please upload your resume first")
            return
        
        # Extract skills and experience from user's actual resume
        resume_text = st.session_state.current_resume['text']
        skills = self.extract_skills_from_resume(resume_text)
        
        if not skills:
            st.warning("⚠️ No skills detected in your resume. Please ensure your resume includes technical skills.")
            return
        
        st.write(f"**Detected Skills:** {', '.join(skills[:8])}")
        
        # User inputs for job search - NO HARDCODING
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location = st.text_input("📍 Preferred Location", "Remote")
        
        with col2:
            job_title = st.text_input("💼 Job Title Preference", "")
        
        with col3:
            # User inputs experience - NO HARDCODING
            experience = st.number_input("🎯 Years of Experience", min_value=0, max_value=30, value=self.extract_experience_from_resume(resume_text))
        
        limit = st.slider("Number of Recommendations", 3, 10, 5)
        
        if st.button("🔍 Find Job Recommendations"):
            with st.spinner("🔍 Searching for matching jobs..."):
                recommendations = self.job_recommender.get_job_recommendations(
                    skills, experience, location, limit
                )
                
                if recommendations:
                    st.subheader(f"🎉 Found {len(recommendations)} Jobs")
                    
                    for i, job in enumerate(recommendations, 1):
                        job_class = "real-job" if not job.get('is_generated', True) else "generated-job"
                        with st.expander(f"{i}. {job['title']} at {job['company']} - 💰 {job['salary']}", expanded=i==1):
                            st.markdown(f'<div class="job-card {job_class}">', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**🏢 Company:** {job['company']}")
                                st.write(f"**📍 Location:** {job['location']}")
                                st.write(f"**💰 Salary:** {job['salary']}")
                                st.write(f"**📅 Posted:** {job.get('posted_date', 'Recently')}")
                                st.write(f"**🔍 Source:** {'Real' if not job.get('is_generated', True) else 'Generated'}")
                                
                                st.write("**📋 Description:**")
                                st.write(job['description'])
                            
                            with col2:
                                if st.button(f"📨 Apply", key=f"apply_{i}", use_container_width=True):
                                    if job.get('source_url'):
                                        st.success(f"🎯 Redirecting to {job['company']} application")
                                        # Show actual application link
                                        st.markdown(f"[Click here to apply]({job['source_url']})", unsafe_allow_html=True)
                                    else:
                                        st.info("💡 Application portal would open here in a real implementation")
                                        st.info("For real jobs, configure SerpAPI key in your environment variables")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("❌ No job recommendations found. Try adjusting your search criteria.")
    
    def virtual_assistant_section(self):
        """Enhanced Virtual Assistant section with FIXED chat interface"""
        st.subheader("🤖 CareerGPT - Your AI Career Assistant")
        
        st.info("💬 Ask me anything about resumes, job search, interviews, or career advice!")
        
        # Display chat history in a scrollable container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in st.session_state.job_seeker_chat_history:
            if chat["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {chat["message"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>CareerGPT:</strong> {chat["message"]}</div>', 
                          unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick action buttons
        st.subheader("🚀 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📝 Resume Tips", use_container_width=True):
                st.session_state.job_seeker_chat_history.append({
                    "role": "user", 
                    "message": "Give me top 5 resume tips for ATS optimization in 2024"
                })
                st.rerun()
        
        with col2:
            if st.button("💼 Interview Prep", use_container_width=True):
                st.session_state.job_seeker_chat_history.append({
                    "role": "user", 
                    "message": "How should I prepare for a technical interview for a software developer role?"
                })
                st.rerun()
        
        with col3:
            if st.button("🎯 Career Advice", use_container_width=True):
                st.session_state.job_seeker_chat_history.append({
                    "role": "user", 
                    "message": "What career growth opportunities should I explore based on current market trends?"
                })
                st.rerun()
        
        with col4:
            if st.button("🔍 Job Search", use_container_width=True):
                st.session_state.job_seeker_chat_history.append({
                    "role": "user", 
                    "message": "What are the best strategies for job searching in the current market?"
                })
                st.rerun()
        
        # FIXED: Chat input moved to the VERY BOTTOM, outside all containers
        # This ensures it's at the main level of the app
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to chat
            st.session_state.job_seeker_chat_history.append({"role": "user", "message": user_input})
            
            # Generate AI response
            with st.spinner("CareerGPT is thinking..."):
                try:
                    response = self.gemini.chat_assistant(user_input, st.session_state.job_seeker_chat_history)
                    st.session_state.job_seeker_chat_history.append({"role": "assistant", "message": response})
                except Exception as e:
                    fallback_response = "I'm currently experiencing technical difficulties. Please try again later or check the Gemini API configuration."
                    st.session_state.job_seeker_chat_history.append({"role": "assistant", "message": fallback_response})
            
            st.rerun()
    
    def extract_skills_from_resume(self, resume_text):
        """Extract skills from resume text"""
        if not resume_text:
            return []
            
        skills_keywords = [
            'Python', 'Java', 'SQL', 'JavaScript', 'Machine Learning', 'Data Analysis',
            'AWS', 'Docker', 'Communication', 'Teamwork', 'Problem Solving', 'Leadership',
            'Project Management', 'Agile', 'Scrum', 'Excel', 'PowerPoint', 'Word',
            'HTML', 'CSS', 'React', 'Angular', 'Vue', 'Node.js', 'Express', 'Django',
            'Flask', 'FastAPI', 'MongoDB', 'MySQL', 'PostgreSQL', 'Oracle', 'Git',
            'GitHub', 'Jenkins', 'Kubernetes', 'Linux', 'Windows', 'macOS'
        ]
        found_skills = [skill for skill in skills_keywords if skill.lower() in resume_text.lower()]
        return found_skills
    
    def extract_experience_from_resume(self, resume_text):
        """Extract experience from resume text - NO HARDCODING"""
        if not resume_text:
            return 1
            
        matches = re.findall(r'(\d+)\s*(?:years?|yrs?)', resume_text.lower())
        if matches:
            return max([int(m) for m in matches])
        else:
            # Try to infer from context without hardcoding
            text_lower = resume_text.lower()
            if any(word in text_lower for word in ['senior', 'lead', 'manager', 'director']):
                return 5
            elif any(word in text_lower for word in ['mid-level', 'intermediate', 'experienced']):
                return 3
            else:
                return 1

    def recruiter_portal(self):
        """Enhanced Recruiter Portal with Resume Upload and Shortlisting"""
        st.markdown('<div class="portal-header"><h1>🏢 Recruiter Portal</h1><p>Upload resumes, create job descriptions, and screen candidates</p></div>', 
                   unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📄 Upload Resumes", "📋 Create Job Description", "👥 Screen & Shortlist"])
        
        with tab1:
            self.upload_resumes_section()
        
        with tab2:
            self.upload_jd_section()
        
        with tab3:
            self.screen_candidates_section()
    
    def upload_resumes_section(self):
        """Section for recruiters to upload multiple resumes"""
        st.subheader("📄 Upload Candidate Resumes")
        
        st.info("💡 Upload multiple resumes to build your candidate pool. These will appear in the screening section.")
        
        # Multiple file upload
        uploaded_files = st.file_uploader(
            "Choose candidate resume files (PDF, DOCX, PNG, JPG, JPEG)",
            type=['pdf', 'docx', 'png', 'jpg', 'jpeg'],
            key="recruiter_resume_upload",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for resume_file in uploaded_files:
                with st.spinner(f"🔄 Processing {resume_file.name}..."):
                    file_bytes = resume_file.getvalue()
                    resume_text = self.file_parser.parse_file(file_bytes, resume_file.name)
                    
                    if resume_text:
                        # Create candidate object
                        candidate_id = str(uuid.uuid4())[:8]
                        candidate = {
                            'id': candidate_id,
                            'name': f"Candidate_{candidate_id}",
                            'email': f"candidate_{candidate_id}@company.com",
                            'resume_text': resume_text,
                            'filename': resume_file.name,
                            'uploaded_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'status': 'pending',
                            'ats_score': 0,
                            'skills': self.extract_skills_from_resume(resume_text),
                            'experience': self.extract_experience_from_resume(resume_text)
                        }
                        
                        # Check if already exists
                        existing_ids = [c['id'] for c in st.session_state.recruiter_resumes]
                        if candidate_id not in existing_ids:
                            st.session_state.recruiter_resumes.append(candidate)
                            st.success(f"✅ {resume_file.name} uploaded successfully!")
                        else:
                            st.info(f"ℹ️ {resume_file.name} already uploaded")
                    else:
                        st.error(f"❌ Could not extract text from {resume_file.name}")
        
        # Show uploaded resumes
        if st.session_state.recruiter_resumes:
            st.subheader(f"📂 Uploaded Resumes ({len(st.session_state.recruiter_resumes)})")
            
            for candidate in st.session_state.recruiter_resumes:
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"**{candidate['name']}**")
                    st.write(f"Skills: {', '.join(candidate['skills'][:5]) if candidate['skills'] else 'Not detected'}")
                    st.write(f"Experience: {candidate['experience']} years")
                
                with col2:
                    status_color = {
                        'pending': 'orange',
                        'shortlisted': 'green', 
                        'rejected': 'red'
                    }.get(candidate['status'], 'gray')
                    st.markdown(f"Status: <span style='color: {status_color}; font-weight: bold;'>{candidate['status'].title()}</span>", 
                              unsafe_allow_html=True)
                    st.write(f"Uploaded: {candidate['uploaded_at']}")
                
                with col3:
                    if st.button("👀 View", key=f"view_{candidate['id']}"):
                        with st.expander(f"Resume Content - {candidate['name']}"):
                            st.text_area("Resume Text", candidate['resume_text'], height=200, key=f"resume_{candidate['id']}")
                
                st.markdown("---")
        
        # Clear all button
        if st.session_state.recruiter_resumes and st.button("🗑️ Clear All Resumes", type="secondary"):
            st.session_state.recruiter_resumes = []
            st.rerun()
    
    def upload_jd_section(self):
        """Section for uploading job description"""
        st.subheader("📋 Create Job Description")
        
        jd_file = st.file_uploader(
            "Upload Job Description",
            type=['pdf', 'docx', 'png', 'jpg', 'jpeg', 'txt'],
            key="recruiter_jd"
        )
        
        jd_text = st.text_area("Or paste job description", height=200,
                             placeholder="Paste the complete job description here...",
                             key="recruiter_jd_text")
        
        if jd_file:
            with st.spinner("🔄 Parsing job description..."):
                jd_bytes = jd_file.getvalue()
                jd_text = self.file_parser.parse_file(jd_bytes, jd_file.name)
        
        if jd_text:
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Job Title*", placeholder="e.g., Senior Python Developer")
            with col2:
                company = st.text_input("Company Name*", placeholder="Your Company")
            
            # Additional JD details
            col3, col4 = st.columns(2)
            with col3:
                location = st.text_input("Location", "Remote")
            with col4:
                experience_required = st.number_input("Years Experience Required", min_value=0, max_value=30, value=3)
            
            if st.button("💾 Save Job Description") and title and company:
                jd_id = str(uuid.uuid4())[:8]
                jd_data = {
                    'id': jd_id,
                    'title': title,
                    'company': company,
                    'location': location,
                    'experience_required': experience_required,
                    'description_text': jd_text,
                    'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Store in session state
                if 'recruiter_jds' not in st.session_state:
                    st.session_state.recruiter_jds = []
                
                st.session_state.recruiter_jds.append(jd_data)
                st.session_state.selected_jd = jd_data
                
                st.success("✅ Job description saved successfully!")
                
                # Extract and show insights from the JD
                skills = self.extract_skills_from_text(jd_text)
                if skills:
                    st.write(f"**Key Skills Required:** {', '.join(skills[:10])}")
            elif st.button("💾 Save Job Description"):
                st.error("❌ Please fill in all required fields (Title and Company)")
    
    def screen_candidates_section(self):
        """Screen and shortlist candidates against job description"""
        st.subheader("👥 Screen & Shortlist Candidates")
        
        if not st.session_state.recruiter_resumes:
            st.warning("⚠️ Please upload some resumes first in the 'Upload Resumes' tab")
            return
        
        # Job description selection
        if 'recruiter_jds' not in st.session_state or not st.session_state.recruiter_jds:
            st.warning("⚠️ Please create a job description first in the 'Create Job Description' tab")
            return
        
        jd_options = {f"{jd['title']} at {jd['company']}": jd for jd in st.session_state.recruiter_jds}
        selected_jd_label = st.selectbox("Select Job Description", list(jd_options.keys()))
        selected_jd = jd_options[selected_jd_label]
        
        st.session_state.selected_jd = selected_jd
        
        st.write(f"**Selected JD:** {selected_jd['title']} at {selected_jd['company']}")
        st.write(f"**Location:** {selected_jd['location']} | **Experience Required:** {selected_jd['experience_required']} years")
        st.write(f"**Description Preview:** {selected_jd['description_text'][:200]}...")
        
        if st.button("🚀 Screen All Candidates", type="primary", use_container_width=True):
            with st.spinner("🔄 Screening candidates against job description..."):
                jd_text = selected_jd['description_text']
                screened_candidates = []
                
                progress_bar = st.progress(0)
                total_candidates = len(st.session_state.recruiter_resumes)
                
                for idx, candidate in enumerate(st.session_state.recruiter_resumes):
                    resume_text = candidate['resume_text']
                    
                    # Extract features and calculate score
                    features, _, _ = self.ats_inference.extract_features(resume_text, jd_text)
                    
                    if features:
                        ats_score = self.ats_inference.predict_ats_score(features)
                        candidate['ats_score'] = ats_score
                        candidate['fit_category'] = self.ats_inference.get_ats_category(ats_score)
                        candidate['screened_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    screened_candidates.append(candidate)
                    progress_bar.progress((idx + 1) / total_candidates)
                
                # Sort by score
                screened_candidates.sort(key=lambda x: x.get('ats_score', 0), reverse=True)
                st.session_state.recruiter_resumes = screened_candidates
                
                st.success(f"✅ Screened {len(screened_candidates)} candidates!")
        
        # Display screening results
        if any(candidate.get('ats_score') for candidate in st.session_state.recruiter_resumes):
            st.subheader(f"📊 Screening Results for {selected_jd['title']}")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                min_score = st.slider("Minimum ATS Score", 0, 100, 50)
            with col2:
                status_filter = st.selectbox("Status Filter", ["All", "Pending", "Shortlisted", "Rejected"])
            with col3:
                sort_by = st.selectbox("Sort By", ["ATS Score", "Experience", "Name"])
            
            # Filter and sort candidates
            filtered_candidates = [c for c in st.session_state.recruiter_resumes 
                                 if c.get('ats_score', 0) >= min_score]
            
            if status_filter != "All":
                filtered_candidates = [c for c in filtered_candidates if c['status'] == status_filter.lower()]
            
            if sort_by == "ATS Score":
                filtered_candidates.sort(key=lambda x: x.get('ats_score', 0), reverse=True)
            elif sort_by == "Experience":
                filtered_candidates.sort(key=lambda x: x.get('experience', 0), reverse=True)
            else:
                filtered_candidates.sort(key=lambda x: x['name'])
            
            st.write(f"**Showing {len(filtered_candidates)} candidates**")
            
            # Display candidates
            for candidate in filtered_candidates:
                candidate_class = candidate['status']
                ats_score = candidate.get('ats_score', 0)
                fit_category = candidate.get('fit_category', 'Not Screened')
                
                st.markdown(f'<div class="candidate-card {candidate_class}">', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                
                with col1:
                    st.write(f"### {candidate['name']}")
                    st.write(f"**Skills:** {', '.join(candidate['skills'][:5]) if candidate['skills'] else 'Not detected'}")
                    st.write(f"**Experience:** {candidate['experience']} years")
                    st.write(f"**File:** {candidate['filename']}")
                
                with col2:
                    st.metric("ATS Score", f"{ats_score:.1f}")
                    st.write(f"**Fit:** {fit_category}")
                
                with col3:
                    # Status actions
                    current_status = candidate['status']
                    if current_status == 'pending':
                        if st.button("⭐ Shortlist", key=f"shortlist_{candidate['id']}", use_container_width=True):
                            candidate['status'] = 'shortlisted'
                            if candidate['id'] not in [c['id'] for c in st.session_state.shortlisted_candidates]:
                                st.session_state.shortlisted_candidates.append(candidate)
                            st.rerun()
                        if st.button("❌ Reject", key=f"reject_{candidate['id']}", use_container_width=True):
                            candidate['status'] = 'rejected'
                            st.rerun()
                    elif current_status == 'shortlisted':
                        st.success("✅ Shortlisted")
                        if st.button("↩️ Undo", key=f"undo_short_{candidate['id']}", use_container_width=True):
                            candidate['status'] = 'pending'
                            st.session_state.shortlisted_candidates = [c for c in st.session_state.shortlisted_candidates if c['id'] != candidate['id']]
                            st.rerun()
                    else:  # rejected
                        st.error("❌ Rejected")
                        if st.button("↩️ Undo", key=f"undo_reject_{candidate['id']}", use_container_width=True):
                            candidate['status'] = 'pending'
                            st.rerun()
                
                with col4:
                    if st.button("👀 View Resume", key=f"view_res_{candidate['id']}", use_container_width=True):
                        with st.expander(f"Resume - {candidate['name']}"):
                            st.text_area("Resume Content", candidate['resume_text'], height=200, 
                                       key=f"resume_view_{candidate['id']}")
                    
                    if st.button("📋 Contact", key=f"contact_{candidate['id']}", use_container_width=True):
                        st.info(f"Contact: {candidate['email']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Shortlisted candidates summary
            if st.session_state.shortlisted_candidates:
                st.subheader("⭐ Shortlisted Candidates")
                shortlisted_count = len(st.session_state.shortlisted_candidates)
                avg_score = sum(c.get('ats_score', 0) for c in st.session_state.shortlisted_candidates) / shortlisted_count
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Shortlisted", shortlisted_count)
                with col2:
                    st.metric("Average Score", f"{avg_score:.1f}")
                
                # Export shortlisted candidates
                if st.button("📤 Export Shortlisted Candidates"):
                    shortlisted_data = []
                    for candidate in st.session_state.shortlisted_candidates:
                        shortlisted_data.append({
                            'Name': candidate['name'],
                            'Email': candidate['email'],
                            'ATS Score': candidate.get('ats_score', 0),
                            'Fit Category': candidate.get('fit_category', 'Not Screened'),
                            'Experience': candidate['experience'],
                            'Skills': ', '.join(candidate['skills']) if candidate['skills'] else '',
                            'Screened At': candidate.get('screened_at', 'Not Screened')
                        })
                    
                    df = pd.DataFrame(shortlisted_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"shortlisted_candidates_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

    def extract_skills_from_text(self, text):
        """Extract skills from any text"""
        if not text:
            return []
            
        skills_keywords = [
            'Python', 'Java', 'SQL', 'JavaScript', 'Machine Learning', 
            'Data Analysis', 'AWS', 'Docker', 'Communication', 'Teamwork',
            'React', 'Angular', 'Vue', 'Node.js', 'Express', 'Django',
            'Flask', 'MongoDB', 'MySQL', 'PostgreSQL', 'Git', 'GitHub'
        ]
        found_skills = [skill for skill in skills_keywords if skill.lower() in text.lower()]
        return found_skills

    def run(self):
        """Main application runner"""
        # Setup sidebar and get navigation
        self.setup_sidebar()
        
        st.sidebar.title("🚀 Navigation")
        app_mode = st.sidebar.radio("Select Portal", ["Job Seeker Portal", "Recruiter Portal"])
        
        # Run selected portal
        if app_mode == "Job Seeker Portal":
            self.job_seeker_portal()
        else:
            self.recruiter_portal()

# Run the app
if __name__ == "__main__":
    # Initialize database
    try:
        from setup_database import setup_database
        setup_database()
        st.success("✅ Database initialized successfully!")
    except Exception as e:
        st.error(f"❌ Database error: {e}")
    
    # Run app
    app = AdvancedATSApp()
    app.run()