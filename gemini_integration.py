import google.generativeai as genai
import os
from dotenv import load_dotenv

class FixedGeminiIntegration:
    def __init__(self, api_key=None):
        load_dotenv()
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = None
        self.is_working = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                # Test the connection
                test_response = self.model.generate_content("Hello")
                if test_response:
                    self.is_working = True
                    print("✅ Gemini API working")
            except Exception as e:
                print(f"⚠️ Gemini API failed: {e}")
                self.is_working = False
        else:
            print("⚠️ No Gemini API key found")

    def generate_resume_feedback(self, resume_text, job_description, ats_score, features):
        if self.is_working and self.model:
            try:
                prompt = f"""You are an ATS expert. ATS Score: {ats_score}/100

Give concise feedback:
- 2 strengths
- 3 improvements

Resume: {resume_text[:1000]}
Job Description: {job_description[:1000]}"""
                
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Gemini error: {e}")
                return self._fallback_feedback(ats_score)
        else:
            return self._fallback_feedback(ats_score)

    def _fallback_feedback(self, ats_score):
        if ats_score >= 80:
            return """## 🎉 Excellent Match!

**Strengths:**
- Strong keyword alignment with job requirements
- Well-structured professional format

**Improvements:**
- Add more quantifiable achievements (numbers, percentages)
- Include recent project highlights
- Optimize technical skills section"""
        elif ats_score >= 60:
            return """## 👍 Good Match

**Strengths:**
- Relevant skills present
- Acceptable formatting

**Improvements:**
- Add measurable achievements with metrics
- Improve keyword density by 15-20%
- Strengthen professional summary section"""
        else:
            return """## 📋 Needs Improvement

**Strengths:**
- Basic content structure present

**Improvements:**
- Restructure resume with clear sections
- Add job-relevant keywords from description
- Quantify all work results with numbers
- Ensure all major sections are complete"""

    def chat_assistant(self, message, chat_history):
        if self.is_working and self.model:
            try:
                context = "You are a professional career advisor. Keep answers under 120 words. Be clear, friendly, and practical."
                
                history_text = "\n".join([f"{h['role']}: {h['message']}" for h in chat_history[-5:]])
                
                full_prompt = f"{context}\n\nConversation:\n{history_text}\n\nUser: {message}\nAssistant:"
                
                response = self.model.generate_content(full_prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Chat error: {e}")
                return self._rule_based_reply(message)
        else:
            return self._rule_based_reply(message)

    def _rule_based_reply(self, message):
        msg = message.lower()
        if 'resume' in msg or 'cv' in msg:
            return "✅ Resume Tip: Tailor your resume for each job, highlight achievements with numbers (increased sales by 30%), and keep it concise (1-2 pages)."
        elif 'interview' in msg:
            return "✅ Interview Tip: Use the STAR method (Situation, Task, Action, Result), research the company thoroughly, and prepare 3 smart questions to ask."
        elif 'job' in msg or 'search' in msg:
            return "✅ Job Search Tip: Use LinkedIn, company career sites, and job boards. Track applications in a spreadsheet and follow up after 1 week."
        elif 'skill' in msg:
            return "✅ Skills Tip: Focus on in-demand skills for your field. Take online courses (Coursera, Udemy) and build portfolio projects to demonstrate expertise."
        else:
            return "💬 I can help with resumes, interviews, job search strategies, career planning, and skill development. What would you like to know?"

# Global instance
gemini_client = None

def get_gemini_client():
    global gemini_client
    if gemini_client is None:
        gemini_client = FixedGeminiIntegration()
    return gemini_client