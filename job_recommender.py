import os
import requests
from datetime import datetime, timedelta

class EnhancedJobRecommender:
    """Enhanced job recommendations with SerpAPI fallback"""
    
    def __init__(self):
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        self.serpapi_working = False
        
        # Test SerpAPI if key exists
        if self.serpapi_key:
            try:
                test_params = {
                    'engine': 'google_jobs',
                    'q': 'python developer',
                    'hl': 'en',
                    'api_key': self.serpapi_key,
                    'num': 1
                }
                response = requests.get('https://serpapi.com/search', params=test_params, timeout=5)
                if response.status_code == 200:
                    self.serpapi_working = True
                    print("✅ SerpAPI working")
            except Exception as e:
                print(f"⚠️ SerpAPI test failed: {e}")
    
    def get_job_recommendations(self, skills, experience, location="Remote", limit=5):
        """Get job recommendations with fallback"""
        
        if self.serpapi_working:
            try:
                return self._get_real_jobs(skills, experience, location, limit)
            except Exception as e:
                print(f"SerpAPI error: {e}")
                return self._get_enhanced_recommendations(skills, experience, location, limit)
        else:
            print("ℹ️ Using enhanced recommendations (SerpAPI not available)")
            return self._get_enhanced_recommendations(skills, experience, location, limit)
    
    def _get_real_jobs(self, skills, experience, location, limit):
        """Fetch real jobs from SerpAPI"""
        seniority = self._get_seniority_level(experience)
        primary_skill = skills[0] if skills else "software"
        query = f"{seniority} {primary_skill} developer {location}"
        
        params = {
            'engine': 'google_jobs',
            'q': query,
            'hl': 'en',
            'api_key': self.serpapi_key,
            'num': limit
        }
        
        response = requests.get('https://serpapi.com/search', params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        jobs = data.get('jobs_results', [])
        
        if jobs:
            recommendations = []
            for job in jobs[:limit]:
                recommendations.append({
                    'title': job.get('title', 'Position Available'),
                    'company': job.get('company_name', 'Company'),
                    'location': job.get('location', location),
                    'description': (job.get('description', '')[:200] + '...') if len(job.get('description', '')) > 200 else job.get('description', 'No description available'),
                    'salary': job.get('detected_extensions', {}).get('salary', 'Competitive salary'),
                    'source_url': job.get('apply_link', '#'),
                    'is_generated': False,
                    'posted_date': 'Recently'
                })
            
            print(f"✅ Found {len(recommendations)} real jobs from SerpAPI")
            return recommendations
        else:
            return self._get_enhanced_recommendations(skills, experience, location, limit)
    
    def _get_enhanced_recommendations(self, skills, experience, location, limit):
        """Enhanced fallback recommendations"""
        seniority = self._get_seniority_level(experience)
        primary_skill = skills[0] if skills else "Software"
        
        companies = [
            "Tech Solutions Inc", "DataWorks Corp", "Cloud Innovations", 
            "Digital Transform LLC", "SoftTech Systems", "CodeCraft Studios",
            "Innovate Labs", "FutureTech Enterprises", "SmartDev Solutions",
            "Quantum Computing Inc", "AI Innovations", "BlockChain Tech"
        ]
        
        job_titles = self._generate_job_titles(skills, seniority)
        locations = [location, "Remote", "Hybrid", f"{location} / Remote"]
        
        recommendations = []
        for i in range(limit):
            company = companies[i % len(companies)]
            job_title = job_titles[i % len(job_titles)]
            job_location = locations[i % len(locations)]
            
            base_salary = self._calculate_salary(seniority, experience, skills)
            
            job = {
                'title': job_title,
                'company': company,
                'location': job_location,
                'description': self._generate_job_description(job_title, skills, seniority),
                'salary': f"${base_salary:,} - ${base_salary + 15000:,}",
                'source_url': None,
                'is_generated': True,
                'posted_date': (datetime.now() - timedelta(days=i % 7)).strftime('%Y-%m-%d')
            }
            recommendations.append(job)
        
        return recommendations
    
    def _generate_job_titles(self, skills, seniority):
        """Generate relevant job titles"""
        titles = []
        primary_skill = skills[0] if skills else "Software"
        
        if any(skill in skills for skill in ['Python', 'Java', 'JavaScript', 'React', 'Node.js']):
            titles.extend([
                f"{seniority} {primary_skill} Developer",
                f"{seniority} Software Engineer",
                f"Full Stack {primary_skill} Developer",
                f"{primary_skill} Backend Engineer"
            ])
        
        if any(skill in skills for skill in ['Data Analysis', 'Machine Learning', 'SQL', 'Python']):
            titles.extend([
                f"{seniority} Data Analyst",
                f"{seniority} Data Scientist",
                f"Machine Learning Engineer"
            ])
        
        if len(titles) < 4:
            titles.extend([
                f"{seniority} Software Developer",
                f"{seniority} Technical Specialist",
                "Software Engineer",
                "Full Stack Developer"
            ])
        
        return titles[:8]
    
    def _calculate_salary(self, seniority, experience, skills):
        """Calculate realistic salary"""
        base_salary = 60000
        experience_bonus = experience * 3000
        
        seniority_bonus = {
            'Entry-Level': 0,
            'Junior': 5000,
            'Mid-Level': 15000,
            'Senior': 30000,
            'Lead': 45000
        }.get(seniority, 0)
        
        skill_premium = 0
        high_demand_skills = ['Machine Learning', 'AI', 'Python', 'React', 'AWS', 'Docker', 'Kubernetes']
        for skill in skills:
            if skill in high_demand_skills:
                skill_premium += 5000
        
        total_salary = base_salary + experience_bonus + seniority_bonus + min(skill_premium, 20000)
        return min(total_salary, 180000)
    
    def _generate_job_description(self, job_title, skills, seniority):
        """Generate realistic job description"""
        primary_skills = ', '.join(skills[:3]) if skills else "various technologies"
        
        descriptions = [
            f"Seeking a {seniority.lower()} professional for {job_title} role. Required skills: {primary_skills}. "
            f"Responsibilities include developing scalable applications and collaborating with cross-functional teams.",
            
            f"Join our team as a {job_title} working with cutting-edge technologies. "
            f"Must have expertise in {primary_skills}. Strong problem-solving skills and agile experience required.",
            
            f"We're hiring a {job_title} to build innovative solutions. "
            f"Key technologies: {primary_skills}. Work on challenging projects with a talented team."
        ]
        
        return descriptions[len(job_title) % len(descriptions)]
    
    def _get_seniority_level(self, experience):
        """Determine seniority level"""
        if experience >= 8: return "Lead"
        elif experience >= 5: return "Senior"
        elif experience >= 3: return "Mid-Level"
        elif experience >= 1: return "Junior"
        else: return "Entry-Level"