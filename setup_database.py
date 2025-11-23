import sqlite3
import os

def setup_database():
    """
    Initialize SQLite database with required tables
    """
    # Create directories if they don't exist
    os.makedirs('database', exist_ok=True)
    os.makedirs('uploads/resumes', exist_ok=True)
    os.makedirs('uploads/job_descriptions', exist_ok=True)
    
    db_path = 'database/ats_db.sqlite'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Job Seekers table
    c.execute('''CREATE TABLE IF NOT EXISTS job_seekers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  email TEXT UNIQUE,
                  resume_text TEXT,
                  resume_file_path TEXT,
                  skills TEXT,
                  experience TEXT,
                  education TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Job Descriptions table
    c.execute('''CREATE TABLE IF NOT EXISTS job_descriptions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  company TEXT,
                  description_text TEXT,
                  jd_file_path TEXT,
                  required_skills TEXT,
                  experience_required TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Applications table
    c.execute('''CREATE TABLE IF NOT EXISTS applications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_seeker_id INTEGER,
                  jd_id INTEGER,
                  ats_score REAL,
                  ats_category TEXT,
                  similarity_percentage REAL,
                  keyword_match REAL,
                  skill_overlap REAL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (job_seeker_id) REFERENCES job_seekers (id),
                  FOREIGN KEY (jd_id) REFERENCES job_descriptions (id))''')
    
    conn.commit()
    conn.close()
    print("✅ Database setup completed!")
    print(f"📁 Database: {db_path}")
    print("📁 Upload directories created")

if __name__ == '__main__':
    setup_database()