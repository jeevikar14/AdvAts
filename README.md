🎯 Advanced ATS — AI-Powered Resume Screening & Job Matching

An intelligent AI-driven Applicant Tracking System helping job seekers and recruiters make fast, data-driven hiring decisions using machine learning, NLP, and embeddings.

📋 Overview

Advanced ATS combines resume scoring, classification, experience detection, job recommendations, and recruiter decision prediction to streamline the hiring process.

✨ Key Capabilities:

ATS Resume Scoring (0–100) with explainable breakdown

Automatic Resume Classification into 25 job categories

Experience-Level Detection: Entry / Mid / Senior

Job Recommendations filtered by skills, experience & location

AI Career Assistant for resume feedback

Recruiter Decision Prediction (hire/reject prioritization)

🚀 Features

Intelligent Resume Scoring: Keyword + semantic scoring with ML refinement

Resume Classification: TF-IDF + optimized classifiers for 25 categories

Experience Detection: Engineered features + PCA + K-Means clustering

Job Recommender: Real-time job search via SerpAPI

AI Career Assistant: Google Gemini-based feedback & suggestions

Recruiter Prediction: Embedding-based historical decision modeling

🔧 Tech Stack

ML / AI: scikit-learn, SentenceTransformers, NumPy, Pandas

LLM / Feedback: Google Gemini

Web / UI: Streamlit

Database: SQLite

APIs: SerpAPI (jobs), Google Generative AI (feedback)

🎯 Getting Started
Prerequisites

Python 3.8+

Clone & Install
git clone https://github.com/jeevikar14/AdvAts
cd AdvAts
pip install -r requirements.txt

Environment Variables

Create a .env file in the project root:

GEMINI_API_KEY=your_gemini_api_key_here
SERPAPI_KEY=your_serpapi_key_here

Initialize DB & Train
python setup_database.py
python train_ats_score.py
python resume_classifier.py
python experience_classifier.py
python train_recruitor_decision_model.py

Run App
streamlit run app.py

🔍 How It Works

1️⃣ ATS Score Calculation

Keyword Matching (40%) → skills, experience, education

Semantic Similarity (60%) → embeddings via all-MiniLM-L6-v2

Weighted features → Random Forest predicts 0–100

2️⃣ Resume Classification

Preprocessing → TF-IDF (top 3k features) → optimized classifier → GridSearchCV

3️⃣ Experience Detection

Feature extraction → PCA → K-Means clustering → Entry / Mid / Senior

4️⃣ Job Recommendation

Query SerpAPI → filter by skills, experience, location

5️⃣ Recruiter Decision Prediction

Embeddings of resumes, JDs, transcripts → classifier → hire/reject prioritization

📊 Model Performance

ATS Scoring: R² ≈ 77.6%, MAE ≈ 4.59 (Random Forest)

Resume Classifier: Accuracy ≈ 99%, F1 ≈ 0.99 (LogReg)

Experience Clustering: Silhouette ≈ 0.41 (PCA + K-Means)

🎯 Use Cases

Job Seekers: Instant ATS score, actionable feedback, role matching

Recruiters: Automated screening, candidate ranking, experience filtering

🧪 Testing

Run system validation:

python test_system.py


Tests include imports, inference engine, Gemini integration, recommender, and database checks.

🤖 API Keys & Fallbacks

Google Gemini: Optional; fallback → rule-based feedback

SerpAPI: Optional; fallback → local job examples

📌 Notes

Resume DB → database/ats_db.sqlite

Uploaded files → uploads/

Predictions run locally (except optional API calls)

📄 Acknowledgments

Dataset: UpdatedResumeDataset.csv

Sentence Transformers by UKPLab

Google Gemini AI

SerpAPI for job search