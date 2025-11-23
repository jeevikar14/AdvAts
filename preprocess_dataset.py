from datasets import load_dataset
import pandas as pd
import re

# Load dataset
DATASET_NAME = "0xnbk/resume-ats-score-v1-en"
dataset = load_dataset(DATASET_NAME)

def split_resume_jd(text):
    """Split text into resume and JD parts"""
    if not text:
        return "", ""
    
    # Multiple separator patterns
    separators = ['SEP', '===', '---', 'JOB DESCRIPTION:', 'RESUME:']
    
    for sep in separators:
        if sep in text:
            parts = text.split(sep, 1)  # Split only once
            if len(parts) == 2:
                resume = parts[0].replace("RESUME:", "").strip()
                jd = parts[1].replace("JD:", "").strip()
                # Clean up any remaining separator markers
                resume = re.sub(r'SEP|===|---', '', resume).strip()
                jd = re.sub(r'SEP|===|---', '', jd).strip()
                return resume, jd
    
    # Fallback: split by length
    if len(text) > 500:
        split_point = len(text) // 2
        return text[:split_point].strip(), text[split_point:].strip()
    
    return "", ""

# Process dataset
print("🔄 Processing dataset...")
data = []
for sample in dataset['train']:
    text = sample.get('text', '')
    score = sample.get('ats_score') or sample.get('score') or sample.get('label')
    
    if text and score is not None:
        try:
            resume_text, jd_text = split_resume_jd(text)
            if resume_text and jd_text and len(resume_text) > 100 and len(jd_text) > 100:
                data.append({
                    'resume_text': resume_text,
                    'jd_text': jd_text,
                    'score': float(score)
                })
        except:
            continue

# Create DataFrame
df = pd.DataFrame(data)
print(f"✅ Processed {len(df)} samples")

# Save to CSV
df.to_csv("cleaned_dataset.csv", index=False)
print("📁 Dataset saved as 'cleaned_dataset.csv'")

# Display sample
print("\n📊 Sample data:")
print(df.head())
print(f"\n📈 Score range: {df['score'].min()} - {df['score'].max()}")
print(f"📊 Mean score: {df['score'].mean():.2f}")