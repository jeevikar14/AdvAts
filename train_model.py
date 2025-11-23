import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

# Set up page
st.set_page_config(page_title="Resume-Job Matcher", layout="wide")

# Config
MODEL_CONFIG = {
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'ElasticNet': ElasticNet(random_state=42, max_iter=5000)
}

PARAM_GRIDS = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
}

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

class ProgressTracker:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.step = 0
        self.progress_bar = st.progress(0, "Starting...")
        self.status_text = st.empty()
        self.log_placeholder = st.empty()
        self.log_messages = []

    def update(self, step_offset, message):
        self.step += step_offset
        percent = min(100, int((self.step / self.total_steps) * 100))
        self.progress_bar.progress(percent, text=message)
        self.status_text.markdown(f"**{message}**")
        time.sleep(0.01)

    def log(self, text):
        self.log_messages.append(text)
        with self.log_placeholder.expander("Logs", expanded=False):
            st.text('\n'.join(self.log_messages[::-1]))

def batch_embed(tracker, model, texts, name, width, batch_size=64):
    total = (len(texts) + batch_size - 1) // batch_size
    embs = []
    inc = width / total
    for i in range(0, len(texts), batch_size):
        tracker.update(inc, f"Embedding {name} batch {i//batch_size+1}/{total}...")
        embs.extend(model.encode(texts[i:i+batch_size], show_progress_bar=False))
    return np.array(embs)

def main():
    st.title("🎯 Resume-Job Matching Score Predictor")
    
    st.sidebar.header("Configuration")
    search_iter = st.sidebar.slider("Search Iterations", 1, 15, 5)
    cv_folds = st.sidebar.slider("CV Folds", 2, 5, 3)

    uploaded_file = st.file_uploader("Upload CSV (jd_text, resume_text, score)", type='csv')
    
    if uploaded_file:
        tracker = ProgressTracker(10)
        try:
            # 1. Load
            tracker.update(0.5, "Loading data...")
            df = pd.read_csv(uploaded_file).dropna(subset=['jd_text', 'resume_text', 'score'])
            
            X_jd = df['jd_text'].values
            X_res = df['resume_text'].values
            y = df['score'].values
            X_comb = df['jd_text'].astype(str) + " " + df['resume_text'].astype(str)

            tracker.update(0.5, "Splitting...")
            X_jd_tr, X_jd_te, y_tr, y_te = train_test_split(X_jd, y, test_size=0.2, random_state=42)
            X_res_tr, X_res_te, _, _ = train_test_split(X_res, y, test_size=0.2, random_state=42)
            X_comb_tr, X_comb_te, _, _ = train_test_split(X_comb, y, test_size=0.2, random_state=42)

            # 2. Embeddings
            model = get_embedding_model()
            
            # Embed JD
            jd_tr = batch_embed(tracker, model, X_jd_tr, "JD", 0.75)
            jd_te = batch_embed(tracker, model, X_jd_te, "JD", 0.75)
            
            # Embed Resume
            res_tr = batch_embed(tracker, model, X_res_tr, "Resume", 0.75)
            res_te = batch_embed(tracker, model, X_res_te, "Resume", 0.75)
            
            # Embed Combined
            tracker.update(0.5, "Embedding Combined...")
            comb_tr = model.encode(X_comb_tr)
            comb_te = model.encode(X_comb_te)

            # 3. Feature Engineering (Cosine Similarity)
            tracker.update(1.0, "Calculating Similarity...")
            sim_tr = cosine_similarity(jd_tr, res_tr).diagonal().reshape(-1, 1)
            sim_te = cosine_similarity(jd_te, res_te).diagonal().reshape(-1, 1)

            # Stack Features
            X_train = np.hstack([comb_tr, sim_tr])
            X_test = np.hstack([comb_te, sim_te])

            # Scale
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # 4. Training
            results = {}
            step_inc = 3.0 / len(MODEL_CONFIG)
            
            for name, clf in MODEL_CONFIG.items():
                tracker.update(0, f"Training {name}...")
                if name in PARAM_GRIDS:
                    search = RandomizedSearchCV(clf, PARAM_GRIDS[name], n_iter=search_iter, cv=cv_folds, n_jobs=-1, scoring='r2')
                    search.fit(X_train, y_tr)
                    best = search.best_estimator_
                else:
                    best = clf.fit(X_train, y_tr)
                
                pred = best.predict(X_test)
                r2 = r2_score(y_te, pred)
                results[name] = {'model': best, 'r2': r2, 'mae': mean_absolute_error(y_te, pred)}
                tracker.update(step_inc, f"{name} R²: {r2:.4f}")

            # 5. Ensemble & Results
            tracker.update(1.0, "Finalizing...")
            ens_pred = np.mean([r['model'].predict(X_test) for r in results.values()], axis=0)
            ens_r2 = r2_score(y_te, ens_pred)
            ens_mae = mean_absolute_error(y_te, ens_pred)

            st.subheader("Results")
            res_df = pd.DataFrame({k: {'R²': v['r2'], 'MAE': v['mae']} for k, v in results.items()}).T
            st.dataframe(res_df.style.apply(lambda x: ['background: #d1e7dd' if v > 0.5 else '' for v in x], subset=['R²']))

            col1, col2 = st.columns(2)
            col1.metric("Ensemble R²", f"{ens_r2:.4f}")
            col2.metric("Ensemble MAE", f"{ens_mae:.4f}")

            if ens_r2 > 0.5:
                st.success(f"✅ Goal Met: R² {ens_r2:.4f} > 0.5")
            else:
                st.warning("⚠️ Goal Not Met. Increase search iterations.")

            st.download_button("Download Model", joblib.dumps(results), "model.pkl")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()