import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    df = df[["Resume", "Category"]]  # Adjust columns as needed
    df.dropna(inplace=True)
    df["Resume_clean"] = df["Resume"].apply(preprocess_text)
    return df

# --- Text Cleaning ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Load Sentence Transformer ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- Embed Resumes ---
@st.cache_resource
def embed_resumes(texts):
    model = load_model()
    return model.encode(texts, convert_to_tensor=False)

# --- Cosine Similarity ---
def cosine_similarity(vec1, vec2):
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1_norm, vec2_norm)

# --- Search Function ---
def search_resumes(query, df, embeddings, debug=False):
    model = load_model()
    query = preprocess_text(query)
    query_embedding = model.encode([query], convert_to_tensor=False)[0]

    scores = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    df['similarity'] = scores

    if debug:
        for i, row in df.iterrows():
            print(f"{row['Category']}: {row['similarity']:.4f}")

    return df.sort_values(by='similarity', ascending=False)

# --- Main App ---
def main():
    st.set_page_config(page_title="Resume Matcher", layout="wide")
    st.title("📄 Intelligent Resume Search System")

    st.markdown("""
        <style>
        .highlight-score { color: #FF5733; font-weight: bold; }
        .candidate-name { font-size: 20px; font-weight: 600; }
        </style>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ About"):
        st.write("Search resumes using semantic understanding of your job query powered by AI.")

    query = st.text_input("Job Query", placeholder="e.g. React developer with UI UX experience")
    submit = st.button("🔍 Search", use_container_width=True)

    df = load_data()
    resume_embeddings = embed_resumes(df["Resume_clean"].tolist())

    if submit and query:
        results = search_resumes(query, df.copy(), resume_embeddings)

        st.sidebar.header("⚙️ Filter Options")
        min_score = st.sidebar.slider("Minimum Similarity Score", 0.0, 1.0, 0.25, 0.01)
        filtered = results[results["similarity"] >= min_score]
        st.sidebar.write(f"Filtered candidates: {len(filtered)}")

        st.subheader("🎯 Top Matching Candidates")
        if filtered.empty:
            st.warning("No matching candidates found. Try lowering the similarity score or simplifying your query.")
        else:
            for _, row in filtered.iterrows():
                with st.container():
                    st.markdown(f"<div class='candidate-name'>{row['Category']}</div>", unsafe_allow_html=True)
                    st.markdown(f"**Similarity Score:** <span class='highlight-score'>{row['similarity']:.4f}</span>", unsafe_allow_html=True)
                    with st.expander("📝 Resume Preview"):
                        st.write(row['Resume'])
                    st.markdown("---")

if __name__ == "__main__":
    main()
