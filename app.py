import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# Load and preprocess the updated resume dataset
@st.cache_data
def load_dataset():
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    df.dropna(inplace=True)
    df['Resume'] = df['Resume'].apply(clean_text)
    df = df.reset_index(drop=True)
    df['id'] = df.index + 1
    df['name'] = "Candidate " + df['id'].astype(str)
    df['email'] = ["candidate" + str(i+1) + "@example.com" for i in range(len(df))]
    return df

# Text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)                # remove URLs
    text = re.sub(r'<.*?>', '', text)                  # remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)         # remove special characters
    text = re.sub(r'\s+', ' ', text)                   # remove extra whitespace
    return text.strip()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def embed_resumes(texts):
    model = load_model()
    embeddings = model.encode(texts, convert_to_tensor=False)
    return np.array(embeddings)

def cosine_similarity(vec1, vec2):
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1_norm, vec2_norm)

def search_resumes(query, resumes_df, resume_embeddings):
    query_embedding = load_model().encode([query], convert_to_tensor=False)[0]
    scores = [cosine_similarity(query_embedding, emb) for emb in resume_embeddings]
    resumes_df['similarity'] = scores
    sorted_df = resumes_df.sort_values(by='similarity', ascending=False)
    return sorted_df

def main():
    st.set_page_config(page_title="Resume Matcher", layout="wide")
    st.title("\U0001F4C4 Intelligent Resume Search System")

    st.markdown("""
        <style>
        .highlight-score {
            color: #FF5733;
            font-weight: bold;
        }
        .candidate-name {
            font-size: 24px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ About this App"):
        st.write("This tool uses semantic search to find the most relevant resumes for a given job description.")

    query = st.text_input("Job Query", placeholder="Enter job description or keywords...")
    submit = st.button("🔍 Search", use_container_width=True)

    resumes_df = load_dataset()
    resume_embeddings = embed_resumes(resumes_df['Resume'].tolist())

    if submit and query:
        results = search_resumes(query, resumes_df.copy(), resume_embeddings)

        st.sidebar.header("⚙️ Filter & Actions")
        min_score = st.sidebar.slider("Minimum Similarity Score", 0.0, 1.0, 0.5, 0.01)
        filtered = results[results['similarity'] >= min_score]
        st.sidebar.write(f"Filtered candidates: {len(filtered)}")

        st.subheader("Top Matching Candidates")
        for i, row in filtered.iterrows():
            with st.container():
                st.markdown(f"<div class='candidate-name'>{row['name']}</div>", unsafe_allow_html=True)
                st.markdown(f"**Email:** {row['email']}")
                st.markdown(f"**Category:** {row['Category']}")
                st.markdown(f"**Similarity Score:** <span class='highlight-score'>{row['similarity']:.4f}</span>", unsafe_allow_html=True)
                with st.expander("Show Resume Snippet"):
                    st.write(row['Resume'])
                st.markdown("---")

if __name__ == "__main__":
    main()
