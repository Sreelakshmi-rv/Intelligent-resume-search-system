import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load or define resumes data
data = {
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice Johnson", "Bob Lee", "Carol Wong", "David Kim", "Eve Smith"],
    "email": [
        "alice.johnson@example.com",
        "bob.lee@example.com",
        "carol.wong@example.com",
        "david.kim@example.com",
        "eve.smith@example.com"
    ],
    "resume_text": [
        "Python backend developer with 4 years of experience in Django and Flask frameworks. Skilled in REST APIs and microservices.",
        "Frontend engineer specializing in React, Redux, and responsive web design. 3 years experience building SaaS applications.",
        "Full stack developer experienced with Python, JavaScript, AWS cloud, and containerization (Docker, Kubernetes).",
        "Software engineer with expertise in data science, machine learning using Python and R. Published research papers on NLP.",
        "Backend developer focused on Node.js, Express, MongoDB, and GraphQL. 5 years of experience in scalable API development."
    ]
}
resumes_df = pd.DataFrame(data)

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

def search_resumes(query, resumes_df, resume_embeddings, top_k=3):
    query_embedding = load_model().encode([query], convert_to_tensor=False)[0]
    scores = [cosine_similarity(query_embedding, emb) for emb in resume_embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        candidate = resumes_df.iloc[idx]
        results.append({
            "id": candidate['id'],
            "name": candidate['name'],
            "email": candidate['email'],
            "similarity_score": scores[idx],
            "resume_text": candidate['resume_text']
        })
    return results

def main():
    st.title("Intelligent Resume Search System")
    st.write("Enter a job description or query to find matching candidate resumes.")
    model = load_model()
    resume_embeddings = embed_resumes(resumes_df['resume_text'].tolist())
    query = st.text_input("Job Query", value="Python backend developer with 3 years of experience")
    if query:
        results = search_resumes(query, resumes_df, resume_embeddings, top_k=3)
        st.subheader("Top Matching Candidates:")
        for i, candidate in enumerate(results, start=1):
            st.markdown(f"### [{i}] {candidate['name']}")
            st.write(f"**Email:** {candidate['email']}")
            st.write(f"**Similarity Score:** {candidate['similarity_score']:.4f}")
            st.write(f"**Resume Snippet:** {candidate['resume_text'][:200]}…")

if __name__ == "__main__":
    main()
