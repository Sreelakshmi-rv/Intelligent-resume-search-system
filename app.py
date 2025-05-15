import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Sample Data
data = {
    "id": list(range(1, 16)),
    "name": [
        "Alice Johnson", "Bob Lee", "Carol Wong", "David Kim", "Eve Smith",
        "Frank Miller", "Grace Liu", "Henry Adams", "Isla Patel", "Jack Nguyen",
        "Kavya Raj", "Leo Thomas", "Meera Varma", "Noah Singh", "Olivia Costa"
    ],
    "email": [
        "alice.johnson@example.com", "bob.lee@example.com", "carol.wong@example.com", "david.kim@example.com", "eve.smith@example.com",
        "frank.miller@example.com", "grace.liu@example.com", "henry.adams@example.com", "isla.patel@example.com", "jack.nguyen@example.com",
        "kavya.raj@example.com", "leo.thomas@example.com", "meera.varma@example.com", "noah.singh@example.com", "olivia.costa@example.com"
    ],
    "resume_text": [
        "Python backend developer with 4 years of experience in Django and Flask frameworks. Skilled in REST APIs and microservices.",
        "Frontend engineer specializing in React, Redux, and responsive web design. 3 years experience building SaaS applications.",
        "Full stack developer experienced with Python, JavaScript, AWS cloud, and containerization (Docker, Kubernetes).",
        "Software engineer with expertise in data science, machine learning using Python and R. Published research papers on NLP.",
        "Backend developer focused on Node.js, Express, MongoDB, and GraphQL. 5 years of experience in scalable API development.",
        "Cloud architect with deep understanding of Azure and AWS. Expert in designing CI/CD pipelines and cloud security policies.",
        "Mobile app developer with Flutter and Kotlin experience. Built e-commerce and social apps with over 100k downloads.",
        "DevOps engineer with strong knowledge in Terraform, Jenkins, GitHub Actions, Docker, and Kubernetes clusters.",
        "AI engineer with hands-on experience in LLMs, OpenAI API, transformers, and building chatbots and recommendation systems.",
        "Business analyst with SQL, Tableau, and Python proficiency. Worked on multiple cross-functional data analytics projects.",
        "Cybersecurity analyst skilled in penetration testing, Wireshark, Nmap, and implementing secure network architectures.",
        "UI/UX designer with a keen eye for aesthetics and usability. Proficient in Figma, Adobe XD, and design thinking process.",
        "ML engineer with experience in time-series forecasting, CNNs, and large-scale model training on cloud platforms.",
        "Blockchain developer with Solidity, smart contracts, and experience in building DeFi and NFT marketplaces on Ethereum.",
        "Data engineer with Spark, Hadoop, and Airflow experience. Specialized in building data pipelines and ETL workflows."
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
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
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

    with st.expander("\u2139 About this App"):
        st.write("This tool lets you search and rank candidate resumes by relevance to a job description using AI-powered semantic search.")

    query = st.text_input("Job Query", placeholder="Enter job description or keywords...")
    submit = st.button("\U0001F50D Search", use_container_width=True)

    resume_embeddings = embed_resumes(resumes_df['resume_text'].tolist())

    if submit and query:
        results = search_resumes(query, resumes_df.copy(), resume_embeddings)

        st.sidebar.header("\u2699 Filter & Actions")
        min_score = st.sidebar.slider("Minimum Similarity Score", 0.0, 1.0, 0.5, 0.01)
        filtered = results[results['similarity'] >= min_score]
        st.sidebar.write(f"Filtered candidates: {len(filtered)}")

        st.sidebar.markdown("### Save or Send")
        for idx, row in filtered.iterrows():
            if st.sidebar.button(f"Save {row['name']}"):
                st.sidebar.success(f"{row['name']}'s profile saved!")
            if st.sidebar.button(f"Send {row['name']} via Email"):
                st.sidebar.info(f"{row['name']}'s profile sent to your email (simulated).")

        st.sidebar.markdown("---")
        st.sidebar.markdown("Need help? Hover over elements or contact us below.")
        if st.sidebar.button("Send Feedback"):
            st.sidebar.warning("Feedback feature is under construction.")

        st.subheader("Top Matching Candidates")
        for i, row in filtered.iterrows():
            with st.container():
                st.markdown(f"<div class='candidate-name'>{row['name']}</div>", unsafe_allow_html=True)
                st.markdown(f"**Email:** {row['email']}")
                st.markdown(f"**Similarity Score:** <span class='highlight-score'>{row['similarity']:.4f}</span>", unsafe_allow_html=True)
                with st.expander("Show Resume Snippet"):
                    st.write(row['resume_text'])
                st.markdown("---")

if __name__ == "__main__":
    main()
