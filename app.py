import streamlit as st
import pandas as pd
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Can replace with 'intfloat/e5-small-v2' for better results

# Load and preprocess dataset
@st.cache_data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=['Resume'], inplace=True)
    df['Resume_clean'] = df['Resume'].apply(clean_text)
    return df

# Text preprocessing
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

# Filter resumes containing keywords from the query
def keyword_filter(df, query):
    query_keywords = query.lower().split()
    pattern = '|'.join(re.escape(word) for word in query_keywords)
    return df[df['Resume_clean'].str.contains(pattern, case=False, na=False)]

# Streamlit UI
st.set_page_config(page_title="Job-Resume Matcher", layout="wide")
st.title("🔍 Job Posting to Resume Matcher")

# Upload dataset
st.sidebar.header("Upload Resume Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    resume_df = load_and_clean_data(uploaded_file)

    st.success("✅ Resume dataset loaded!")

    job_query = st.text_input("Enter Job Description or Title", placeholder="e.g., Data Scientist with NLP experience")

    if st.button("Find Matching Candidates") and job_query:
        with st.spinner("Embedding and matching resumes..."):

            # Filter resumes by keywords
            filtered_df = keyword_filter(resume_df, job_query)

            if filtered_df.empty:
                st.warning("⚠️ No resumes matched your job keywords. Try simplifying your query.")
            else:
                query_embedding = model.encode([job_query], show_progress_bar=False)

                # Compute similarity
                filtered_df['similarity'] = filtered_df['Resume_clean'].apply(
                    lambda x: cosine_similarity(
                        [model.encode([x], show_progress_bar=False)[0]],
                        query_embedding
                    )[0][0]
                )

                # Keep only strong matches
                filtered_df = filtered_df[filtered_df['similarity'] >= 0.5]
                filtered_df = filtered_df.sort_values(by='similarity', ascending=False)

                if filtered_df.empty:
                    st.warning("⚠️ No strong matches found (similarity ≥ 0.5). Try different keywords.")
                else:
                    st.subheader("🎯 Top Matching Candidates")

                    for _, row in filtered_df.head(10).iterrows():
                        st.markdown(f"**Category:** {row['Category']}")
                        st.markdown(f"**Similarity Score:** `{row['similarity']:.4f}`")
                        st.markdown("📝 **Resume Preview:**")
                        st.code(row['Resume'][:2000])  # Limit preview length
                        st.markdown("---")
else:
    st.info("📂 Upload a CSV file containing resumes to get started.")
