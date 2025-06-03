import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re

# Load models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocessing function
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W', ' ', text)
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha])

st.title("🔍 Resume Matcher")

uploaded_file = st.file_uploader("Upload Resume CSV", type="csv")
job_input = st.text_area("Enter Job Description", "")

if uploaded_file and job_input:
    df = pd.read_csv(uploaded_file)
    df['cleaned_resume'] = df['Resume'].apply(clean_text)
    resume_embeddings = model.encode(df['cleaned_resume'].tolist(), show_progress_bar=False)
    job_embedding = model.encode([clean_text(job_input)])
    
    similarity_scores = cosine_similarity(resume_embeddings, job_embedding)
    df['Match Score'] = similarity_scores.flatten()
    top_matches = df.sort_values(by='Match Score', ascending=False)

    st.success("Top 5 Matching Resumes:")
    st.dataframe(top_matches[['Resume', 'Category', 'Match Score']].head(5))
