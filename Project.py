import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import spacy
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = "".join([page.extract_text() or "" for page in pdf.pages])
    return text.strip()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

def extract_skills(text, skill_list):
    extracted_skills = set()
    for skill in skill_list:
        if skill.lower() in text.lower():
            extracted_skills.add(skill)
    return list(extracted_skills)

def extract_experience(text):
    experience_patterns = [r'(\d+)\s+years?', r'(\d+)\s+yrs?', r'(\d+)\+?\s+years?']
    for pattern in experience_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0

def compute_similarity(resume_texts, job_description):
    vectorizer = TfidfVectorizer()
    corpus = [job_description] + resume_texts
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return similarity_scores.flatten()

def main():
    st.title("🚀 AI Resume Screening & Ranking System")
    job_description = st.text_area("📝 Enter Job Description:")
    uploaded_files = st.file_uploader("📤 Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
    
    if st.button("Analyze & Rank Candidates") and uploaded_files:
        skill_list = ["Python", "Machine Learning", "Data Science", "Java", "SQL", "Cloud Computing"]
        resume_texts = []
        candidates = []
        
        for pdf in uploaded_files:
            text = extract_text_from_pdf(pdf)
            processed_text = preprocess_text(text)
            skills = extract_skills(processed_text, skill_list)
            experience = extract_experience(text)
            resume_texts.append(processed_text)
            candidates.append({"name": pdf.name, "skills": skills, "experience": experience})
        
        similarity_scores = compute_similarity(resume_texts, preprocess_text(job_description))
        
        for i, candidate in enumerate(candidates):
            candidate["score"] = similarity_scores[i] * 100
        
        df = pd.DataFrame(candidates)
        df = df.sort_values(by=["score", "experience"], ascending=[False, False])
        
        st.write("### 📊 Ranked Candidates")
        st.dataframe(df)

if __name__ == "__main__":
    main()