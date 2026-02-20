import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("CareerMatch AI - Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_description = st.text_area("Paste Job Description")

# Skill list (you can expand later)
skills_list = [
    "python", "java", "c++", "machine learning",
    "deep learning", "data analysis", "sql",
    "excel", "tensorflow", "pandas",
    "numpy", "nlp", "power bi"
]

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text.lower()

if uploaded_file and job_description:

    resume_text = extract_text_from_pdf(uploaded_file)
    job_description = job_description.lower()

    # Similarity
    documents = [resume_text, job_description]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    match_percentage = round(similarity[0][0] * 100, 2)

    st.subheader(f"Match Score: {match_percentage}%")

    # Skill extraction
    resume_skills = [skill for skill in skills_list if skill in resume_text]
    jd_skills = [skill for skill in skills_list if skill in job_description]
    missing_skills = list(set(jd_skills) - set(resume_skills))

    # --- UI PART BELOW MUST BE HERE ---
    st.markdown("### ✅ Skills Found in Resume")
    if resume_skills:
        for skill in resume_skills:
            st.markdown(f"- {skill}")
    else:
        st.markdown("No relevant skills found.")

    st.markdown("### 📌 Skills Required in Job")
    if jd_skills:
        for skill in jd_skills:
            st.markdown(f"- {skill}")
    else:
        st.markdown("No specific skills detected in job description.")

    st.markdown("### ❌ Missing Skills")
    if missing_skills:
        for skill in missing_skills:
            st.markdown(f"- {skill}")
    else:
        st.markdown("No missing skills. Good match!")

    if missing_skills:
        st.warning("You should consider learning these skills to improve your match score.")
    else:
        st.success("Great! Your resume matches the job description well.")