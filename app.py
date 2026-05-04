import streamlit as st
import pdfplumber
import spacy
import re

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# ------------------ TITLE ------------------
st.title("AI Resume Analyzer")

# ------------------ SKILLS DATABASE ------------------
skills_list = [
    "python", "java", "c++", "sql",
    "machine learning", "data analysis",
    "html", "css", "javascript", "deep learning"
]

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:

    st.success("File uploaded successfully!")

    # ------------------ TEXT EXTRACTION ------------------
    with pdfplumber.open(uploaded_file) as pdf:
        text = "".join(page.extract_text() or "" for page in pdf.pages)

    text_lower = text.lower()

    # ------------------ NLP PROCESSING ------------------
    doc = nlp(text)

    # ------------------ NAME EXTRACTION ------------------
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    # ------------------ EMAIL EXTRACTION ------------------
    email = re.findall(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+", text)
    email = email[0] if email else "Not found"

    # ------------------ SKILL DETECTION ------------------
    found_skills = [skill for skill in skills_list if skill in text_lower]

    # ------------------ SCORE ------------------
    score = (len(found_skills) / len(skills_list)) * 100

    # ------------------ UI DISPLAY ------------------
    st.header("Resume Analysis Report")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Candidate Info")
        st.write("Name:", name if name else "Not found")
        st.write("Email:", email)

    with col2:
        st.subheader("Score")
        st.write(f"{score:.2f}%")

    # ------------------ SKILLS ------------------
    st.subheader("Detected Skills")
    st.write(found_skills if found_skills else "No skills detected")

    # ------------------ JOB MATCHING ------------------
    st.subheader("Job Matching")

    jd = st.text_area("Paste Job Description")

    if jd:
        jd_lower = jd.lower()

        jd_skills = [skill for skill in skills_list if skill in jd_lower]
        matched_skills = [skill for skill in found_skills if skill in jd_skills]

        match_score = (len(matched_skills) / len(jd_skills)) * 100 if jd_skills else 0

        st.write("Matched Skills:", matched_skills)
        st.write(f"Match Score: {match_score:.2f}%")

    # ------------------ SUGGESTIONS ------------------
    missing_skills = [skill for skill in skills_list if skill not in found_skills]

    st.subheader("💡 Suggestions")
    st.write(missing_skills if missing_skills else "No missing skills ")