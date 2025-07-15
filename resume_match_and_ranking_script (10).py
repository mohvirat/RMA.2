# Resume Match and Ranking Script with Experience Check (OpenAI SDK v1.0+)

import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import openai
import os
import io
import difflib
import re

# Verify and set OpenAI API Key only after login
api_key = None

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def calculate_similarity(text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([jd_text, text])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

def generate_ai_summary(text):
    prompt = f"""
    Analyze the following resume content:

    {text[:4000]}

    Provide a summary with:
    - Key strengths
    - Weaknesses
    - Technologies used
    - Project experience
    Keep it clear and concise.
    """
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional technical recruiter."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Summary failed: {e}"

def fuzzy_match(term, text, threshold=0.6):
    words = text.split()
    matches = [word for word in words if difflib.SequenceMatcher(None, word.lower(), term.lower()).ratio() >= threshold]
    return bool(matches)

def experience_filter(skill, resume_text, required_years):
    prompt = f"""
    Based on the resume content below, estimate how many years of experience the candidate has in the skill: {skill}. Respond with only a number.

    Resume:
    {resume_text[:3000]}
    """
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You estimate experience for skills from resume text. Respond with only a number."},
                {"role": "user", "content": prompt}
            ]
        )
        years = float(response.choices[0].message.content.strip())
        return years >= required_years
    except:
        return False

def evaluate_skills_with_experience(resume_text, skill_experience_map):
    resume_lower = resume_text.lower()
    skill_results = {}
    for skill, required_years in skill_experience_map.items():
        found = fuzzy_match(skill, resume_lower) and experience_filter(skill, resume_text, required_years)
        skill_results[skill] = "🟢" if found else "🔴"

    essential_skills = {k: v for k, v in skill_experience_map.items() if st.session_state.skill_type_map.get(k) == "essential"}
    preferred_skills = {k: v for k, v in skill_experience_map.items() if st.session_state.skill_type_map.get(k) == "preferred"}

    essential_match = sum(skill_results[k] == "🟢" for k in essential_skills)
    preferred_match = sum(skill_results[k] == "🟢" for k in preferred_skills)

    essential_score = (essential_match / len(essential_skills)) * 70 if essential_skills else 0
    preferred_score = (preferred_match / len(preferred_skills)) * 30 if preferred_skills else 0
    weighted_score = round(essential_score + preferred_score, 2)

    return skill_results, weighted_score, essential_match, preferred_match

def color_match_level(score):
    if score >= 80:
        return "🟢 High"
    elif score >= 60:
        return "🟡 Medium"
    else:
        return "🔴 Low"

def extract_skills_from_jd(jd_text):
    prompt = f"""
    Extract two categorized skill lists from the job description below:
    - Essential Skills (must-have, core requirements)
    - Preferred Skills (nice-to-have or secondary skills)
    Provide only comma-separated lists without additional explanation.

    Job Description:
    {jd_text}
    """
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract and classify technical skills from job descriptions."},
                {"role": "user", "content": prompt}
            ]
        )
        output = response.choices[0].message.content.strip()
        lines = output.splitlines()
        essentials, preferreds = [], []
        for line in lines:
            if "Essential" in line:
                essentials = [s.strip() for s in line.split(":")[-1].split(",")]
            elif "Preferred" in line:
                preferreds = [s.strip() for s in line.split(":")[-1].split(",")]
        return essentials[:10], preferreds[:10]
    except Exception as e:
        return [], []

# ---------------------------
# STREAMLIT INTERFACE
# ---------------------------

def main():
    st.set_page_config(page_title="Secure Resume Matcher", layout="centered")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        with st.form("auth_form"):
            username = st.text_input("Enter Username")
            password = st.text_input("Enter Password", type="password")
            submitted = st.form_submit_button("🔐 Login")

        if submitted and username == "Virat" and password == "KleisTech@123":
            st.session_state.authenticated = True
            st.experimental_rerun()
        elif submitted:
            st.error("❌ Access denied. Please check your credentials.")

    if st.session_state.authenticated:
        global api_key
        api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.warning("⚠️ OpenAI API key not found. AI Summary generation may not work.")

        st.title("📄 Resume Matcher with Experience Checks")

        with st.sidebar:
            st.info("Paste a Job Description and upload multiple PDF resumes.")
            jd_text_input = st.text_area("Paste Job Description Below", height=200)

            essential_input = ""
            preferred_input = ""
            if jd_text_input:
                e_skills, p_skills = extract_skills_from_jd(jd_text_input)
                essential_input_edit = st.text_area("Edit Essential Skills (comma-separated)", ", ".join(e_skills))
                preferred_input_edit = st.text_area("Edit Preferred Skills (comma-separated)", ", ".join(p_skills))
                essential_input = [s.strip() for s in essential_input_edit.split(",") if s.strip()]
                preferred_input = [s.strip() for s in preferred_input_edit.split(",") if s.strip()]

            resume_files = st.file_uploader("Resume PDFs", type="pdf", accept_multiple_files=True)

        st.markdown("---")

        st.session_state.skill_type_map = {}
        skill_experience_map = {}

        st.subheader("🔧 Experience Required for Skills")
        st.markdown("Enter the minimum years of experience required for each skill:")

        for skill in essential_input:
            years = st.number_input(f"{skill} (Essential)", min_value=0.0, step=0.5, key=f"ess_{skill}")
            skill_experience_map[skill] = years
            st.session_state.skill_type_map[skill] = "essential"

        for skill in preferred_input:
            years = st.number_input(f"{skill} (Preferred)", min_value=0.0, step=0.5, key=f"pref_{skill}")
            skill_experience_map[skill] = years
            st.session_state.skill_type_map[skill] = "preferred"

        run_eval = st.button("🔁 Run Evaluation")
        reset_inputs = st.button("🧹 Reset All Inputs")

        if reset_inputs:
            st.session_state.clear()
            st.experimental_rerun()

        if jd_text_input and resume_files and run_eval:
            jd_text = jd_text_input
            results = []

            with st.spinner("Processing resumes..."):
                for resume in resume_files:
                    try:
                        text = extract_text_from_pdf(resume)
                        ai_summary = generate_ai_summary(text)
                        skill_map, weighted_score, essential_hit, preferred_hit = evaluate_skills_with_experience(
                            text, skill_experience_map)

                        results.append({
                            "Candidate": resume.name,
                            "Skill Match %": weighted_score,
                            "Match Level": color_match_level(weighted_score),
                            "Essential Skills": f"{essential_hit}/{len([k for k in skill_experience_map if st.session_state.skill_type_map[k] == 'essential'])}",
                            "Preferred Skills": f"{preferred_hit}/{len([k for k in skill_experience_map if st.session_state.skill_type_map[k] == 'preferred'])}",
                            "Skills Table": skill_map,
                            "AI Summary": ai_summary
                        })
                    except Exception as e:
                        st.error(f"Error processing {resume.name}: {e}")

            if results:
                df = pd.DataFrame(results)
                st.subheader("📊 Match Summary Table")
                st.dataframe(df[["Candidate", "Skill Match %", "Essential Skills", "Preferred Skills", "Match Level"]].sort_values(by="Skill Match %", ascending=False))

                csv = df.drop(columns=["Skills Table", "AI Summary"]).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download CSV Report",
                    data=csv,
                    file_name="resume_match_results.csv",
                    mime='text/csv'
                )

                st.subheader("🧠 Matched Skills Breakdown")
                for res in results:
                    st.markdown(f"### {res['Candidate']}")
                    st.markdown("**Skill Match Table:**")
                    skill_df = pd.DataFrame(list(res['Skills Table'].items()), columns=["Skill", "Match"])
                    st.dataframe(skill_df)

                st.subheader("🤖 AI-Generated Summaries")
                for res in results:
                    st.markdown(f"### {res['Candidate']}")
                    st.markdown(res['AI Summary'])

if __name__ == "__main__":
    main()
