# Resume Match and Ranking Script with Authentication

import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import openai
import os
import io
import difflib

# Verify and set OpenAI API Key
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Missing OpenAI API Key. Please set OPENAI_API_KEY in environment variables or .streamlit/secrets.toml")
openai.api_key = api_key

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
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional technical recruiter."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"AI Summary failed: {e}"

def fuzzy_match(term, text, threshold=0.6):
    words = text.split()
    matches = [word for word in words if difflib.SequenceMatcher(None, word.lower(), term.lower()).ratio() >= threshold]
    return bool(matches)

def evaluate_skills(resume_text, essential_skills, preferred_skills):
    all_skills = essential_skills + preferred_skills
    resume_lower = resume_text.lower()
    skill_results = {}
    for skill in all_skills:
        match_found = fuzzy_match(skill, resume_lower)
        skill_results[skill] = "🟢" if match_found else "🔴"
    essential_match = sum(skill_results[k] == "🟢" for k in essential_skills)
    preferred_match = sum(skill_results[k] == "🟢" for k in preferred_skills)
    total_skills = len(essential_skills) + len(preferred_skills)

    # Weighted match: 70% essential, 30% preferred
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

# ---------------------------
# STREAMLIT INTERFACE
# ---------------------------

def main():
    st.set_page_config(page_title="Secure Resume Matcher", layout="centered")
    st.title("🔒 Secure Resume Matcher")

    # Authentication logic
    with st.form("auth_form"):
        username = st.text_input("Enter Username")
        password = st.text_input("Enter Password", type="password")
        submitted = st.form_submit_button("🔐 Login")

    if submitted:
        if username == "Virat" and password == "KleisTech@123":
            st.success("✅ Access granted. Welcome Virat!")

            st.title("📄 Resume Matcher with AI Insights & Skill Scoring")

            with st.sidebar:
                st.info("Paste a Job Description and upload multiple PDF resumes.")
                jd_text_input = st.text_area("Paste Job Description Below", height=200)
                resume_files = st.file_uploader("Resume PDFs", type="pdf", accept_multiple_files=True)

                st.markdown("---")
                essential_input = st.text_area("Essential Skills (comma-separated, editable)", "")
                preferred_input = st.text_area("Preferred Skills (comma-separated, editable)", "")

                run_eval = st.button("🔁 Run Evaluation")
                reset_inputs = st.button("🧹 Reset All Inputs")

            if reset_inputs:
                st.experimental_rerun()

            if jd_text_input and resume_files and run_eval:
                essential_skills = [s.strip() for s in essential_input.split(",") if s.strip()]
                preferred_skills = [s.strip() for s in preferred_input.split(",") if s.strip()]

                jd_text = jd_text_input
                results = []

                with st.spinner("Processing resumes..."):
                    for resume in resume_files:
                        try:
                            text = extract_text_from_pdf(resume)
                            ai_summary = generate_ai_summary(text)
                            skill_map, weighted_score, essential_hit, preferred_hit = evaluate_skills(
                                text, essential_skills, preferred_skills)

                            results.append({
                                "Candidate": resume.name,
                                "Skill Match %": weighted_score,
                                "Match Level": color_match_level(weighted_score),
                                "Essential Skills": f"{essential_hit}/{len(essential_skills)}",
                                "Preferred Skills": f"{preferred_hit}/{len(preferred_skills)}",
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
        else:
            st.error("❌ Access denied. Please check your credentials.")

if __name__ == "__main__":
    main()
