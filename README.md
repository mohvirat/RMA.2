# Resume Matcher with AI Evaluation

This is a deployable Streamlit web application to match resumes to job descriptions using a skill-based evaluation system and GPT-4 powered AI summaries.

## 🚀 Features
- Upload Job Description (TXT)
- Upload multiple Resume PDFs
- Skill-based scoring with essential/preferred categories
- AI-generated summaries using OpenAI
- Downloadable CSV match reports
- Color-coded candidate fit level

## 🛠️ Setup

```bash
pip install -r requirements.txt
```

## 🔐 Add Your API Key
Create a `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY = "sk-..."
```

## ▶️ Run the App
```bash
streamlit run app.py
```

## 🌐 Deploy on Streamlit Cloud
1. Push this repo to GitHub.
2. Go to https://streamlit.io/cloud
3. Deploy using your `app.py` and set secrets in the dashboard.
