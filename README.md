#  AI Resume Analyzer

A clean, modern web app that analyzes a PDF resume with Python and NLP — scoring it, surfacing skills, and matching it against a job description.

## Features
- **Resume score (0–100)** with a transparent breakdown: skills, structure, contact info, content quality
- **ATS compatibility check** — 8-point parsing checklist with a fix hint for each failure
- **Writing-quality review** — weak/passive phrasing, clichés, first-person pronouns, bullet stats, quantified results
- **Skill detection & radar chart** across 6 categories, using whole-word matching and aliases (e.g. `js` → JavaScript)
- **Role-fit recommendations** — best-matching roles (Data Scientist, Full-Stack, DevOps…) with skills to add
- **Experience estimate** — years of experience from date ranges (overlaps merged)
- **Contact extraction** — name (spaCy NER), email, phone, LinkedIn, GitHub
- **Job description matching** — skill coverage, keyword overlap, missing skills/keywords and tailoring advice
- **Exports** — Markdown report and JSON data

## Tech stack
Python · Streamlit · spaCy · pdfplumber

## Project structure
```
app.py            # Streamlit UI
analyzer.py       # Analysis engine (no UI code, easy to test)
assets/style.css  # Theme and components
.streamlit/       # Theme config
```

## Run locally
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes
Works best with text-based PDFs; scanned/image PDFs have no extractable text.
