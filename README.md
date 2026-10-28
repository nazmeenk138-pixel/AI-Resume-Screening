
!pip install -q pypdf scikit-learn gradio
# TASK 2: AI RESUME SCREENING SYSTEM
# Python + Gradio | Google Colab

%pip -q install gradio pypdf scikit-learn pandas

import os
import re
import tempfile
import pandas as pd
import gradio as gr

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Extract text from PDF resumes
def extract_resume_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]
        return "\n".join(pages).strip()
    except Exception:
        return ""


# 2. Screen and rank uploaded resumes
def screen_resumes(files, job_description):
    if not files:
        return pd.DataFrame(), None

    if not job_description or not job_description.strip():
        return pd.DataFrame(), None

    if not isinstance(files, list):
        files = [files]

    resumes = []

    for file in files:
        path = file if isinstance(file, str) else file.name

        if not path.lower().endswith(".pdf"):
            continue

        text = extract_resume_text(path)

        if text:
            resumes.append({
                "Candidate": os.path.basename(path),
                "Resume Text": text
            })

    if not resumes:
        return pd.DataFrame({
            "Message": [
                "No readable PDF text found. Please upload text-based PDF resumes."
            ]
        }), None

    # 3. Calculate similarity with the job description
    documents = [job_description] + [
        item["Resume Text"] for item in resumes
    ]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
        matrix = vectorizer.fit_transform(documents)

        scores = cosine_similarity(
            matrix[0:1], matrix[1:]
        ).flatten()

    except ValueError:
        return pd.DataFrame({
            "Message": ["Not enough readable text to calculate scores."]
        }), None

    # 4. Extract matching keywords and rank candidates
    job_words = set(
        re.findall(r"[a-zA-Z][a-zA-Z+#.]{1,}", job_description.lower())
    )

    results = []

    for item, score in zip(resumes, scores):
        resume_text = item["Resume Text"]
        resume_words = set(
            re.findall(r"[a-zA-Z][a-zA-Z+#.]{1,}", resume_text.lower())
        )

        matched = sorted(job_words.intersection(resume_words))

        results.append({
            "Candidate": item["Candidate"],
            "Match Score (%)": round(float(score) * 100, 2),
            "Matched Keywords": ", ".join(matched[:15]),
            "Review Status": (
                "High Match" if score >= 0.50
                else "Moderate Match" if score >= 0.25
                else "Needs Review"
            )
        })

    df = pd.DataFrame(results)
    df = df.sort_values(
        "Match Score (%)", ascending=False
    ).reset_index(drop=True)

    df.insert(0, "Rank", range(1, len(df) + 1))

    # 5. Create the CSV report before offering the download
    report_path = os.path.join(
        tempfile.gettempdir(),
        "candidate_ranking_report.csv"
    )

    df.to_csv(report_path, index=False)

    return df, report_path


# 6. Create the user interface
with gr.Blocks(title="AI Resume Screening System") as demo:
    gr.Markdown("# AI Resume Screening System")
    gr.Markdown(
        "Upload PDF resumes, enter a job description, "
        "and generate a candidate ranking report."
    )

    resumes_input = gr.File(
        label="Upload Candidate Resumes (PDF)",
        file_count="multiple",
        file_types=[".pdf"],
        type="filepath"
    )

    job_input = gr.Textbox(
        label="Job Description",
        lines=6,
        placeholder=(
            "Example: We need an HR Assistant with communication, "
            "recruitment, MS Excel, teamwork and organizational skills."
        ),
        value=(
            "We are looking for an HR Assistant with communication, "
            "recruitment, human resources, MS Excel, teamwork, "
            "organization and administration skills."
        )
    )

    screen_button = gr.Button("Screen Resumes", variant="primary")

    results_output = gr.Dataframe(
        label="Candidate Ranking Results"
    )

    csv_output = gr.File(label="Download Candidate Ranking CSV")

    screen_button.click(
        fn=screen_resumes,
        inputs=[resumes_input, job_input],
        outputs=[results_output, csv_output]
    )

demo.launch(share=True)
