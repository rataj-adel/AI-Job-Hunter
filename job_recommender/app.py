import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import time
import io

# --- Page Config ---
st.set_page_config(page_title="AI Resume Matcher", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Subtle Animated Background */
    .stApp {
        background: radial-gradient(circle at 15% 50%, #1e1b4b, #0f172a);
        color: #f8fafc;
    }
    
    /* Gradient Text Headers */
    h1, h2, h3 {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Primary Button */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #ec4899 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.25);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(236, 72, 153, 0.4);
    }
    
    /* Custom Job Cards */
    .job-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border 0.3s ease;
    }
    .job-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        border: 1px solid rgba(129, 140, 248, 0.4);
    }
    
    .match-container {
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 10px 20px;
    }
    .match-score {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-right: 5px;
    }
    
    /* Badges */
    .skill-badge {
        display: inline-block;
        padding: 6px 14px;
        margin: 4px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: transform 0.2s;
    }
    .skill-badge:hover {
        transform: scale(1.05);
    }
    .matched-skill {
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .missing-skill {
        background: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    /* Questions Section */
    .question-box {
        background: rgba(56, 189, 248, 0.05);
        border-left: 4px solid #38bdf8;
        padding: 16px 20px;
        margin: 12px 0;
        border-radius: 0 12px 12px 0;
        font-size: 1.05rem;
        line-height: 1.5;
        color: #e2e8f0;
        transition: background 0.3s;
    }
    .question-box:hover {
        background: rgba(56, 189, 248, 0.1);
    }
    
    hr {
        border-color: rgba(255,255,255,0.05);
        margin: 25px 0;
    }
    
    /* File Uploader styling tweak */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255,255,255,0.02);
        border: 2px dashed rgba(255,255,255,0.1);
        border-radius: 12px;
        transition: all 0.3s;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(255,255,255,0.05);
        border-color: #818cf8;
    }
</style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.title("✨ AI Resume Matcher & Interview Prep")
st.markdown("<p style='font-size: 1.2rem; color: #94a3b8; margin-bottom: 40px;'>Upload your resume and explore job matches with AI-generated interview questions.</p>", unsafe_allow_html=True)

# --- Cached Model Loading ---
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sbert():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_flan_t5():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model, DEVICE

# --- Logic Functions ---
def parse_skills(s):
    if isinstance(s, list):
        return [str(x).strip() for x in s if str(x).strip()]
    return [x.strip() for x in str(s).split(';') if x.strip()]

def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(file_bytes) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def build_matcher(nlp, skills_vocab):
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc(skill) for skill in skills_vocab]
    matcher.add('SKILLS', patterns)
    return matcher

def extract_exact_skill_matches(text, nlp, matcher):
    doc = nlp(text)
    matches = matcher(doc)
    return sorted({doc[start:end].text for match_id, start, end in matches})

def extract_candidate_phrases(text, nlp, matcher):
    doc = nlp(text)
    phrases = set(extract_exact_skill_matches(text, nlp, matcher))
    for chunk in doc.noun_chunks:
        cleaned = chunk.text.strip()
        if cleaned and 1 <= len(cleaned.split()) <= 4:
            phrases.add(cleaned)
    for ent in doc.ents:
        if ent.label_ in ('ORG', 'PRODUCT', 'LANGUAGE', 'WORK_OF_ART'):
            phrases.add(ent.text.strip())
    return sorted(p for p in phrases if p)

def normalize_to_canonical_skills(phrases, vocab, vocab_embeddings, model, threshold=0.60):
    canonical = set()
    if not phrases:
        return []
    phrase_embeddings = model.encode(phrases, convert_to_tensor=True)
    cos_scores = util.cos_sim(phrase_embeddings, vocab_embeddings)
    for i, phrase in enumerate(phrases):
        scores = cos_scores[i].cpu().numpy()
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) >= threshold:
            canonical.add(vocab[best_idx])
    return sorted(canonical)

def hybrid_score(candidate_skills, job_skills, model, w_jaccard=0.5, w_sbert=0.5):
    set_a, set_b = set(candidate_skills), set(job_skills)
    
    # Jaccard
    union = set_a | set_b
    jac = len(set_a & set_b) / len(union) if union else 0.0
    
    # SBERT
    if not candidate_skills or not job_skills:
        sbert_sim = 0.0
    else:
        emb_a = model.encode(candidate_skills, convert_to_tensor=True).mean(dim=0)
        emb_b = model.encode(job_skills, convert_to_tensor=True).mean(dim=0)
        sbert_sim = float(util.cos_sim(emb_a, emb_b))
        
    return w_jaccard * jac + w_sbert * sbert_sim

def recommend_top_jobs(candidate_skills, jobs_df, model, top_n=3, w_jaccard=0.5, w_sbert=0.5):
    rows = []
    for _, row in jobs_df.iterrows():
        job_skills = row['skills_list']
        title = row.get('Title', row.get('job_title', 'Unknown Title'))
        score = hybrid_score(candidate_skills, job_skills, model, w_jaccard, w_sbert)
        matched = sorted(set(candidate_skills) & set(job_skills))
        missing = sorted(set(job_skills) - set(candidate_skills))
        rows.append({
            'job_title': title,
            'hybrid_score': round(score, 4),
            'matched_skills': matched,
            'missing_skills': missing
        })
    result_df = pd.DataFrame(rows).sort_values('hybrid_score', ascending=False).reset_index(drop=True)
    return result_df.head(top_n)

def build_prompt(job_title, matched_skills, missing_skills, candidate_skills, num_questions=5, question_type="mixed"):
    matched = matched_skills[:5] if matched_skills else []
    missing = missing_skills[:5] if missing_skills else []
    candidate = candidate_skills[:10] if candidate_skills else []

    if question_type == "technical":
        return f"""You are a Senior Technical Interviewer.
Generate exactly {num_questions} UNIQUE technical interview questions for a {job_title} role.
Candidate Skills: {', '.join(candidate)}
Strong Skills: {', '.join(matched)}
Missing Skills: {', '.join(missing)}
Rules: Ask specific scenario questions. Include one question about missing skills. Output ONLY numbered questions."""
    elif question_type == "behavioral":
        return f"""You are an HR Interviewer.
Generate exactly {num_questions} behavioral interview questions for a {job_title}.
Candidate Skills: {', '.join(candidate)}
Focus on: teamwork, leadership, communication, problem solving, adaptability. Output only numbered questions."""
    else:
        return f"""Generate exactly {num_questions} interview questions (Half technical, Half behavioral) for {job_title}.
Candidate Skills: {', '.join(candidate)}
Matched Skills: {', '.join(matched)}
Missing Skills: {', '.join(missing)}
Output only numbered questions."""

def clean_questions(text):
    questions = []
    for line in text.split("\n"):
        line = line.strip()
        if not line: continue
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r"^Question\s*\d*[:\-]?\s*", "", line, flags=re.IGNORECASE)
        line = line.strip()
        if len(line) < 20: continue
        if not line.endswith("?"): line += "?"
        if line not in questions: questions.append(line)
    return questions

def generate_interview_questions(tokenizer, model, DEVICE, job_title, matched_skills, missing_skills, candidate_skills, num_questions=5, question_type="mixed"):
    prompt = build_prompt(job_title, matched_skills, missing_skills, candidate_skills, num_questions, question_type)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=250, temperature=0.9, do_sample=True, top_p=0.95, repetition_penalty=1.2)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    questions = clean_questions(generated_text)
    
    # Fallback logic if model fails
    if len(questions) < num_questions:
        for skill in matched_skills[:2]:
            questions.append(f"Can you explain a real project where you used {skill} successfully?")
        for skill in missing_skills[:1]:
            questions.append(f"What is your current knowledge of {skill}, and how would you learn it quickly?")
        questions.extend([
            f"Why are you interested in the {job_title} position?",
            "Tell me about your most challenging technical project."
        ])
    
    return list(dict.fromkeys(questions))[:num_questions]

# --- Load Sample Resume ---
@st.cache_data
def load_sample_resume():
    # Sample resume text as bytes
    sample_text = """
    PROFESSIONAL SUMMARY
    Experienced Data Scientist with 5+ years of experience in machine learning, deep learning, and data analysis. 
    Strong background in Python, SQL, and cloud technologies. Passionate about NLP and computer vision applications.

    TECHNICAL SKILLS
    Python, SQL, Machine Learning, Deep Learning, NLP, PyTorch, TensorFlow, Docker, AWS, Data Visualization,
    Tableau, R, Git, CI/CD, Data Analysis, Statistics, Big Data, Spark

    PROFESSIONAL EXPERIENCE

    Senior Data Scientist | TechCorp | 2021-Present
    • Developed and deployed machine learning models for customer churn prediction using PyTorch and AWS
    • Built ETL pipelines for processing large-scale data using Spark and SQL
    • Implemented NLP solutions for sentiment analysis and text classification
    • Collaborated with cross-functional teams to deliver data-driven insights

    Data Scientist | DataViz Inc | 2019-2021
    • Created interactive dashboards using Tableau and Power BI
    • Performed statistical analysis and A/B testing
    • Wrote complex SQL queries for data extraction and analysis
    • Developed predictive models using scikit-learn and XGBoost

    EDUCATION
    M.S. in Computer Science | Stanford University | 2019
    B.S. in Mathematics | MIT | 2017

    CERTIFICATIONS
    AWS Certified Solutions Architect
    Google Professional Data Engineer
    """
    return io.BytesIO(sample_text.encode('utf-8'))

# --- UI Layout ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/942/942748.png", width=60)
    st.header("📂 Data Sources")
    
    # Check if we should use uploaded or pre-loaded data
    use_preloaded = st.checkbox("Use sample data", value=True, help="Use pre-loaded sample data instead of uploading")
    
    if not use_preloaded:
        jobs_file = st.file_uploader("📂 Upload Jobs Dataset (CSV)", type="csv")
        resume_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")
    else:
        jobs_file = None
        resume_file = None
        st.info("Using pre-loaded sample data")
    
    st.markdown("---")
    st.header("⚙️ Settings")
    q_type = st.selectbox("Interview Question Type", ["mixed", "technical", "behavioral"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Analyze & Match", type="primary", use_container_width=True)

# Main Area
if not run_btn:
    # Display initial message with sample data option
    st.markdown("""
        <div style='text-align: center; padding: 100px 20px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/2065/2065157.png' width='120' style='opacity: 0.5; margin-bottom: 20px;'/>
            <h2 style='color: #64748b;'>Ready to analyze your resume</h2>
            <p style='color: #475569; font-size: 1.1rem;'>
                Use the sidebar to choose between sample data or upload your own files, then click "Analyze & Match"
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Load data based on user selection
    with st.status("🔮 Analyzing Your Profile...", expanded=True) as status:
        try:
            st.write("Loading AI Models...")
            nlp = load_spacy()
            sbert_model = load_sbert()
            q_tokenizer, q_model, DEVICE = load_flan_t5()
            
            st.write("Processing Jobs Database...")
            
            # Load jobs data
            if use_preloaded or jobs_file is None:
                # Use sample jobs dataset
                try:
                    # Try to load from a sample jobs dataset
                    # If you have a sample jobs dataset file, you can load it here
                    # For now, let's create a sample jobs dataset
                    sample_jobs = {
                        'Title': [
                            'Data Scientist',
                            'Machine Learning Engineer',
                            'Software Engineer',
                            'Data Analyst',
                            'DevOps Engineer'
                        ],
                        'Keywords': [
                            'Python;SQL;Machine Learning;Deep Learning;NLP;PyTorch;TensorFlow;Docker;AWS',
                            'Python;SQL;Machine Learning;Deep Learning;PyTorch;TensorFlow;Docker;Kubernetes;AWS;GCP',
                            'Python;Java;C++;Git;Docker;Kubernetes;AWS;CI/CD;SQL;Microservices',
                            'Python;SQL;Tableau;Power BI;Data Visualization;Statistics;Excel;R',
                            'Python;AWS;Docker;Kubernetes;CI/CD;Linux;Terraform;Ansible;Jenkins'
                        ]
                    }
                    jobs_df = pd.DataFrame(sample_jobs)
                    resume_bytes = load_sample_resume()
                except Exception as e:
                    st.error(f"Error loading sample jobs data: {str(e)}")
                    st.stop()
            else:
                # Load uploaded jobs dataset
                jobs_df = pd.read_csv(jobs_file)
                # Load uploaded resume
                resume_bytes = resume_file
            
            # Process skills
            skills_col = "Keywords" if "Keywords" in jobs_df.columns else "Skills" if "Skills" in jobs_df.columns else "skills"
            jobs_df['skills_list'] = jobs_df[skills_col].apply(parse_skills)
            if "Title" in jobs_df.columns:
                jobs_df = jobs_df.drop_duplicates(subset=["Title"])
            
            all_skills = sorted(set(skill for skills in jobs_df['skills_list'] for skill in skills))
            skill_vocab_embeddings = sbert_model.encode(all_skills, convert_to_tensor=True)
            matcher = build_matcher(nlp, all_skills)

            st.write("Extracting details from Resume...")
            resume_text = extract_text_from_pdf(resume_bytes)
            candidate_phrases = extract_candidate_phrases(resume_text, nlp, matcher)
            canonical_resume_skills = normalize_to_canonical_skills(candidate_phrases, all_skills, skill_vocab_embeddings, sbert_model, threshold=0.60)
            
            st.write("Finding the perfect matches...")
            top3_df = recommend_top_jobs(canonical_resume_skills, jobs_df, sbert_model, top_n=3)
            
            status.update(label="Analysis Complete! ✨", state="complete", expanded=False)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.header("🏆 Your Top Job Matches")
            
            # Step 3: Display Results
            for idx, row in top3_df.iterrows():
                score_pct = int(row['hybrid_score'] * 100)
                
                matched_html = "".join([f'<span class="skill-badge matched-skill">{s}</span>' for s in row['matched_skills']])
                missing_html = "".join([f'<span class="skill-badge missing-skill">{s}</span>' for s in row['missing_skills']])
                
                if not matched_html: matched_html = "<span style='color:#94a3b8;'>No exact matches found.</span>"
                if not missing_html: missing_html = "<span style='color:#94a3b8;'>You have all required skills! 🎉</span>"

                st.markdown(f"""
                <div class="job-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <h2 style="margin: 0; padding-bottom: 10px;">#{idx+1} {row['job_title']}</h2>
                        <div class="match-container">
                            <span class="match-score">{score_pct}%</span>
                            <span style="color: #6ee7b7; font-weight: 600;">Match</span>
                        </div>
                    </div>
                    <hr>
                    <div style="display: flex; gap: 40px; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 300px;">
                            <h4 style="color: #6ee7b7; margin-bottom: 15px; display: flex; align-items: center;">
                                <span style="font-size: 1.5rem; margin-right: 8px;">✅</span> Your Strengths
                            </h4>
                            <div style="line-height: 2;">{matched_html}</div>
                        </div>
                        <div style="flex: 1; min-width: 300px;">
                            <h4 style="color: #fca5a5; margin-bottom: 15px; display: flex; align-items: center;">
                                <span style="font-size: 1.5rem; margin-right: 8px;">⚠️</span> Skills to Develop
                            </h4>
                            <div style="line-height: 2;">{missing_html}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 🎯 AI Interview Coach")
                with st.spinner(f"Preparing custom questions for {row['job_title']}..."):
                    questions = generate_interview_questions(
                        q_tokenizer, q_model, DEVICE, 
                        row['job_title'], row['matched_skills'], row['missing_skills'], canonical_resume_skills, 
                        num_questions=5, question_type=q_type
                    )
                    
                    for i, q in enumerate(questions, 1):
                        st.markdown(f'<div class="question-box"><strong>Q{i}:</strong> {q}</div>', unsafe_allow_html=True)
                
                st.markdown("<br><br>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.stop()
