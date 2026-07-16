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
st.markdown("<p style='font-size: 1.2rem; color: #94a3b8; margin-bottom: 40px;'>Upload your resume and get matched with top job opportunities!</p>", unsafe_allow_html=True)

# --- Load Sample Jobs Dataset (مدمج في الكود) ---
@st.cache_data
def load_sample_jobs_data():
    """تحميل قاعدة بيانات الوظائف النموذجية"""
    jobs_data = {
        'Title': [
            'Data Scientist',
            'Machine Learning Engineer',
            'Software Engineer',
            'Data Analyst',
            'DevOps Engineer',
            'Cloud Architect',
            'AI Research Scientist',
            'Backend Developer',
            'Frontend Developer',
            'Full Stack Developer',
            'Product Manager',
            'UX/UI Designer',
            'Business Analyst',
            'Data Engineer',
            'Site Reliability Engineer',
            'Security Engineer',
            'Network Engineer',
            'Database Administrator',
            'Mobile Developer (iOS)',
            'Mobile Developer (Android)',
            'Game Developer',
            'Embedded Systems Engineer',
            'Robotics Engineer',
            'NLP Engineer',
            'Computer Vision Engineer',
            'Healthcare Data Scientist',
            'Financial Analyst',
            'Marketing Analyst',
            'Supply Chain Analyst',
            'HR Analytics Specialist'
        ],
        'Keywords': [
            'Python;SQL;Machine Learning;Deep Learning;NLP;PyTorch;TensorFlow;Docker;AWS;Data Visualization;Statistics',
            'Python;SQL;Machine Learning;Deep Learning;PyTorch;TensorFlow;Docker;Kubernetes;AWS;GCP;MLOps;CI/CD',
            'Python;Java;C++;Git;Docker;Kubernetes;AWS;CI/CD;SQL;Microservices;REST APIs;System Design',
            'Python;SQL;Tableau;Power BI;Data Visualization;Statistics;Excel;R;Business Intelligence;Dashboard',
            'Python;AWS;Docker;Kubernetes;CI/CD;Linux;Terraform;Ansible;Jenkins;Monitoring;Cloud;Automation',
            'AWS;Azure;GCP;Cloud Computing;Docker;Kubernetes;Terraform;Python;Security;Networking;Microservices',
            'Python;Machine Learning;Deep Learning;NLP;PyTorch;TensorFlow;Research;Mathematics;Statistics;Data Analysis',
            'Python;Java;SQL;Git;Docker;REST APIs;Microservices;Spring Boot;Hibernate;AWS;PostgreSQL;MongoDB',
            'JavaScript;React;HTML;CSS;TypeScript;Redux;Next.js;Vue.js;Webpack;UI/UX;Git;REST APIs',
            'JavaScript;Python;React;Node.js;MongoDB;SQL;REST APIs;Git;Docker;AWS;TypeScript;HTML;CSS',
            'Product Management;Agile;Scrum;JIRA;Analytics;UX;Market Research;Strategy;Communication;Leadership',
            'Figma;Adobe XD;UI/UX Design;Prototyping;Sketch;Wireframing;User Research;Design Thinking;HTML;CSS',
            'SQL;Excel;Python;Business Intelligence;Power BI;Tableau;Data Analysis;Reporting;Requirements Gathering',
            'Python;SQL;Data Warehousing;ETL;Spark;Hadoop;AWS;Redshift;Airflow;Kafka;Data Modeling;Big Data',
            'Linux;AWS;Kubernetes;Docker;Python;Monitoring;CI/CD;Git;Networking;Automation;Incident Management',
            'Security;Python;Network Security;Penetration Testing;Firewall;Cryptography;AWS Security;ISO 27001',
            'Networking;TCP/IP;Routers;Switches;Firewalls;Linux;Cisco;Cloud Networking;Load Balancers;DNS',
            'SQL;Database Administration;Oracle;MySQL;PostgreSQL;MongoDB;Data Modeling;Backup;Performance Tuning',
            'Swift;iOS Development;Xcode;UIKit;Core Data;REST APIs;Git;App Store;Mobile Development;UI/UX',
            'Kotlin;Android Development;Android Studio;Room;REST APIs;Firebase;Git;Mobile Development;UI/UX',
            'C++;Unity;Game Development;3D Modeling;Animation;Unreal Engine;Game Design;Python;Physics',
            'C;C++;Embedded Systems;Microcontrollers;RTOS;IoT;Python;Hardware;Linux;Firmware',
            'Python;C++;Robotics;ROS;Computer Vision;Machine Learning;Sensor Fusion;Control Systems;MATLAB',
            'Python;NLP;Machine Learning;Deep Learning;PyTorch;TensorFlow;Transformers;Language Models;Data Science',
            'Python;Computer Vision;Deep Learning;PyTorch;TensorFlow;Image Processing;OpenCV;Neural Networks',
            'Python;SQL;Healthcare;Machine Learning;Data Analysis;Statistics;Medical Imaging;Bioinformatics;R',
            'SQL;Excel;Python;Financial Analysis;Statistics;Data Visualization;Power BI;Tableau;Reporting',
            'SQL;Excel;Python;Marketing Analytics;Data Analysis;Google Analytics;Power BI;Tableau;Statistics',
            'SQL;Excel;Python;Data Analysis;Supply Chain;Logistics;Operations;Power BI;Tableau;Statistics',
            'SQL;Python;HR Analytics;Data Analysis;Statistics;Power BI;Tableau;Excel;Reporting;People Analytics'
        ]
    }
    return pd.DataFrame(jobs_data)

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

def extract_text_from_pdf(uploaded_file):
    """استخراج النص من ملف PDF مرفوع"""
    uploaded_file.seek(0)
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
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

# --- UI Layout ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/942/942748.png", width=60)
    st.header("📂 Data Source")
    
    # اختيار مصدر البيانات - Use Sample Jobs Dataset
    use_preloaded = st.checkbox(
        "Use Sample Jobs Dataset", 
        value=True,  # يبدأ البرنامج من غير Sample Jobs
        help="Use pre-loaded sample jobs dataset (30 job positions)"
    )
    
    if use_preloaded:
        jobs_file = None
        st.success("✅ Using sample jobs dataset (30 positions)")
    else:
        jobs_file = st.file_uploader("📂 Upload Jobs Dataset (CSV)", type="csv")
    
    st.markdown("---")
    st.header("📄 Resume Upload")
    
    # رفع الـ Resume في كل الحالات (خارج الـ if)
    resume_file = st.file_uploader(
        "📄 Upload Your Resume (PDF)", 
        type="pdf",
        help="Upload your resume in PDF format"
    )
    
    st.markdown("---")
    st.header("⚙️ Settings")
    q_type = st.selectbox("Interview Question Type", ["mixed", "technical", "behavioral"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Analyze & Match", type="primary", use_container_width=True)

# Main Area
if not analyze_btn:
    # عرض رسالة ترحيب
    st.markdown("""
        <div style='text-align: center; padding: 80px 20px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/2065/2065157.png' width='120' style='opacity: 0.5; margin-bottom: 20px;'/>
            <h2 style='color: #64748b;'>Ready to find your dream job</h2>
            <p style='color: #475569; font-size: 1.1rem;'>
                Upload your resume (PDF) and choose data source in the sidebar, then click "Analyze & Match"
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    # --- تحميل البيانات حسب اختيار المستخدم ---
    with st.status("🔮 Analyzing Your Profile...", expanded=True) as status:
        try:
            st.write("🔄 Loading AI Models...")
            nlp = load_spacy()
            sbert_model = load_sbert()
            q_tokenizer, q_model, DEVICE = load_flan_t5()
            
            # --- تحميل Jobs Dataset ---
            if use_preloaded:
                st.write("📊 Loading sample jobs dataset...")
                jobs_df = load_sample_jobs_data()
            else:
                if jobs_file is None:
                    st.error("❌ Please upload a jobs dataset!")
                    st.stop()
                st.write("📊 Loading uploaded jobs dataset...")
                jobs_df = pd.read_csv(jobs_file)
            
            # --- التحقق من رفع الـ Resume ---
            if resume_file is None:
                st.error("❌ Please upload your resume!")
                st.stop()
            
            # --- معالجة البيانات ---
            # Process skills
            skills_col = "Keywords" if "Keywords" in jobs_df.columns else "Skills" if "Skills" in jobs_df.columns else "skills"
            jobs_df['skills_list'] = jobs_df[skills_col].apply(parse_skills)
            if "Title" in jobs_df.columns:
                jobs_df = jobs_df.drop_duplicates(subset=["Title"])
            
            all_skills = sorted(set(skill for skills in jobs_df['skills_list'] for skill in skills))
            skill_vocab_embeddings = sbert_model.encode(all_skills, convert_to_tensor=True)
            matcher = build_matcher(nlp, all_skills)

            st.write("📄 Extracting details from Resume...")
            # قراءة محتوى الـ Resume مباشرة من الملف المرفوع
            resume_text = extract_text_from_pdf(resume_file)
            
            candidate_phrases = extract_candidate_phrases(resume_text, nlp, matcher)
            canonical_resume_skills = normalize_to_canonical_skills(candidate_phrases, all_skills, skill_vocab_embeddings, sbert_model, threshold=0.60)
            
            st.write("🎯 Finding the perfect matches...")
            top3_df = recommend_top_jobs(canonical_resume_skills, jobs_df, sbert_model, top_n=3)
            
            status.update(label="Analysis Complete! ✨", state="complete", expanded=False)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- عرض النتائج ---
            if use_preloaded:
                st.success(f"✅ Using sample jobs dataset: {len(jobs_df)} job positions analyzed")
            else:
                st.success(f"✅ Using uploaded jobs dataset: {len(jobs_df)} job positions analyzed")
            
            st.header("🏆 Your Top Job Matches")
            
            # Display Results
            for idx, row in top3_df.iterrows():
                score_pct = int(row['hybrid_score'] * 100)
                
                matched_html = "".join([f'<span class="skill-badge matched-skill">{s}</span>' for s in row['matched_skills'][:10]])
                missing_html = "".join([f'<span class="skill-badge missing-skill">{s}</span>' for s in row['missing_skills'][:10]])
                
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
            st.error(f"❌ An error occurred during analysis: {str(e)}")
            st.stop()
