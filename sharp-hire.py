import streamlit as st
import pandas as pd
import json
import os
from anthropic import Anthropic
from openai import OpenAI
from pypdf import PdfReader
from docx import Document

# --- CONFIGURATION ---
st.set_page_config(page_title="Sharp Hire v1.3", page_icon="🎯", layout="wide")

# --- SHARP PALETTE CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1c1c1c !important;
        color: #00e5ff !important;
        border: 1px solid #333 !important;
    }
    /* 3-COLUMN UPLOADER LAYOUT */
    div[data-testid="stFileUploader"] section {
        background-color: #161b22;
        border: 2px dashed #00e5ff; 
        border-radius: 10px;
        min-height: 120px !important; 
        display: flex; align-items: center; justify-content: center;
    }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #00e5ff, #d500f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(45deg, #00e5ff, #00ffab) !important;
        color: #000000 !important;
        border: none !important;
        font-weight: 800 !important;
        text-transform: uppercase;
    }
    .stAlert { background-color: #1c1c1c; border: 1px solid #333; color: #00e5ff; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None
if 'transcript_text' not in st.session_state: st.session_state.transcript_text = ""
if 'cv_text' not in st.session_state: st.session_state.cv_text = ""
if 'jd_text' not in st.session_state: st.session_state.jd_text = ""
if 'processing_status' not in st.session_state: st.session_state.processing_status = "Ready."

# --- SECRETS ---
try:
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
except:
    st.error("❌ Missing API Keys.")
    st.stop()

# --- CLIENTS ---
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- FUNCTIONS ---

def extract_text_from_file(file):
    file_type = file.name.split('.')[-1].lower()
    if file_type in ['mp3', 'm4a', 'wav', 'mp4', 'mpeg', 'mpga']:
        return transcribe_audio(file)
    elif file_type == 'pdf':
        try:
            reader = PdfReader(file)
            return "\n".join([page.extract_text() for page in reader.pages])
        except: return "Error reading PDF"
    elif file_type == 'docx':
        try:
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        except: return "Error reading DOCX"
    elif file_type in ['txt', 'md']:
        return file.read().decode("utf-8")
    return "Unsupported format."

def transcribe_audio(file):
    try:
        transcript = openai_client.audio.transcriptions.create(model="whisper-1", file=file)
        return transcript.text
    except Exception as e:
        return f"Whisper Error: {str(e)}"

def analyze_triangulation(transcript, cv_text, jd_text, mode):
    """
    Triangulates data between the Job Description, CV, and Interview Transcript.
    """
    detail = "Provide detailed evidence." if mode == "Deep Analysis" else "Be concise."

    system_prompt = f"""
    You are an elite Talent Intelligence Analyst. 
    You have three data points: 
    1. A Job Description (JD)
    2. A Candidate CV
    3. An Interview Transcript.

    **YOUR MISSION:**
    Perform a "Triangulation Analysis" to validate truth, fit, and recruiter performance.

    **ANALYSIS LOGIC:**
    1. **JD Validation:** Does the candidate *actually* meet the JD requirements based on their answers? (Score specific fit).
    2. **CV Truth Check:** Did the candidate contradict their CV? Or did they prove their written claims with deep verbal knowledge?
    3. **Recruiter Gap Analysis:** Did the recruiter forget to ask about critical "Must-Haves" listed in the JD?

    **OUTPUT JSON STRUCTURE:**
    {{
        "call_summary": ["Bullet 1", "Bullet 2"],
        "candidate": {{
            "name": "Name",
            "scores": {{
                "role_fit": int, 
                "communication": int, 
                "culture_fit": int, 
                "technical_proficiency": int
            }},
            "cv_reality_check": "Short paragraph analyzing if their interview performance matched their CV claims.",
            "match_to_jd": "Specific assessment of how well they fit the JD requirements.",
            "strengths": ["..."],
            "improvements": ["..."],
            "notable_moments": ["..."]
        }},
        "recruiter": {{
            "scores": {{
                "structure": int,
                "question_depth": int,
                "jd_coverage": int
            }},
            "missed_topics": ["List key JD requirements the recruiter forgot to ask about"],
            "coaching": ["..."]
        }}
    }}
    """

    user_msg = f"""
    JOB DESCRIPTION:
    {jd_text[:10000]}

    CANDIDATE CV:
    {cv_text[:10000]}

    INTERVIEW TRANSCRIPT:
    {transcript[:50000]}
    """

    try:
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=4000,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}]
        )
        response = message.content[0].text
        if "```json" in response: response = response.split("```json")[1].split("```")[0]
        return json.loads(response)
    except Exception as e:
        return {"error": str(e)}

def render_neon_progress(label, score, max_score=10):
    pct = (score / max_score) * 100
    color = "#ff4b4b"
    if score >= 4: color = "#ffa700"
    if score >= 7: color = "#39ff14"
    if score >= 9: color = "#00e5ff"

    st.markdown(f"""
    <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #e0e0e0; font-size: 0.9rem;">{label}</span>
            <span style="color: {color}; font-weight: bold;">{score}/10</span>
        </div>
        <div style="background-color: #333; height: 8px; border-radius: 4px; margin-top: 4px;">
            <div style="background-color: {color}; width: {pct}%; height: 100%; border-radius: 4px; box-shadow: 0 0 5px {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- LAYOUT ---

st.title("🎯 Sharp Hire v1.3")
st.markdown("Context-Aware Interview Intelligence")

# 3 COLUMN INPUTS
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 1. The Job (JD)")
    jd_file = st.file_uploader("Upload Job Description", type=['pdf','docx','txt'], key="jd")

with c2:
    st.markdown("### 2. The Person (CV)")
    cv_file = st.file_uploader("Upload Candidate CV", type=['pdf','docx','txt'], key="cv")

with c3:
    st.markdown("### 3. The Call (Audio/Text)")
    call_file = st.file_uploader("Upload Recording", type=['mp3','wav','m4a','pdf','docx'], key="call")

st.write("")
col_btn, col_stat = st.columns([1, 2])
with col_btn:
    start_btn = st.button("Start Triangulation", type="primary", use_container_width=True)
with col_stat:
    st.info(f"**Status:** {st.session_state.processing_status}")

# PROCESS LOGIC
if start_btn:
    if not (jd_file and cv_file and call_file):
        st.warning("⚠️ Please upload ALL 3 files for triangulation analysis.")
    else:
        st.session_state.processing_status = "Extracting Text..."
        st.rerun()

if start_btn and jd_file and cv_file and call_file:
    with st.spinner("Reading & Transcribing..."):
        st.session_state.jd_text = extract_text_from_file(jd_file)
        st.session_state.cv_text = extract_text_from_file(cv_file)
        st.session_state.transcript_text = extract_text_from_file(call_file)
    
    st.session_state.processing_status = "Triangulating Intelligence..."
    with st.spinner("Comparing JD vs CV vs Interview..."):
        res = analyze_triangulation(st.session_state.transcript_text, st.session_state.cv_text, st.session_state.jd_text, "Deep Analysis")
        st.session_state.analysis_result = res
        st.session_state.processing_status = "Analysis Complete."
        st.rerun()

# RESULTS
if st.session_state.analysis_result:
    r = st.session_state.analysis_result
    if "error" in r:
        st.error(r['error'])
    else:
        st.divider()
        c_cand, c_rec = st.columns(2)
        
        with c_cand:
            st.subheader(f"👤 {r['candidate']['name']}")
            with st.container(border=True):
                s = r['candidate']['scores']
                render_neon_progress("JD Role Fit", s['role_fit'])
                render_neon_progress("Tech Proficiency", s['technical_proficiency'])
                render_neon_progress("Culture", s['culture_fit'])
                
                st.markdown("---")
                st.markdown("#### 🕵️ CV vs Reality Check")
                st.info(r['candidate']['cv_reality_check'])
                
                with st.expander("Fit Analysis"):
                    st.write(r['candidate']['match_to_jd'])

        with c_rec:
            st.subheader("🎧 Recruiter Performance")
            with st.container(border=True):
                rs = r['recruiter']['scores']
                render_neon_progress("JD Coverage", rs['jd_coverage'])
                render_neon_progress("Question Depth", rs['question_depth'])
                
                st.markdown("---")
                st.markdown("#### ⚠️ Missed Topics (from JD)")
                if r['recruiter']['missed_topics']:
                    for m in r['recruiter']['missed_topics']: st.markdown(f"❌ {m}")
                else:
                    st.success("Great coverage! No major topics missed.")

                with st.expander("Coaching Tips"):
                    for c in r['recruiter']['coaching']: st.markdown(f"- {c}")
