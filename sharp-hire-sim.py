import streamlit as st
import pandas as pd
import json
import os
from anthropic import Anthropic
from openai import OpenAI
from pypdf import PdfReader
from docx import Document

# --- CONFIGURATION ---
st.set_page_config(page_title="Sharp Hire v1.9", page_icon="🎯", layout="wide")

# --- SHARP PALETTE CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* INPUTS */
    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1c1c1c !important;
        color: #00e5ff !important;
        border: 1px solid #333 !important;
        font-family: 'Helvetica Neue', sans-serif !important;
    }
    
    /* UPLOADER */
    div[data-testid="stFileUploader"] section {
        background-color: #161b22;
        border: 2px dashed #00e5ff; 
        border-radius: 10px;
        min-height: 120px !important; 
        display: flex; align-items: center; justify-content: center;
    }
    div[data-testid="stFileUploader"] section:hover { border-color: #00ffab; }

    /* HEADERS */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #00e5ff, #d500f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* BUTTONS */
    div[data-testid="stButton"] button {
        background: linear-gradient(45deg, #00e5ff, #00ffab) !important;
        color: #000000 !important;
        border: none !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        transition: transform 0.2s;
    }
    div[data-testid="stButton"] button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px #00ffab;
    }
    
    /* COST METRIC */
    div[data-testid="stMetricValue"] {
        color: #39ff14 !important; /* Neon Green */
        font-family: monospace;
        font-size: 1.8rem !important;
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
# Initialize Costs
if 'costs' not in st.session_state: st.session_state.costs = {"OpenAI (Audio)": 0.0, "Anthropic (Intel)": 0.0}
if 'total_cost' not in st.session_state: st.session_state.total_cost = 0.0

# --- SECRETS ---
try:
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
except:
    st.error("❌ Missing API Keys.")
    st.stop()

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- FUNCTIONS ---

def track_cost(provider, amount):
    st.session_state.costs[provider] += amount
    st.session_state.total_cost += amount

def extract_text_from_file(file):
    try:
        file_type = file.name.split('.')[-1].lower()
        if file_type in ['mp3', 'm4a', 'wav', 'mp4', 'mpeg', 'mpga']:
            return transcribe_audio(file)
        elif file_type == 'pdf':
            reader = PdfReader(file)
            return "\n".join([page.extract_text() for page in reader.pages])
        elif file_type == 'docx':
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_type in ['txt', 'md']:
            return file.read().decode("utf-8")
        return "Unsupported format."
    except Exception as e:
        return f"Error extracting {file.name}: {str(e)}"

def transcribe_audio(file):
    try:
        transcript = openai_client.audio.transcriptions.create(model="whisper-1", file=file)
        track_cost("OpenAI (Audio)", 0.06) # Est cost per file
        return transcript.text
    except Exception as e:
        return f"Whisper Error: {str(e)}"

def clean_json_response(txt):
    txt = txt.strip()
    if "```json" in txt: txt = txt.split("```json")[1].split("```")[0]
    elif "```" in txt: txt = txt.split("```")[1].split("```")[0]
    return txt.strip()

def analyze_forensic(transcript, cv_text, jd_text):
    system_prompt = f"""
    You are a FORENSIC Talent Auditor. Your job is to strictly evaluate a hiring interaction.
    
    **DATA POINTS:**
    1. **JD (The Standard):** What is required.
    2. **CV (The Claim):** What the candidate says they did.
    3. **TRANSCRIPT (The Evidence):** What actually happened.

    **SCORING PROTOCOL (0-10):**
    - **5/10 is AVERAGE.** Do not give 7s or 8s for "okay" answers.
    - **Discrepancy Penalty:** If Transcript contradicts CV, deduct 3 points immediately.

    **OUTPUT JSON:**
    {{
        "candidate": {{
            "name": "Name",
            "scores": {{
                "technical_depth": 0,
                "communication_clarity": 0,
                "cultural_alignment": 0,
                "role_match_index": 0
            }},
            "score_reasoning": {{
                "tech_reason": "Why this score? Cite specific evidence.",
                "comm_reason": "Why this score?",
                "match_reason": "Why this score?"
            }},
            "flags": {{
                "cv_discrepancies": ["List specific contradictions"],
                "red_flags": ["Behavioral or technical concerns"]
            }},
            "summary_verdict": "Hire / No Hire / Strong Hire"
        }},
        "recruiter": {{
            "scores": {{
                "question_difficulty": 0,
                "listening_skills": 0,
                "jd_coverage": 0
            }},
            "missed_opportunities": ["Critical JD topics the recruiter forgot to ask"],
            "coaching_tip": "One high-impact tip to improve."
        }}
    }}
    """

    user_msg = f"JD: {jd_text[:10000]}\nCV: {cv_text[:10000]}\nTRANSCRIPT: {transcript[:50000]}"

    try:
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            temperature=0.1, 
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}]
        )
        track_cost("Anthropic (Intel)", 0.03) # Est cost per analysis
        return json.loads(clean_json_response(message.content[0].text))
    except Exception as e:
        return {"error": str(e)}

def render_neon_progress(label, score, max_score=10):
    pct = (score / max_score) * 100
    color = "#ff4b4b" # Red
    if score >= 5: color = "#ffa700" # Orange
    if score >= 7: color = "#39ff14" # Green
    if score >= 9: color = "#00e5ff" # Cyan (Elite)

    st.markdown(f"""
    <div style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
            <span style="color: #e0e0e0;">{label}</span>
            <span style="color: {color}; font-weight: bold;">{score}/10</span>
        </div>
        <div style="background-color: #222; height: 8px; border-radius: 4px; margin-top: 4px;">
            <div style="background-color: {color}; width: {pct}%; height: 100%; border-radius: 4px; box-shadow: 0 0 6px {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- LAYOUT ---

# HEADER WITH COST CALCULATOR
col_head, col_cost = st.columns([4, 1])
with col_head:
    st.title("🎯 Sharp Hire v1.9")
    st.markdown("Forensic Interview Intelligence")
with col_cost:
    st.metric("Session Cost", f"${st.session_state.total_cost:.4f}")
    with st.expander("Breakdown"):
        st.write(st.session_state.costs)

# INPUTS
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 1. JD")
    jd_file = st.file_uploader("Job Description", type=['pdf','docx','txt','md'], key="jd", label_visibility="collapsed")
with c2:
    st.markdown("### 2. CV")
    cv_file = st.file_uploader("Candidate CV", type=['pdf','docx','txt','md'], key="cv", label_visibility="collapsed")
with c3:
    st.markdown("### 3. Transcript")
    call_file = st.file_uploader("Interview Audio/Text", type=['mp3','wav','m4a','pdf','docx','txt','md'], key="call", label_visibility="collapsed")

st.write("")
start_btn = st.button("Start Forensic Audit", type="primary", use_container_width=True)

if start_btn:
    if not (jd_file and cv_file and call_file):
        st.warning("⚠️ Please upload ALL 3 files.")
    else:
        try:
            # STATUS TILE CONTAINER
            status_container = st.status("🚀 Initializing Audit...", expanded=True)
            
            status_container.write("📂 Extracting text layers...")
            st.session_state.jd_text = extract_text_from_file(jd_file)
            st.session_state.cv_text = extract_text_from_file(cv_file)
            st.session_state.transcript_text = extract_text_from_file(call_file)
            
            status_container.write("🔍 Cross-referencing CV against Audio...")
            status_container.write("⚖️ Auditing Recruiter Performance...")
            
            res = analyze_forensic(st.session_state.transcript_text, st.session_state.cv_text, st.session_state.jd_text)
            st.session_state.analysis_result = res
            
            status_container.update(label="✅ Forensic Audit Complete", state="complete", expanded=False)
            st.rerun() 
        except Exception as e:
            st.error(f"Critical Error: {e}")
            st.stop()

# --- FORENSIC RESULTS DASHBOARD ---
if st.session_state.analysis_result:
    r = st.session_state.analysis_result
    if "error" in r:
        st.error(r['error'])
    else:
        st.divider()
        c_cand, c_rec = st.columns(2)
        
        # CANDIDATE AUDIT
        with c_cand:
            cand = r['candidate']
            st.subheader(f"👤 {cand['name']}")
            st.caption(f"Verdict: **{cand['summary_verdict']}**")
            
            with st.container(border=True):
                s = cand['scores']
                render_neon_progress("Role Match Index", s['role_match_index'])
                st.caption(f"_{cand['score_reasoning']['match_reason']
