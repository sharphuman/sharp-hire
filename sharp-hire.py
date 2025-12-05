import streamlit as st
import pandas as pd
import json
import os
from anthropic import Anthropic
from openai import OpenAI
from pypdf import PdfReader
from docx import Document

# --- CONFIGURATION ---
st.set_page_config(page_title="Sharp Hire v1.2", page_icon="🎯", layout="wide")

# --- SHARP PALETTE CSS (NEON/BLACK THEME) ---
st.markdown("""
<style>
    /* MAIN BACKGROUND */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* INPUTS */
    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1c1c1c !important;
        color: #00e5ff !important; /* Neon Cyan */
        border: 1px solid #333 !important;
        font-family: 'Helvetica Neue', sans-serif !important;
    }
    
    /* SQUARE FILE UPLOADER */
    div[data-testid="stFileUploader"] section {
        background-color: #161b22;
        border: 2px dashed #00e5ff; 
        border-radius: 15px;
        min-height: 200px !important; 
        display: flex; align-items: center; justify-content: center;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #00ffab; /* Green on hover */
    }

    /* HEADERS */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #00e5ff, #d500f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* ACTION BUTTON (Cyan to Green Gradient) */
    div[data-testid="stButton"] button {
        background: linear-gradient(45deg, #00e5ff, #00ffab) !important;
        color: #000000 !important;
        border: none !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    div[data-testid="stButton"] button:hover {
        box-shadow: 0 0 20px #00ffab;
        transform: scale(1.02);
    }

    /* METRIC CARDS */
    div[data-testid="stMetricValue"] {
        color: #00ffab !important; /* Neon Green Score */
    }
    
    .score-card {
        background-color: #1c1c1c;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #d500f9;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None
if 'transcript_text' not in st.session_state: st.session_state.transcript_text = ""
if 'processing_status' not in st.session_state: st.session_state.processing_status = "Ready."

# --- SECRETS LOADING ---
try:
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
except:
    st.error("❌ Missing API Keys. Please set ANTHROPIC_API_KEY and OPENAI_API_KEY.")
    st.stop()

# --- CLIENTS ---
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- CORE FUNCTIONS ---

def extract_text_from_file(file):
    """Router to extract text based on file type (Audio vs Doc)."""
    file_type = file.name.split('.')[-1].lower()
    
    # 1. AUDIO/VIDEO -> WHISPER
    if file_type in ['mp3', 'm4a', 'wav', 'mp4', 'mpeg', 'mpga']:
        return transcribe_audio(file)
    
    # 2. PDF
    elif file_type == 'pdf':
        try:
            reader = PdfReader(file)
            return "\n".join([page.extract_text() for page in reader.pages])
        except Exception as e:
            return f"Error reading PDF: {e}"
            
    # 3. DOCX
    elif file_type == 'docx':
        try:
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            return f"Error reading DOCX: {e}"
            
    # 4. TEXT
    elif file_type in ['txt', 'md']:
        return file.read().decode("utf-8")
        
    return "Unsupported file format."

def transcribe_audio(file):
    """Transcribes audio using OpenAI Whisper."""
    try:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1", 
            file=file
        )
        return transcript.text
    except Exception as e:
        if "413" in str(e): return "Error: Audio file exceeds 25MB limit. Please compress or split."
        return f"Whisper Error: {str(e)}"

def analyze_call(transcript, mode):
    """Main Analysis Engine using Claude 3.5 Sonnet."""
    
    detail_instruction = "Provide detailed feedback." if mode == "Deep Analysis" else "Provide high-level bullet points."

    system_prompt = f"""
    You are an expert Talent Acquisition Coach and Technical Hiring Manager.
    Your task is to analyze a transcript of an interview call between a **Recruiter** and a **Candidate**.
    
    **CONTEXT:**
    If the transcript does not explicitly label speakers, you must INFER who is who based on context.

    **TASK:**
    Analyze the call and output a valid JSON object.
    {detail_instruction}

    **SCORING CRITERIA (1-10 Scale):**
    - 1-3: Poor / Red Flag
    - 4-6: Average / OK
    - 7-8: Good / Strong
    - 9-10: Exceptional / Elite

    **OUTPUT JSON STRUCTURE:**
    {{
        "call_summary": ["Bullet 1", "Bullet 2", "Bullet 3"],
        "candidate": {{
            "name": "Inferred Name or 'Candidate'",
            "scores": {{
                "role_fit": int,
                "communication": int,
                "culture_fit": int,
                "motivation": int,
                "technical_proficiency": int
            }},
            "technical_explanation": "Short text explaining the technical score.",
            "strengths": ["..."],
            "improvements": ["..."],
            "notable_moments": ["Quote or moment..."]
        }},
        "recruiter": {{
            "scores": {{
                "call_structure": int,
                "question_quality": int,
                "candidate_experience": int
            }},
            "coaching_feedback": ["..."],
            "strengths": ["..."]
        }}
    }}
    """

    # Safety truncate
    user_message = f"Here is the transcript:\n\n{transcript[:60000]}"

    try:
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=4000,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        
        response_text = message.content[0].text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
            
        return json.loads(response_text)

    except Exception as e:
        return {"error": str(e)}

def render_neon_progress(label, score, max_score=10):
    """Custom HTML/CSS Progress Bar for the Sharp Look"""
    percentage = (score / max_score) * 100
    color = "#ff4b4b" # Red
    if score >= 4: color = "#ffa700" # Orange/Yellow
    if score >= 7: color = "#39ff14" # Neon Green
    if score >= 9: color = "#00e5ff" # Cyan (Elite)

    st.markdown(f"""
    <div style="margin-bottom: 15px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="color: #e0e0e0; font-weight: 600;">{label}</span>
            <span style="color: {color}; font-weight: bold;">{score}/10</span>
        </div>
        <div style="background-color: #333; border-radius: 5px; height: 10px; width: 100%;">
            <div style="background-color: {color}; width: {percentage}%; height: 100%; border-radius: 5px; box-shadow: 0 0 8px {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- UI LAYOUT ---

st.title("🎯 Sharp Hire v1.2")
st.markdown("Automated Interview Intelligence & Coaching")

# --- 2 COLUMN LAYOUT (Input | Status) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📎 Upload Interview")
    uploaded_file = st.file_uploader(
        "Drag and drop call recording or transcript...", 
        type=['mp3', 'm4a', 'wav', 'mp4', 'mpeg', 'pdf', 'docx', 'txt'], 
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### ⚙️ Config")
    analysis_mode = st.radio("Depth", ["Light (Fast)", "Deep Analysis"], index=1)
    st.info(st.session_state.processing_status)

# START BUTTON
st.write("")
if st.button("Start Sharp Analysis", type="primary", use_container_width=True):
    
    if uploaded_file:
        st.session_state.processing_status = "Processing File..."
        st.session_state.analysis_result = None
        st.rerun()
    
    if uploaded_file:
        # 1. EXTRACT / TRANSCRIBE
        with st.spinner("🎧 Reading / Transcribing..."):
            extracted_text = extract_text_from_file(uploaded_file)
            st.session_state.transcript_text = extracted_text
        
        if "Error" in extracted_text or "Unsupported" in extracted_text:
            st.error(extracted_text)
            st.session_state.processing_status = "Error."
        else:
            # 2. ANALYZE
            st.session_state.processing_status = "Analyzing Intelligence..."
            with st.spinner("🧠 Analyzing behaviors & skills..."):
                result = analyze_call(extracted_text, analysis_mode)
                st.session_state.analysis_result = result
                st.session_state.processing_status = "Analysis Complete."
                st.rerun()
    else:
        st.warning("Please upload a file first.")

# --- RESULTS DASHBOARD ---
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    
    if "error" in res:
        st.error(f"Analysis Failed: {res['error']}")
    else:
        st.divider()
        col_cand, col_rec = st.columns(2)
        
        # --- CANDIDATE CARD ---
        with col_cand:
            st.markdown(f"### 👤 Candidate: {res['candidate']['name']}")
            with st.container(border=True):
                c_scores = res['candidate']['scores']
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("Tech Score", f"{c_scores['technical_proficiency']}/10")
                with c2:
                    st.caption(f"**Insight:** {res['candidate']['technical_explanation']}")
                
                st.divider()
                render_neon_progress("Role Fit", c_scores['role_fit'])
                render_neon_progress("Communication", c_scores['communication'])
                render_neon_progress("Culture Fit", c_scores['culture_fit'])
                render_neon_progress("Motivation", c_scores['motivation'])
            
            with st.expander("📝 Candidate Feedback & Strengths", expanded=True):
                st.markdown("**✅ Strengths**")
                for s in res['candidate']['strengths']: st.markdown(f"- {s}")
                st.markdown("**⚠️ Areas to Improve**")
                for i in res['candidate']['improvements']: st.markdown(f"- {i}")

        # --- RECRUITER CARD ---
        with col_rec:
            st.markdown("### 🎧 Recruiter Performance")
            with st.container(border=True):
                r_scores = res['recruiter']['scores']
                avg_rec = round(sum(r_scores.values()) / 3, 1)
                st.metric("Recruiter Effectiveness", f"{avg_rec}/10")
                
                st.divider()
                render_neon_progress("Call Structure", r_scores['call_structure'])
                render_neon_progress("Question Quality", r_scores['question_quality'])
                render_neon_progress("Candidate Experience", r_scores['candidate_experience'])
            
            with st.expander("🎓 Coaching Tips", expanded=True):
                st.markdown("**💡 Actionable Feedback**")
                for c in res['recruiter']['coaching_feedback']: st.markdown(f"- {c}")

        st.divider()
        c_sum, c_quotes = st.columns(2)
        
        with c_sum:
            st.markdown("### 📄 Call Summary")
            for item in res['call_summary']:
                st.markdown(f"• {item}")
                
        with c_quotes:
            st.markdown("### 💬 Notable Moments")
            for moment in res['candidate']['notable_moments']:
                st.info(f'"{moment}"')

        with st.expander("View Raw Transcript"):
            st.text(st.session_state.transcript_text)
