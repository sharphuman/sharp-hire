import streamlit as st
import pandas as pd
import json
import os
import io
from anthropic import Anthropic
from fpdf import FPDF # Requires pip install fpdf
from docx import Document # Requires pip install python-docx

# --- CONFIGURATION ---
st.set_page_config(page_title="Sharp Hire SIM v2.6", page_icon="🎲", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1c1c1c !important;
        color: #00e5ff !important;
        border: 1px solid #333 !important;
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
    div[data-testid="stMetricValue"] { color: #39ff14 !important; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# --- STATE ---
if 'sim_data' not in st.session_state: st.session_state.sim_data = None
if 'session_cost' not in st.session_state: st.session_state.session_cost = 0.0000

# --- SECRETS ---
try:
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
except:
    st.error("❌ Missing Anthropic API Key.")
    st.stop()

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# --- HELPER FUNCTIONS ---

def track_cost(amount):
    st.session_state.session_cost += amount

def clean_json(text):
    text = text.strip()
    if "```json" in text: text = text.split("```json")[1].split("```")[0]
    elif "```" in text: text = text.split("```")[1].split("```")[0]
    return json.loads(text)

# --- FILE GENERATORS ---

def create_docx(text):
    """Converts text string to DOCX bytes."""
    doc = Document()
    for line in text.split('\n'):
        doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

def create_pdf(text):
    """Converts text string to PDF bytes (Latin-1 safe)."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    # Sanitize text for FPDF (simple ascii/latin replacement)
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, safe_text)
    return pdf.output(dest='S').encode('latin-1')

# --- GENERATION LOGIC ---

def generate_full_scenario(job_title, industry, level, requirements):
    prompt = f"""
    Generate a detailed recruitment scenario.
    ROLE: {job_title} | INDUSTRY: {industry} | LEVEL: {level} | REQS: {requirements}

    TASK:
    1. Write a professional JD.
    2. Create 2 Candidates:
       - Candidate A (Strong Fit)
       - Candidate B (Risky/Bad Fit)
    
    FOR EACH CANDIDATE:
    - Write a FULL CV text.
    - Write a FULL VERBATIM TRANSCRIPT text (Interview dialogue).

    OUTPUT JSON:
    {{
        "job_description": "Full JD...",
        "candidates": [
            {{ "id": "A", "name": "Name", "vibe": "Strong", "cv_text": "...", "transcript": "..." }},
            {{ "id": "B", "name": "Name", "vibe": "Weak", "cv_text": "...", "transcript": "..." }}
        ]
    }}
    """
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=8000, temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        track_cost(0.06)
        return clean_json(msg.content[0].text)
    except Exception as e:
        return {"error": str(e)}

def analyze_candidates(jd, cands):
    prompt = f"""
    Analyze these 2 candidates against the JD.
    JD: {jd[:1500]}
    CAND A: {cands[0]['transcript'][:3000]}
    CAND B: {cands[1]['transcript'][:3000]}
    
    OUTPUT JSON:
    {{
        "analyses": [
            {{ "id": "A", "role_fit_score": 9, "comm_score": 9, "tech_score": 9, "culture_score": 9, "verdict": "Hire", "reasoning": "..." }},
            {{ "id": "B", "role_fit_score": 4, "comm_score": 5, "tech_score": 5, "culture_score": 4, "verdict": "No Hire", "reasoning": "..." }}
        ]
    }}
    """
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=4000, temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        track_cost(0.03)
        return clean_json(msg.content[0].text)
    except Exception as e:
        return {"error": str(e)}

# --- UI ---

col_title, col_cost = st.columns([4, 1])
with col_title:
    st.title("🎲 Sharp Hire: Asset Factory")
    st.markdown("Generate full-length interview assets (PDF/DOCX) for testing.")
with col_cost:
    st.metric("Session Cost", f"${st.session_state.session_cost:.4f}")

c1, c2 = st.columns(2)
with c1: 
    job_title = st.text_input("Role", "Senior Infrastructure Engineer")
    industry = st.text_input("Industry", "Enterprise IT")
with c2: 
    level = st.selectbox("Seniority", ["Senior/Lead", "Executive", "Mid-Level"])
    requirements = st.text_input("Requirements", "Quest Migration, AD, VMware, 20 Years Exp")

if st.button("🎲 Run Simulation", type="primary", use_container_width=True):
    with st.status("⚙️ Generating Simulation Data...", expanded=True) as status:
        st.write("📝 Creating Content...")
        scenario = generate_full_scenario(job_title, industry, level, requirements)
        if "error" in scenario: st.error(scenario['error']); st.stop()
            
        st.write("🧠 Analyzing...")
        analysis = analyze_candidates(scenario['job_description'], scenario['candidates'])
        if "error" in analysis: st.error(analysis['error']); st.stop()
        
        # Merge
        final_data = []
        for i, cand in enumerate(scenario['candidates']):
            an = next((x for x in analysis['analyses'] if x["id"] == cand["id"]), None)
            if an: final_data.append({**cand, **an})
            
        st.session_state.sim_data = {"jd": scenario['job_description'], "results": final_data}
        status.update(label="Assets Ready!", state="complete", expanded=False)
        st.rerun()

# --- DOWNLOADS & RESULTS ---
if st.session_state.sim_data:
    res = st.session_state.sim_data['results']
    jd_txt = st.session_state.sim_data['jd']
    
    st.divider()
    st.subheader("📂 Download Assets (PDF / Word)")
    
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**1. The Job Description**")
        st.download_button("📄 Download PDF", create_pdf(jd_txt), "JD.pdf", "application/pdf")
        st.download_button("📝 Download Word", create_docx(jd_txt), "JD.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    
    with d2:
        name_a = res[0]['name']
        st.markdown(f"**2. {name_a} (Strong)**")
        # CV A
        st.download_button(f"📄 CV PDF", create_pdf(res[0]['cv_text']), f"{name_a}_CV.pdf", "application/pdf")
        st.download_button(f"📝 CV Word", create_docx(res[0]['cv_text']), f"{name_a}_CV.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        # Transcript A
        st.download_button(f"💬 Transcript PDF", create_pdf(res[0]['transcript']), f"{name_a}_Transcript.pdf", "application/pdf")
        st.download_button(f"💬 Transcript Word", create_docx(res[0]['transcript']), f"{name_a}_Transcript.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    with d3:
        name_b = res[1]['name']
        st.markdown(f"**3. {name_b} (Weak)**")
        # CV B
        st.download_button(f"📄 CV PDF", create_pdf(res[1]['cv_text']), f"{name_b}_CV.pdf", "application/pdf")
        st.download_button(f"📝 CV Word", create_docx(res[1]['cv_text']), f"{name_b}_CV.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        # Transcript B
        st.download_button(f"💬 Transcript PDF", create_pdf(res[1]['transcript']), f"{name_b}_Transcript.pdf", "application/pdf")
        st.download_button(f"💬 Transcript Word", create_docx(res[1]['transcript']), f"{name_b}_Transcript.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    st.divider()
    
    # Leaderboard
    df_data = [{"Name": r['name'], "Role Fit": r['role_fit_score'], "Tech": r['tech_score'], "Verdict": r['verdict']} for r in res]
    st.dataframe(pd.DataFrame(df_data), use_container_width=True, hide_index=True)
