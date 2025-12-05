import streamlit as st
import pandas as pd
import json
import os
from anthropic import Anthropic

# --- CONFIGURATION ---
st.set_page_config(page_title="Sharp Hire SIM v2.3", page_icon="🎲", layout="wide")

# --- SHARP PALETTE CSS ---
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
    /* NEON GRADIENT BUTTONS */
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
    .stAlert { background-color: #1c1c1c; border: 1px solid #333; color: #00e5ff; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'sim_data' not in st.session_state: st.session_state.sim_data = None
if 'processing_status' not in st.session_state: st.session_state.processing_status = "Ready to Simulate."

# --- SECRETS ---
try:
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
except:
    st.error("❌ Missing Anthropic API Key.")
    st.stop()

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# --- FUNCTIONS ---

def clean_json(text):
    text = text.strip()
    if "```json" in text: text = text.split("```json")[1].split("```")[0]
    elif "```" in text: text = text.split("```")[1].split("```")[0]
    return json.loads(text)

def generate_full_scenario(job_title, industry, level, requirements):
    """
    Generates 1 JD and 2 DEEP Candidates (Quality over Quantity).
    """
    prompt = f"""
    You are a Hiring Simulation Engine. Generate a detailed, life-like recruitment scenario for a simulation.
    
    **PARAMETERS:**
    - Role: {job_title}
    - Industry: {industry}
    - Level: {level}
    - Key Reqs: {requirements}

    **TASK:**
    1. **Job Description (JD):** Write a full, professional JD with Responsibilities and Requirements.
    2. **Create 2 Distinct Candidates:**
       - **Candidate A (The Strong Fit):** Competent, good communicator, clear experience.
       - **Candidate B (The Risk/Bad Fit):** Nervous, evasive, or lacks specific key skills despite a good CV.
    
    **CRITICAL INSTRUCTION: DETAIL LEVEL**
    - **CV:** Do NOT summarize. Write the **FULL TEXT** of a resume. List companies, dates (e.g. 2018-2023), bullet points of projects, and skills. It must look like a text-dump of a PDF.
    - **TRANSCRIPT:** Do NOT summarize. Write a **VERBATIM SCRIPT** of the interview. 
        - Include "Umm", "Uh", pauses, and interruptions to make it realistic.
        - The Recruiter should ask deep technical questions based on the requirements.
        - The Candidate should give long, multi-sentence answers (or struggle significantly).
        - Length: At least 20-30 exchanges per candidate.

    **OUTPUT JSON STRUCTURE:**
    {{
        "job_description": "Full JD Text...",
        "candidates": [
            {{
                "id": "A",
                "name": "Name",
                "vibe": "Strong Match",
                "cv_text": "EXPERIENCE\\n\\nSenior Engineer | Tech Corp | 2019-Present\\n- Managed Active Directory...",
                "transcript": "Recruiter: Hi, thanks for joining.\\nCandidate: Yeah, great to be here.\\nRecruiter: Tell me about your experience with Quest Migration..."
            }},
            {{
                "id": "B",
                "name": "Name",
                "vibe": "Risky / Weak",
                "cv_text": "...",
                "transcript": "..."
            }}
        ]
    }}
    """
    
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", 
            max_tokens=8000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return clean_json(msg.content[0].text)
    except Exception as e:
        return {"error": str(e)}

def analyze_candidates(scenario_data):
    """
    Analyzes the candidates using the "Sharp Hire" logic.
    """
    jd = scenario_data['job_description']
    cands = scenario_data['candidates']
    
    prompt = f"""
    Analyze these 2 candidates against the Job Description.
    
    JOB DESCRIPTION: {jd[:2000]}
    
    CANDIDATE DATA: {json.dumps(cands)}
    
    **TASK:**
    Return a JSON with detailed scoring.
    
    **OUTPUT JSON:**
    {{
        "analyses": [
            {{
                "id": "A",
                "role_fit_score": 9,
                "comm_score": 9,
                "tech_score": 9,
                "culture_score": 9,
                "verdict": "Strong Hire",
                "reasoning": "..."
            }},
            {{
                "id": "B",
                "role_fit_score": 4,
                "comm_score": 5,
                "tech_score": 5,
                "culture_score": 4,
                "verdict": "No Hire",
                "reasoning": "..."
            }}
        ]
    }}
    """
    
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", 
            max_tokens=4000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        return clean_json(msg.content[0].text)
    except Exception as e:
        return {"error": str(e)}

# --- UI LAYOUT ---

st.title("🎲 Sharp Hire: Asset Factory")
st.markdown("Generate full-length interview assets for testing.")

# INPUTS
c1, c2 = st.columns(2)
with c1: 
    job_title = st.text_input("Role Title", "Senior Infrastructure Engineer")
    industry = st.text_input("Industry", "Enterprise IT")
with c2: 
    level = st.selectbox("Seniority", ["Senior/Lead", "Executive", "Mid-Level"])
    requirements = st.text_input("Key Requirements", "Quest Migration, Active Directory, VMware, 20 Years Exp")

if st.button("🎲 Run Simulation", type="primary", use_container_width=True):
    with st.status("⚙️ Fabricating Reality...", expanded=True) as status:
        st.write("📝 Drafting detailed Job Description...")
        st.write("👤 Inventing Candidates (Generating full CVs & Transcripts)...")
        
        # 1. Generate Content
        scenario = generate_full_scenario(job_title, industry, level, requirements)
        
        if "error" in scenario:
            st.error(f"Generation Failed: {scenario['error']}")
            st.stop()
            
        st.write("🧠 Analyzing Performance...")
        
        # 2. Analyze Content
        analysis = analyze_candidates(scenario)
        
        if "error" in analysis:
            st.error(f"Analysis Failed: {analysis['error']}")
            st.stop()
        
        # 3. Merge Data
        final_data = []
        for i, cand in enumerate(scenario['candidates']):
            matching_analysis = next((item for item in analysis['analyses'] if item["id"] == cand["id"]), None)
            
            if matching_analysis:
                merged = {**cand, **matching_analysis}
                final_data.append(merged)
            
        st.session_state.sim_data = {"jd": scenario['job_description'], "results": final_data}
        status.update(label="Assets Generated!", state="complete", expanded=False)

# RESULTS
if st.session_state.sim_data:
    results = st.session_state.sim_data['results']
    jd_text = st.session_state.sim_data['jd']
    
    st.divider()
    
    # 1. DOWNLOAD CENTER (New Feature)
    st.subheader("📂 Download Assets (For Testing)")
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        st.markdown("**1. The Job**")
        st.download_button("⬇️ Download JD", jd_text, file_name="job_description.md")
        
    with col_d2:
        st.markdown(f"**2. {results[0]['name']} (Strong)**")
        st.download_button(f"⬇️ CV ({results[0]['name']})", results[0]['cv_text'], file_name=f"{results[0]['name']}_CV.md")
        st.download_button(f"⬇️ Transcript ({results[0]['name']})", results[0]['transcript'], file_name=f"{results[0]['name']}_Transcript.md")

    with col_d3:
        st.markdown(f"**3. {results[1]['name']} (Weak)**")
        st.download_button(f"⬇️ CV ({results[1]['name']})", results[1]['cv_text'], file_name=f"{results[1]['name']}_CV.md")
        st.download_button(f"⬇️ Transcript ({results[1]['name']})", results[1]['transcript'], file_name=f"{results[1]['name']}_Transcript.md")

    st.divider()

    # 2. ANALYSIS PREVIEW
    t_jd, t1, t2 = st.tabs(["📄 Job Description", f"👤 {results[0]['name']}", f"👤 {results[1]['name']}"])
    
    with t_jd:
        st.markdown(jd_text)

    # Candidate Tabs
    tabs = [t1, t2]
    for i, tab in enumerate(tabs):
        if i < len(results):
            cand = results[i]
            with tab:
                c_info, c_docs = st.columns([1, 1])
                
                with c_info:
                    st.markdown(f"### Verdict: {cand['verdict']}")
                    st.info(f"**AI Analysis:** {cand['reasoning']}")
                    
                    st.progress(cand['role_fit_score']/10, text=f"Role Fit: {cand['role_fit_score']}/10")
                    st.progress(cand['comm_score']/10, text=f"Communication: {cand['comm_score']}/10")
                    st.progress(cand['culture_score']/10, text=f"Culture Fit: {cand['culture_score']}/10")

                with c_docs:
                    with st.expander("📄 View Generated CV", expanded=False):
                        st.text(cand['cv_text'])
                    
                    with st.expander("💬 View Interview Transcript", expanded=True):
                        st.text(cand['transcript'])
