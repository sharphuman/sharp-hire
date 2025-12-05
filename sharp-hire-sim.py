import streamlit as st
import pandas as pd
import json
import os
from anthropic import Anthropic

# --- CONFIGURATION ---
st.set_page_config(page_title="Sharp Hire SIM v2.0", page_icon="🎲", layout="wide")

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
    /* Comparison Table Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #333;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'sim_data' not in st.session_state: st.session_state.sim_data = None
if 'processing_status' not in st.session_state: st.session_state.processing_status = "Ready to Simulate."

# --- SECRETS ---
try:
    # We only need Anthropic for text generation/simulation
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

def generate_full_scenario(job_title, industry, level):
    """
    Generates 1 JD and 3 Candidates (Good, Avg, Bad) in one go.
    """
    prompt = f"""
    You are a Hiring Simulation Engine. Generate a full recruitment scenario.
    
    **PARAMETERS:**
    - Role: {job_title}
    - Industry: {industry}
    - Level: {level}

    **TASK:**
    1. Create a short, professional Job Description (JD).
    2. Create 3 Candidates with distinct profiles:
       - **Candidate A (The Unicorn):** High skills, great culture fit.
       - **Candidate B (The Stretch):** Good attitude, missing some key tech skills.
       - **Candidate C (The Red Flag):** Good paper resume, but arrogant/evasive in interview.
    3. For each candidate, generate a brief CV summary and a dialogue transcript (approx 400 words) of their interview.

    **OUTPUT JSON STRUCTURE:**
    {{
        "job_description": "Full text of JD...",
        "candidates": [
            {{
                "id": "A",
                "name": "Name",
                "vibe": "The Unicorn",
                "cv_summary": "...",
                "transcript": "Recruiter: ... \nCandidate: ..."
            }},
            {{
                "id": "B",
                "name": "Name",
                "vibe": "The Stretch",
                "cv_summary": "...",
                "transcript": "..."
            }},
            {{
                "id": "C",
                "name": "Name",
                "vibe": "The Red Flag",
                "cv_summary": "...",
                "transcript": "..."
            }}
        ]
    }}
    """
    
    try:
        msg = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=4000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return clean_json(msg.content[0].text)
    except Exception as e:
        return {"error": str(e)}

def analyze_candidates(scenario_data):
    """
    Analyzes all 3 candidates against the JD.
    """
    jd = scenario_data['job_description']
    cands = scenario_data['candidates']
    
    prompt = f"""
    Analyze these 3 candidates against the Job Description.
    
    JOB DESCRIPTION: {jd[:1000]}...
    
    CANDIDATE DATA: {json.dumps(cands)}
    
    **TASK:**
    Return a JSON with detailed scoring for each.
    
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
                "role_fit_score": 6,
                "comm_score": 8,
                "tech_score": 5,
                "culture_score": 9,
                "verdict": "Maybe / Train",
                "reasoning": "..."
            }},
            {{
                "id": "C",
                "role_fit_score": 8,
                "comm_score": 4,
                "tech_score": 8,
                "culture_score": 2,
                "verdict": "No Hire",
                "reasoning": "..."
            }}
        ]
    }}
    """
    
    try:
        msg = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=4000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        return clean_json(msg.content[0].text)
    except Exception as e:
        return {"error": str(e)}

# --- UI LAYOUT ---

st.title("🎲 Sharp Hire: Simulation Mode")
st.markdown("Generate and compare synthetic candidates for demo/training.")

# INPUTS
c1, c2, c3 = st.columns(3)
with c1: job_title = st.text_input("Role Title", "Senior React Developer")
with c2: industry = st.text_input("Industry", "Fintech")
with c3: level = st.selectbox("Seniority", ["Junior", "Mid-Level", "Senior/Lead", "Executive"])

if st.button("🎲 Run Simulation", type="primary", use_container_width=True):
    with st.status("⚙️ Generating Simulation Data...", expanded=True) as status:
        st.write("📝 Drafting Job Description & Candidates...")
        scenario = generate_full_scenario(job_title, industry, level)
        
        if "error" in scenario:
            st.error(scenario['error'])
            st.stop()
            
        st.write("🧠 Analyzing Candidates...")
        analysis = analyze_candidates(scenario)
        
        # Merge Data
        final_data = []
        for i, cand in enumerate(scenario['candidates']):
            # Find matching analysis
            an = next(item for item in analysis['analyses'] if item["id"] == cand["id"])
            merged = {**cand, **an}
            final_data.append(merged)
            
        st.session_state.sim_data = {"jd": scenario['job_description'], "results": final_data}
        status.update(label="Simulation Complete!", state="complete", expanded=False)

# RESULTS
if st.session_state.sim_data:
    results = st.session_state.sim_data['results']
    
    st.divider()
    
    # 1. COMPARISON TABLE
    st.subheader("📊 Candidate Leaderboard")
    
    # Create simple dataframe for display
    df_data = []
    for r in results:
        df_data.append({
            "Name": r['name'],
            "Archetype": r['vibe'],
            "Role Fit": r['role_fit_score'],
            "Tech Skills": r['tech_score'],
            "Culture": r['culture_score'],
            "Verdict": r['verdict']
        })
    
    df = pd.DataFrame(df_data)
    
    # Use standard dataframe with simple highlighting
    st.dataframe(
        df,
        column_config={
            "Role Fit": st.column_config.ProgressColumn("Fit", format="%d", min_value=0, max_value=10),
            "Tech Skills": st.column_config.ProgressColumn("Tech", format="%d", min_value=0, max_value=10),
            "Culture": st.column_config.ProgressColumn("Vibe", format="%d", min_value=0, max_value=10),
        },
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # 2. DEEP DIVE TABS
    t_jd, t1, t2, t3 = st.tabs(["📄 Job Description", f"👤 {results[0]['name']}", f"👤 {results[1]['name']}", f"👤 {results[2]['name']}"])
    
    with t_jd:
        st.markdown(st.session_state.sim_data['jd'])

    # Loop through candidates to create tabs
    tabs = [t1, t2, t3]
    for i, tab in enumerate(tabs):
        cand = results[i]
        with tab:
            c_info, c_chat = st.columns([1, 1])
            
            with c_info:
                st.markdown(f"### Verdict: {cand['verdict']}")
                st.info(f"**AI Reasoning:** {cand['reasoning']}")
                
                st.markdown("#### 📝 CV Summary")
                st.caption(cand['cv_summary'])
                
                st.markdown("#### 📊 Scores")
                st.progress(cand['role_fit_score']/10, text=f"Role Fit: {cand['role_fit_score']}/10")
                st.progress(cand['comm_score']/10, text=f"Communication: {cand['comm_score']}/10")
                st.progress(cand['culture_score']/10, text=f"Culture Fit: {cand['culture_score']}/10")

            with c_chat:
                st.markdown("#### 💬 Interview Transcript")
                with st.container(border=True):
                    st.text(cand['transcript'])
