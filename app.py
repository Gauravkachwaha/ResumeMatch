import sys
import os
import warnings

# ── Must happen BEFORE any langgraph/langchain import ──────────────────────────
# Suppresses warnings at the OS env level so they never reach stderr.
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    # Add Anaconda DLL directories so C extensions (sqlite3, chromadb) load correctly
    for _p in [r"C:\Anaconda\Library\bin", r"C:\Anaconda\DLLs", r"C:\Anaconda"]:
        if os.path.exists(_p):
            try:
                os.add_dll_directory(_p)
            except Exception:
                pass
            os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")


import uuid
import streamlit as st

from src.matching_agent import run_agent, get_graph_ascii_visualization





# Page Configuration
st.set_page_config(
    page_title="AI Resume Job Matcher Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern visual aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .candidate-badge {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-high {
        color: #15803D;
        font-weight: 700;
    }
    .score-med {
        color: #B45309;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_agent_state" not in st.session_state:
    st.session_state.latest_agent_state = None


# Sidebar
with st.sidebar:
    st.markdown("## 🤖 Recruiter Control Panel")
    
    st.markdown("### ⚙️ System Configuration")
    st.info("""
    - **LLM Provider**: OpenRouter (`openrouter/free`)
    - **Embeddings**: `all-MiniLM-L6-v2`
    - **Vector DB**: ChromaDB
    - **Keyword Search**: BM25
    - **Agent Framework**: LangGraph
    """)

    st.markdown("### 🎯 Multi-Round Screening Config")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Round 1 Shortlist", "Top 10")
    with col_s2:
        st.metric("Round 3 Finalists", "Top 3")

    st.markdown("---")
    st.markdown(f"**Session Thread ID**: `{st.session_state.thread_id[:8]}...`")
    
    if st.button("🆕 New Conversation", use_container_width=True, type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.latest_agent_state = None
        # Do NOT call st.rerun() — session state change is picked up on next natural rerun

    st.markdown("---")
    with st.expander("🗺️ LangGraph State Machine Visualizer"):
        st.code(get_graph_ascii_visualization(), language="text")


# Main Interface Header
st.markdown("<div class='main-header'>🤖 AI Resume Job Matcher & Recruiter Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>LangGraph-powered stateful agent with hybrid RAG retrieval, candidate ranking, explainability tools, and multi-round screening.</div>", unsafe_allow_html=True)


# Quick Action Example Prompts
st.markdown("##### ⚡ Quick Recruiter Actions")
col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)

quick_prompt = None
with col_q1:
    if st.button("🔍 Find Python Devs", use_container_width=True):
        quick_prompt = "Find Python developers with 3+ years experience in FastAPI and PostgreSQL."
with col_q2:
    if st.button("📊 Compare Top 3", use_container_width=True):
        quick_prompt = "Compare the top 3 candidates side by side."
with col_q3:
    if st.button("💡 Explain Ranking", use_container_width=True):
        quick_prompt = "Why did the top candidate rank higher than the second candidate?"
with col_q4:
    if st.button("🎯 Interview Questions", use_container_width=True):
        quick_prompt = "Generate tailored interview questions for the top candidate."
with col_q5:
    if st.button("🏆 Full Screening", use_container_width=True):
        quick_prompt = "Run a full multi-round screening process on the top Python candidates."


# Quick Action Buttons & Chat Input Capture
user_text_to_process = None

if quick_prompt:
    user_text_to_process = quick_prompt
elif user_prompt := st.chat_input("Enter job description, candidate query, or refinement prompt..."):
    user_text_to_process = user_prompt


# Process User Input if captured
if user_text_to_process:
    st.session_state.chat_history.append({"role": "user", "content": user_text_to_process})
    
    with st.spinner("🤖 Recruiter Agent reasoning & executing graph nodes..."):
        try:
            agent_res = run_agent(user_text_to_process, thread_id=st.session_state.thread_id)
            st.session_state.latest_agent_state = agent_res
            report_output = agent_res.get("report", "Agent execution completed.")
        except Exception as _exc:
            import traceback
            report_output = f"⚠️ Agent error: {_exc}\n\n```\n{traceback.format_exc()}\n```"
            st.error(report_output)

    st.session_state.chat_history.append({"role": "assistant", "content": report_output})


# Render Conversation History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



# Active State Outputs & Rich Cards (Below Chat)
state = st.session_state.latest_agent_state
if state:
    st.markdown("---")
    st.markdown("### 📋 Active Agent State & Candidate Analytics")
    
    tab_candidates, tab_screening, tab_feedback, tab_debug = st.tabs([
        "👥 Candidate Shortlist", 
        "🏆 Multi-Round Screening", 
        "🔄 Human Feedback & Refinement",
        "🔍 Agent State Debugger"
    ])
    
    # TAB 1: Ranked Candidates
    with tab_candidates:
        ranked = state.get("ranked_candidates", [])
        parsed_jd = state.get("parsed_jd", {})
        
        if parsed_jd:
            st.markdown(f"**Target Role**: `{parsed_jd.get('title', 'N/A')}` | **Must-Have Skills**: `{', '.join(parsed_jd.get('must_have_skills', []))}`")
        
        if ranked:
            st.markdown(f"Showing **{len(ranked)}** ranked candidates from RAG + JobMatcher engine:")
            
            for i, cand in enumerate(ranked, 1):
                c_name = cand.get("candidate_name", "Candidate")
                score = cand.get("match_score", 0)
                exp = cand.get("experience_years", 0.0)
                matched = cand.get("matched_skills", [])
                missing = cand.get("missing_skills", [])
                reasoning = cand.get("reasoning", "")
                excerpts = cand.get("relevant_excerpts", [])
                
                score_class = "score-high" if score >= 60 else "score-med"
                
                with st.expander(f"#{i}. {c_name} — Score: {score}/100 (~{exp} yrs exp)", expanded=(i == 1)):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown(f"**Match Score**: <span class='{score_class}'>{score}/100</span>", unsafe_allow_html=True)
                        st.markdown(f"**Experience**: {exp} years")
                        st.markdown(f"**Matched Skills**: {', '.join(matched) if matched else 'None'}")
                        if missing:
                            st.markdown(f"**Missing Skills**: `:red[{', '.join(missing)}]`")
                    with c2:
                        st.markdown(f"**Matcher Reasoning**: {reasoning}")
                        if excerpts:
                            st.markdown(f"**Resume Excerpts**: *\"{excerpts[0]}\"*")
        else:
            st.info("No candidates loaded in active state yet. Submit a search query above.")

    # TAB 2: Multi-Round Screening Funnel
    with tab_screening:
        screening_round = state.get("screening_round", 0)
        screening_status = state.get("screening_status", "NOT_STARTED")
        final_recs = state.get("final_recommendations", [])
        
        st.markdown(f"**Screening Status**: `{screening_status}` | **Current Round**: `Round {screening_round}/3`")
        
        # Step Progress
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("#### Round 1: Initial Screen")
            st.caption("Top 10 from ~100 raw resumes via hybrid search")
            if screening_round >= 1:
                st.success("✅ Completed (Top 10 Shortlisted)")
        with p2:
            st.markdown("#### Round 2: Deep Analysis")
            st.caption("Qualitative evidence & skill gap assessment")
            if screening_round >= 2:
                st.success("✅ Completed (Top 3 Finalists)")
        with p3:
            st.markdown("#### Round 3: Final Evaluation")
            st.caption("Mandatory must-have requirement audit")
            if screening_round >= 3:
                st.success("✅ Completed (Recommendations Issued)")
                
        if final_recs:
            st.markdown("---")
            st.markdown("#### 🥇 Final Recommendations")
            for rec in final_recs:
                name = rec.get("candidate_name", "Candidate")
                decision = rec.get("recommendation", "NO-HIRE")
                confidence = rec.get("confidence", 0)
                reasoning = rec.get("reasoning", "")
                focus = rec.get("interview_focus_areas", [])
                
                if decision == "HIRE":
                    st.success(f"**{name}** — Recommendation: **HIRE** (Confidence: {confidence}%)")
                else:
                    st.error(f"**{name}** — Recommendation: **NO-HIRE** (Confidence: {confidence}%)")
                
                st.markdown(f"**Rationale**: {reasoning}")
                if focus:
                    st.markdown(f"**Interview Focus**: {', '.join(focus)}")
                st.markdown("---")

    # TAB 3: Human Feedback & Requirement Refinement
    with tab_feedback:
        st.markdown("#### 🔄 Recruiter Human Feedback Loop")
        st.markdown("Provide feedback to refine requirements, add mandatory skills, or adjust experience weightings.")
        
        fb_input = st.text_input("Recruiter Feedback / Requirement Adjustment:", placeholder="e.g. 'Make Docker mandatory and prioritize FastAPI experience.'")
        if st.button("🚀 Apply Feedback & Re-Rank Candidates"):
            if fb_input:
                st.session_state.chat_history.append({"role": "user", "content": fb_input})
                with st.spinner("🤖 Applying Recruiter Feedback & re-ranking candidates..."):
                    try:
                        agent_res = run_agent(fb_input, thread_id=st.session_state.thread_id)
                        st.session_state.latest_agent_state = agent_res
                        report_output = agent_res.get("report", "Agent execution completed.")
                        st.success("Feedback applied successfully!")
                    except Exception as _exc:
                        import traceback
                        report_output = f"⚠️ Agent error: {_exc}\n\n```\n{traceback.format_exc()}\n```"
                        st.error(report_output)
                st.session_state.chat_history.append({"role": "assistant", "content": report_output})


            else:
                st.warning("Please enter feedback before submitting.")


    # TAB 4: Agent State Debugger
    with tab_debug:
        st.markdown("#### 🔍 LangGraph AgentState Inspector")
        debug_state = {
            "current_intent": state.get("current_intent"),
            "raw_jd": state.get("raw_jd"),
            "parsed_jd": state.get("parsed_jd"),
            "top_k": state.get("top_k"),
            "screening_round": state.get("screening_round"),
            "screening_status": state.get("screening_status"),
            "candidate_count": len(state.get("ranked_candidates", [])),
            "thread_id": st.session_state.thread_id
        }
        st.json(debug_state)
