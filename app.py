import importlib
import streamlit as st
import pandas as pd
import engine.pipeline
import engine.semantic_matcher
import engine.keyword_matcher
import engine.hybrid_scorer
import engine.parser

# Ensure Streamlit reloads edited engine modules across reruns
importlib.reload(engine.parser)
importlib.reload(engine.keyword_matcher)
importlib.reload(engine.semantic_matcher)
importlib.reload(engine.hybrid_scorer)
importlib.reload(engine.pipeline)

from engine.semantic_matcher import LocalSemanticMatcher
from engine.pipeline import run_smart_shortlist_pipeline
from engine.explainer import generate_explanation
from engine.bonus import scan_jd_bias

st.set_page_config(page_title="InternLoom Shortlisting Engine", layout="wide", page_icon="🎯")

st.title("🎯 InternLoom Smart Shortlisting Engine")
st.caption("Parallel CPU Lexical Matcher + Batched Local Semantic Inference • 100% Offline (Zero Cloud APIs)")

# Lazy-loaded model instance (boots in ~0.001s, no weight loading on startup)
@st.cache_resource
def get_semantic_matcher():
    return LocalSemanticMatcher()

semantic_matcher = get_semantic_matcher()

with st.sidebar:
    st.header("📂 1. Upload Documents")
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
    resume_files = st.file_uploader("Upload Resumes (Batch)", type=["pdf"], accept_multiple_files=True)

    st.header("⚙️ 2. Core Skills (must-have)")
    default_core = "React, Node.js, REST API, JavaScript, MongoDB, Git"
    core_skills_input = st.text_area("Must-Have Skills (weight 2×)", default_core)

    st.header("⚙️ 3. Bonus Skills (nice-to-have)")
    default_bonus = "SQL, TypeScript, Docker, Testing, AWS, Agile"
    bonus_skills_input = st.text_area("Good-to-Have Skills (weight 1×)", default_bonus)


if jd_file and resume_files:
    if st.button("🚀 Process & Rank Candidates", type="primary"):
        with st.spinner("Executing parallel parsing & batched semantic scoring..."):
            core_skills = [s.strip() for s in core_skills_input.split(",") if s.strip()]
            bonus_skills = [s.strip() for s in bonus_skills_input.split(",") if s.strip()]
            all_skills = core_skills + [s for s in bonus_skills if s not in core_skills]
            skill_weights = {s: 2.0 for s in core_skills}
            skill_weights.update({s: 1.0 for s in bonus_skills if s not in core_skills})

            resume_sources = {rf.name.replace(".pdf", ""): rf.read() for rf in resume_files}

            core_stack = [s for s in ["React", "Node.js"] if s in core_skills] or core_skills[:2]

            ranked, jd_text, timing_stats = run_smart_shortlist_pipeline(
                jd_source=jd_file.read(),
                resume_sources=resume_sources,
                required_skills=all_skills,
                skill_weights=skill_weights,
                semantic_matcher=semantic_matcher,
                core_skills=core_stack,
            )

            st.session_state["results"] = ranked
            st.session_state["jd_text"] = jd_text
            st.session_state["skills"] = all_skills
            st.session_state["timing"] = timing_stats

if "results" in st.session_state:
    results = st.session_state["results"]
    jd_text = st.session_state.get("jd_text", "")
    skills = st.session_state.get("skills", [])

    bias_warnings = scan_jd_bias(jd_text)
    if bias_warnings:
        with st.expander("🛡️ JD Inclusivity & Bias Audit (Bonus Feature)", expanded=True):
            for b in bias_warnings:
                st.warning(f"**[{b['risk']} Risk] {b['category']}:** {b['message']}")

    tabs = st.tabs(["🏆 Ranked Leaderboard", "🔍 Top 3 Explainability"])

    with tabs[0]:
        st.subheader("Leaderboard (Sensible Score Spread)")
        df = pd.DataFrame([{
            "Rank": c["rank"],
            "Candidate": c["name"],
            "Final Score": f"{c['score']}/100",
            "Semantic Fit": f"{c['semantic_pct']}%",
            "Keyword Match": f"{c['keyword_pct']}%",
            "Skills Found": len(c["matched"]),
            "Missing Skills": len(c["missing"])
        } for c in results])
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Top 3 Candidate Breakdown & Evidence")
        cols = st.columns(3)
        for i in range(min(3, len(results))):
            cand = results[i]
            with cols[i]:
                st.info(f"### #{cand['rank']} — {cand['name']}")
                st.metric("Final Score", f"{cand['score']}/100", delta=f"{cand['semantic_pct']}% Semantic")
                st.markdown(cand.get("explanation") or generate_explanation(cand))
