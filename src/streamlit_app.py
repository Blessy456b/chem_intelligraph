# streamlit_app.py
import os
import streamlit as st
from agents_orchestrator import run_multi_agent, load_summary_memory
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ChemIntelliGraph — Agentic RAG Lab", layout="wide")
st.title("🧪 ChemIntelliGraph — Agentic RAG Lab")
st.markdown("Enter two reactants. System tries RAG → LLM → Fact-check (SerpAPI) → Safety → Memory")

col1, col2 = st.columns(2)
with col1:
    reactant_a = st.text_input("Reactant A", value="Zn")
with col2:
    reactant_b = st.text_input("Reactant B", value="HCl")

mode = st.radio("Choose AI Mode", ["Strict Mode (Verified Only)", "Creative Mode (Scientific Reasoning)"])
selected_mode = "strict" if "Strict" in mode else "creative"

if st.button("Analyze Reaction"):
    with st.spinner("Running multi-agent pipeline..."):
        res = run_multi_agent(reactant_a.strip(), reactant_b.strip(), mode=selected_mode)
    st.markdown("### ⚗️ Result")
    if res.get("reaction"):
        st.markdown(f"**Reaction:** `{res['reaction']}`")
        st.write(f"**Confidence:** {res.get('confidence', 'N/A')}")
        st.write(f"**Hazard Level:** {res.get('hazard', 'Unknown')}")
        st.write("**Sources:** " + (", ".join(res.get("sources", [])) or "None"))
        st.success("Result produced.")
    else:
        st.warning(res.get("message", "No reaction found."))

    st.markdown("### 🧭 Agent Trace (chronological)")
    for step in res.get("trace", []):
        st.json(step)

    st.markdown("### 💾 Memory & History")
    mem = load_summary_memory()
    if mem:
        st.dataframe(mem[::-1])  # recent first
    else:
        st.write("No memory stored yet.")
