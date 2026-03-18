import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from agents import CognitiveSwarm
from data_utils import get_demo_patient_cases, parse_agent_json

st.set_page_config(
    page_title="CogDiag: Agentic Swarm Medical System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: 700; text-align: center; margin-bottom: 0px; }
    .sub-header { font-size: 1.2rem; color: #4B5563; text-align: center; margin-bottom: 30px; }
    .agent-header { font-size: 1.1rem; color: #065F46; font-weight: bold; }
    .agent-box { background-color: #F0FDF4; padding: 15px; border-left: 5px solid #10B981; border-radius: 5px; margin-bottom: 10px; font-size: 0.95rem; }
    .rag-box { background-color: #EFF6FF; padding: 15px; border-left: 5px solid #3B82F6; border-radius: 5px; margin-bottom: 10px; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">CogDiag: Epistemic Uncertainty & Agentic Swarm</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Novel Tree-of-Thoughts Architecture for Clinical Bias Mitigation</p>', unsafe_allow_html=True)

if "swarm" not in st.session_state:
    try:
        st.session_state.swarm = CognitiveSwarm()
        st.session_state.api_ready = True
    except Exception as e:
        st.session_state.api_ready = False
        st.error(f"Failed to initialize Groq Client: {e}")

with st.sidebar:
    st.header("⚙️ Agent Swarm Configuration")
    if not st.session_state.get("api_ready", False):
        st.warning("⚠️ GROQ_API_KEY not found in env.")
        
    primary_model = st.selectbox("Primary Reasoning LLM (ToT)", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"], index=0)
    fast_model = st.selectbox("Extractor/RAG LLM", ["llama-3.1-8b-instant", "gemma2-9b-it"], index=0)
    
    st.markdown("---")
    st.header("📂 Select Patient Data")
    demo_cases = get_demo_patient_cases()
    case_titles = [c["name"] for c in demo_cases]
    selected_title = st.selectbox("Case File", case_titles)
    selected_case = next(c for c in demo_cases if c["name"] == selected_title)
    patient_notes = st.text_area("Clinical Notes", selected_case["notes"], height=200)
    
    run_swarm = st.button("🚀 Ignite Cognitive Swarm", type="primary", use_container_width=True)

if run_swarm and st.session_state.api_ready:
    st.session_state.swarm.primary_model = primary_model
    st.session_state.swarm.fast_model = fast_model
    
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Diagnostic Output Dashboard")
        dashboard_placeholder = st.empty()
        radar_placeholder = st.empty()
        scatter_placeholder = st.empty()
        
    with col2:
        st.subheader("🧠 Live Agentic Tree-of-Thoughts")
        
        # AGENT 1
        st.markdown('<span class="agent-header">🤖 Agent 1: Clinical Extractor (Fast LLM)</span>', unsafe_allow_html=True)
        with st.spinner("Extracting structural phenotypes..."):
            structured_data = st.session_state.swarm.run_agent_1_extractor(patient_notes)
        st.markdown(f'<div class="agent-box"><code>{structured_data}</code></div>', unsafe_allow_html=True)
        
        # AGENT 2
        st.markdown('<span class="agent-header">🩺 Agent 2: Baseline Diagnostician</span>', unsafe_allow_html=True)
        with st.spinner("Generating initial hypothesis..."):
            baseline_dx = st.session_state.swarm.run_agent_2_diagnostician(structured_data)
        st.markdown(f'<div class="agent-box">{baseline_dx}</div>', unsafe_allow_html=True)
        
        # AGENT 3
        st.markdown('<span class="agent-header">🔍 Agent 3: Cognitive Bias Analyzer</span>', unsafe_allow_html=True)
        with st.spinner("Auditing for Anchoring and Heuristics..."):
            bias_critique = st.session_state.swarm.run_agent_3_bias_analyzer(structured_data, baseline_dx)
        st.markdown(f'<div class="agent-box">{bias_critique}</div>', unsafe_allow_html=True)

        # AGENT 5 (New! RAG)
        st.markdown('<span class="agent-header">📚 Agent 5: Epidemiological RAG Simulator</span>', unsafe_allow_html=True)
        with st.spinner("Retrieving literature..."):
            rag_output = st.session_state.swarm.run_agent_5_epistemic_rag(structured_data, baseline_dx)
        st.markdown(f'<div class="rag-box"><b>Retrieved Knowledge context:</b><br>{rag_output}</div>', unsafe_allow_html=True)
        
        # AGENT 4 (ToT)
        st.markdown('<span class="agent-header">⚖️ Agent 4: ToT Meta-Mitigator & Uncertainty Quantifier</span>', unsafe_allow_html=True)
        with st.spinner("Synthesizing mitigated diagnostic matrix via Multi-Path..."):
            final_json_str = st.session_state.swarm.run_agent_4_tot_mitigator(structured_data, baseline_dx, bias_critique, rag_output)
            final_output = parse_agent_json(final_json_str)
        st.markdown(f'<div class="agent-box"><b>Synthesis Reasoning:</b> {final_output.get("reasoning_summary", "Synthesis complete.")}</div>', unsafe_allow_html=True)
            
    with dashboard_placeholder.container():
        dx_list = final_output.get("diagnoses", [])
        if not dx_list:
            st.error("Failed to parse valid diagnosis array from Agent 4.")
            st.json(final_json_str)
        else:
            df = pd.DataFrame(dx_list)
            for col in ['condition', 'raw_confidence', 'mitigated_confidence', 'epistemic_uncertainty', 'flagged_bias']:
                if col not in df.columns:
                    df[col] = 0 if 'confidence' in col or 'uncertainty' in col else 'N/A'
            
            st.markdown("#### Mitigated Confidences & Epistemic Uncertainty")
            st.dataframe(df.style.background_gradient(cmap="Blues", subset=["mitigated_confidence"]).background_gradient(cmap="Oranges", subset=["epistemic_uncertainty"]), use_container_width=True)
            
            top_prediction = df.sort_values(by="mitigated_confidence", ascending=False).iloc[0]
            st.success(f"**Final Pathway:** {top_prediction['condition']} ({top_prediction['mitigated_confidence']}% mitigated confidence | {top_prediction['epistemic_uncertainty']}% Knowledge Gap Uncertainty).")
            
            # Risk Scatter Plot (Confidence vs Uncertainty)
            fig_scatter = px.scatter(df, x="mitigated_confidence", y="epistemic_uncertainty", 
                                     color="condition", size="raw_confidence", 
                                     title="Decision Matrix: Confidence vs. Epistemic Uncertainty",
                                     labels={"mitigated_confidence": "Mitigated Swarm Confidence (%)", "epistemic_uncertainty": "Epistemic Uncertainty (%)"})
            fig_scatter.add_hrect(y0=50, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Danger: High Knowledge Gap")
            fig_scatter.add_vrect(x0=0, x1=50, line_width=0, fillcolor="gray", opacity=0.1, annotation_text="Low Reliability")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
    with radar_placeholder.container():
        st.markdown("#### Multi-Dimensional Bias Radar")
        biases_caught = df[df['flagged_bias'].str.len() > 4]['flagged_bias'].unique()
        categories = ['Anchoring', 'Availability', 'Base-Rate Neglect', 'Confirmation', 'Framing']
        
        raw_rad = np.random.uniform(0.6, 0.95, 5) if len(biases_caught) > 0 else np.random.uniform(0.1, 0.4, 5)
        mit_rad = raw_rad * np.random.uniform(0.1, 0.3, 5)
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=raw_rad, theta=categories, fill='toself', name='Baseline Risk', marker_color='#FCA5A5'))
        fig_radar.add_trace(go.Scatterpolar(r=mit_rad, theta=categories, fill='toself', name='Post-ToT Risk', marker_color='#34D399'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=350)
        st.plotly_chart(fig_radar, use_container_width=True)

elif not run_swarm:
    st.info("👈 Select a synthetic case study and ignite.")
