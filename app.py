"""
app.py — Streamlit UI for the Toxic Comment Classifier API
Run with:  streamlit run app.py
Make sure api.py is already running on port 8000.
"""

import time
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_URL = "https://yasmine0421-toxic-comments.hf.space"

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
threshold = 0.5  # Default threshold for toxicity classification

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main { background-color: #0e1117; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card h2 { font-family: 'Space Mono', monospace; color: #e2e8f0; margin: 0; }
    .metric-card p  { color: #94a3b8; margin: 0.3rem 0 0 0; font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase; }

    /* Prediction box */
    .toxic-box {
        background: linear-gradient(135deg, #3d0000 0%, #1a0000 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        animation: pulse-red 2s infinite;
    }
    .clean-box {
        background: linear-gradient(135deg, #003d1a 0%, #001a0a 100%);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 1.5rem 2rem;
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.2); }
        50%       { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
    }

    .result-label {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .toxic-label { color: #ef4444; }
    .clean-label { color: #22c55e; }
    .prob-text   { color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    .sidebar-title {
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        color: #60a5fa;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Header */
    .app-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-bottom: 1px solid #334155;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        text-align: center;
    }
    .app-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
    }
    .app-subtitle { color: #64748b; font-size: 0.95rem; margin-top: 0.3rem; }

    div[data-testid="stTextArea"] textarea {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        letter-spacing: 0.03em;
        transition: all 0.2s;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #60a5fa, #3b82f6);
        transform: translateY(-1px);
    }
    /* Hide hamburger menu */
    #MainMenu {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <p class="app-title">Toxic Comment Detector</p>
  <p class="app-subtitle">BERT fine-tuned on Jigsaw · Real-time toxicity analysis</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">API Status</p>', unsafe_allow_html=True)

    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        if r.status_code == 200:
            info = r.json()
            st.success("API Online")
            m = info.get("metrics", {})
            if m.get("test_f1"):
                st.markdown(f"- **Accuracy**: {m.get('test_accuracy','—')}")
                st.markdown(f"- **F1-Score**: {m.get('test_f1','—')}")
                st.markdown(f"- **AUC-ROC**: {m.get('test_auc','—')}")
        else:
            st.error("API returned an error")
    except Exception:
        st.error("API Offline")

    st.markdown("---")
    st.markdown('<p class="sidebar-title">About</p>', unsafe_allow_html=True)
    st.caption(
        "This app uses a BERT model fine-tuned on the "
        "[Jigsaw Toxic Comment dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)."
    )

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Single Comment", "Batch Analysis"])

# ─── TAB 1: Single Comment ────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Analyze a Comment")
        user_text = st.text_area(
            "Enter comment text",
            height=160,
            placeholder="Type or paste a comment here...",
            label_visibility="collapsed"
        )
        analyze_btn = st.button("Analyze", key="single_btn")

    with col2:
        st.markdown("### Try These Examples")
        examples = {
            "Normal": "I really enjoyed reading this article, thanks for sharing!",
            "Toxic 1": "You're an absolute idiot and no one wants you here.",
            "Toxic 2": "Go kill yourself, you worthless piece of trash.",
            "Debate": "I strongly disagree with your political views, but I respect your right to hold them.",
            "Mild": "What a stupid question, did you even read the article?",
        }
        for label, ex_text in examples.items():
            if st.button(label, key=f"ex_{label}"):
                st.session_state["example_text"] = ex_text
                st.rerun()

        if "example_text" in st.session_state:
            st.info(f"**Selected:** {st.session_state['example_text'][:80]}...")

    # Auto-fill from example selection
    if "example_text" in st.session_state and not user_text:
        user_text = st.session_state["example_text"]

    # Run prediction
    if analyze_btn and user_text.strip():
        with st.spinner("Analyzing..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"text": user_text, "threshold": 0.5},
                    timeout=10
                )
                resp.raise_for_status()
                result = resp.json()

                st.markdown("---")
                toxic_p = result["toxic_prob"]
                clean_p = result["clean_prob"]
                is_toxic = result["is_toxic"]

                box_class   = "toxic-box" if is_toxic else "clean-box"
                label_class = "toxic-label" if is_toxic else "clean-label"

                st.markdown(f"""
                <div class="{box_class}">
                    <p class="result-label {label_class}"> {result['label']}</p>
                    <p class="prob-text">
                        Toxic probability: <strong>{toxic_p:.1%}</strong> &nbsp;|&nbsp;
                        Clean probability: <strong>{clean_p:.1%}</strong> &nbsp;|&nbsp;
                        Latency: {result.get('latency_ms', '—')} ms
                    </p>
                </div>
                """, unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure `uvicorn api:app --port 8000` is running.")
            except Exception as e:
                st.error(f"Error: {e}")

    elif analyze_btn:
        st.warning("Please enter some text to analyze.")

# ─── TAB 2: Batch Analysis ────────────────────────────────────────────────────
with tab2:
    st.markdown("### Batch Comment Analysis")
    st.caption("Enter one comment per line (up to 32 at a time).")

    batch_input = st.text_area(
        "Comments (one per line)",
        height=220,
        placeholder="Comment 1\nComment 2\nComment 3\n...",
        label_visibility="collapsed"
    )
    batch_btn = st.button("Analyze All", key="batch_btn")

    if batch_btn and batch_input.strip():
        texts = [t.strip() for t in batch_input.strip().split("\n") if t.strip()]
        if len(texts) > 32:
            st.warning("Maximum 32 comments per batch. Only the first 32 will be analyzed.")
            texts = texts[:32]

        with st.spinner(f"Analyzing {len(texts)} comments..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict/batch",
                    json={"texts": texts, "threshold": 0.5},
                    timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                preds = data["predictions"]

                # Summary metrics
                toxic_count = sum(1 for p in preds if p["is_toxic"])
                clean_count = len(preds) - toxic_count
                avg_toxic_p = sum(p["toxic_prob"] for p in preds) / len(preds)

                c1, c2, c3, c4 = st.columns(4)
                for col, val, label in [
                    (c1, len(preds),            "Total Comments"),
                    (c2, toxic_count,            "Toxic"),
                    (c3, clean_count,            "Clean"),
                    (c4, f"{avg_toxic_p:.1%}",  "Avg Toxicity"),
                ]:
                    col.markdown(f"""
                    <div class="metric-card">
                        <h2>{val}</h2>
                        <p>{label}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Distribution chart
                col_a, col_b = st.columns(2)
                with col_a:
                    pie_fig = px.pie(
                        values=[toxic_count, clean_count],
                        names=["Toxic", "Clean"],
                        color_discrete_map={"Toxic": "#ef4444", "Clean": "#22c55e"},
                        title="Toxicity Distribution",
                        hole=0.5
                    )
                    pie_fig.update_layout(
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font_color="#e2e8f0", height=280,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)

                with col_b:
                    probs = [p["toxic_prob"] for p in preds]
                    hist_fig = px.histogram(
                        x=probs, nbins=20,
                        color_discrete_sequence=["#3b82f6"],
                        title="Toxic Probability Distribution",
                        labels={"x": "Toxic Probability", "y": "Count"}
                    )
                    hist_fig.add_vline(x=threshold, line_dash="dash", line_color="#fbbf24",
                                       annotation_text=f"Threshold ({threshold})",
                                       annotation_font_color="#fbbf24")
                    hist_fig.update_layout(
                        paper_bgcolor="#0e1117", plot_bgcolor="#1e293b",
                        font_color="#e2e8f0", height=280,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(hist_fig, use_container_width=True)

                # Results table
                st.markdown("### Results")
                df_results = pd.DataFrame([{
                    "Comment":       p["text"][:80] + "..." if len(p["text"]) > 80 else p["text"],
                    "Label":         p["label"],
                    "Toxic Prob":    f"{p['toxic_prob']:.3f}",
                    "Clean Prob":    f"{p['clean_prob']:.3f}",
                } for p in preds])

                def style_label(val):
                    if val == "TOXIC":
                        return "background-color: #450a0a; color: #ef4444; font-weight: bold"
                    return "background-color: #14532d; color: #22c55e; font-weight: bold"

                styled = df_results.style.map(style_label, subset=["Label"])
                st.dataframe(styled, use_container_width=True, height=400)

                # Download CSV
                csv = df_results.to_csv(index=False)
                st.download_button(
                    "Download Results as CSV",
                    data=csv,
                    file_name="toxicity_results.csv",
                    mime="text/csv"
                )

                st.caption(f"Batch latency: {data.get('latency_ms', '—')} ms")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API.")
            except Exception as e:
                st.error(f"Error: {e}")

    elif batch_btn:
        st.warning("Please enter at least one comment.")


