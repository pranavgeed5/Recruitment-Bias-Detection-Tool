"""
app.py — AI-Based Recruitment Bias Detection Tool
Run: streamlit run app.py
"""

import os
import sys
import pickle
import pathlib
import textwrap

import streamlit as st
import pandas as pd

# ── Bootstrap: generate data + train if models missing ───────────────────────
BASE = pathlib.Path(__file__).parent

def ensure_models():
    if not (BASE / "model_best.pkl").exists():
        st.info("⏳ First run — generating dataset and training models (≈20 sec)…")
        with st.spinner("Training…"):
            import importlib, sys as _sys
            # Run generate_dataset
            spec = importlib.util.spec_from_file_location("gd", BASE / "generate_dataset.py")
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # Run model training
            spec2 = importlib.util.spec_from_file_location("mt", BASE / "model.py")
            mod2  = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(mod2)
        st.success("✅ Models ready!")
        st.rerun()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RecruitLens — Bias Detector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Dark themed app */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

h1, h2, h3 { color: #f0f2f8; }

/* Score ring container */
.score-ring {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
}

/* Bias tag pills */
.bias-tag {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 4px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.tag-gender { background: #3d1a6e; color: #d4a9ff; border: 1px solid #7b39d4; }
.tag-age    { background: #1a3d2e; color: #7fffc4; border: 1px solid #2ea86a; }
.tag-name   { background: #3d2610; color: #ffcf82; border: 1px solid #d4892a; }
.tag-none   { background: #1a2030; color: #8090b0; border: 1px solid #3a4560; }

/* Highlighted text box */
.highlight-box {
    background: #13161e;
    border: 1px solid #2a2f3f;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.8;
    color: #b0bcd4;
    white-space: pre-wrap;
    word-break: break-word;
}
.highlight-box mark {
    background: #4a1e1e;
    color: #ff8080;
    border-radius: 4px;
    padding: 1px 5px;
    font-weight: 600;
}

/* Suggestion card */
.suggestion-card {
    background: #131820;
    border-left: 3px solid #4a7cff;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #a0b0d0;
    line-height: 1.6;
}

/* Metric cards */
.metric-card {
    background: #13161e;
    border: 1px solid #2a2f3f;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #4a7cff; }
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.75rem; color: #6070a0; letter-spacing: 0.06em; text-transform: uppercase; margin-top: 4px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3355ff 0%, #6622cc 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
    font-family: 'Space Grotesk', sans-serif;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* Text areas */
.stTextArea textarea {
    background: #13161e !important;
    color: #d0d8f0 !important;
    border: 1px solid #2a2f3f !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.84rem !important;
}

/* Divider */
hr { border-color: #2a2f3f; }
</style>
""", unsafe_allow_html=True)


# ── Load assets ───────────────────────────────────────────────────────────────
ensure_models()

@st.cache_resource
def load_assets():
    from utils import load_model
    from preprocessing import load_vectorizer
    model = load_model(str(BASE / "model_best.pkl"))
    vec   = load_vectorizer(str(BASE / "vectorizer.pkl"))
    return model, vec

model, vectorizer = load_assets()

from utils import compute_bias_score, generate_explanation, highlight_bias_words


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem 0;">
  <h1 style="font-size:2.6rem; font-weight:700; letter-spacing:-0.02em; margin:0;">
    ⚖️ RecruitLens
  </h1>
  <p style="color:#6070a0; font-size:1rem; margin-top:0.4rem; letter-spacing:0.04em;">
    AI-BASED RECRUITMENT BIAS DETECTION TOOL
  </p>
</div>
<hr>
""", unsafe_allow_html=True)


# ── Input section ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📄 Resume Input")
    upload = st.file_uploader("Upload resume (TXT or PDF)", type=["txt", "pdf"])

    resume_text = ""
    if upload:
        if upload.type == "application/pdf":
            try:
                import pdfplumber
                with pdfplumber.open(upload) as pdf:
                    resume_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                st.warning("pdfplumber not installed. Paste text below instead.")
        else:
            resume_text = upload.read().decode("utf-8", errors="ignore")

    resume_text = st.text_area(
        "Or paste resume text here",
        value=resume_text,
        height=280,
        placeholder="Paste resume content here…",
    )

with col_right:
    st.markdown("### 💼 Job Description")
    jd_text = st.text_area(
        "Paste job description (optional — used for context)",
        height=280,
        placeholder="Paste job description here…",
    )

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🔍 Analyze Bias", use_container_width=True)


# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze_btn:
    if not resume_text.strip():
        st.error("Please provide resume text before analyzing.")
        st.stop()

    with st.spinner("Scanning for bias signals…"):
        result      = compute_bias_score(resume_text, model, vectorizer)
        explanation = generate_explanation(result, jd_text)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## 📊 Analysis Results")

    # ── Score + category cards ─────────────────────────────────────────────
    score = result["bias_score"]
    color = "#ff4444" if score >= 60 else "#ffaa00" if score >= 30 else "#22cc88"
    level = "HIGH RISK" if score >= 60 else "MODERATE" if score >= 30 else "LOW RISK"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{color};">{score}</div>
          <div class="metric-label">Bias Score / 100</div>
          <div style="margin-top:6px; font-size:0.75rem; color:{color}; font-weight:600;">{level}</div>
        </div>""", unsafe_allow_html=True)

    icons = {"gender": ("♂♀", "#d4a9ff", "Gender Bias"),
             "age"   : ("⏳", "#7fffc4",  "Age Bias"),
             "name"  : ("🪪", "#ffcf82",  "Name Bias")}

    for col, (key, (icon, clr, label)) in zip([c2, c3, c4], icons.items()):
        detected = result[f"{key}_bias"]
        status   = "DETECTED" if detected else "CLEAR"
        s_color  = clr if detected else "#4a5570"
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:{s_color};">{icon}</div>
              <div class="metric-label">{label}</div>
              <div style="margin-top:6px; font-size:0.75rem; color:{s_color}; font-weight:600;">{status}</div>
            </div>""", unsafe_allow_html=True)

    # ── Progress bar ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**Bias Score: {score}/100**")
    bar_color = "#ff4444" if score >= 60 else "#ffaa00" if score >= 30 else "#22cc88"
    st.markdown(f"""
    <div style="background:#1a1e28; border-radius:8px; height:18px; overflow:hidden; margin-bottom:1rem;">
      <div style="width:{score}%; background:{bar_color}; height:100%; border-radius:8px;
           transition:width 0.8s ease; display:flex; align-items:center; padding-left:8px;">
        <span style="font-size:0.7rem; color:white; font-weight:600;">{score}%</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Highlighted text ───────────────────────────────────────────────────
    st.markdown("### 🔦 Highlighted Resume Text")
    st.caption("Bias-related words are highlighted in red.")

    # Convert **WORD** markers to <mark>
    import re
    display_text = re.sub(
        r"\*\*([A-Z0-9 '\-]+)\*\*",
        r"<mark>\1</mark>",
        result["highlighted"]
    )
    display_text = display_text.replace("\n", "<br>")
    st.markdown(f'<div class="highlight-box">{display_text}</div>', unsafe_allow_html=True)

    # ── Found words summary ────────────────────────────────────────────────
    st.markdown("<br>**Detected Bias Words:**", unsafe_allow_html=True)
    tag_map = {"gender": "tag-gender", "age": "tag-age", "name": "tag-name"}
    any_found = False
    tags_html = ""
    for cat, words in result["found_words"].items():
        for w in words:
            tags_html += f'<span class="bias-tag {tag_map[cat]}">{cat}: {w}</span>'
            any_found = True
    if not any_found:
        tags_html = '<span class="bias-tag tag-none">No bias keywords detected</span>'
    st.markdown(tags_html, unsafe_allow_html=True)

    # ── Explanation ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💡 Explanation")
    st.info(explanation["explanation"])

    # ── Suggestions ───────────────────────────────────────────────────────
    st.markdown("### ✅ Suggestions")
    for s in explanation["suggestions"]:
        st.markdown(f'<div class="suggestion-card">→ {s}</div>', unsafe_allow_html=True)

    # ── Model confidence ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🤖 Model Details"):
        st.markdown(f"- **ML Model Confidence (bias):** `{result['model_prob']}%`")
        st.markdown(f"- **Keyword Score Component:** `{min(sum(len(v) for v in result['found_words'].values()) * 12, 60)}/60`")
        st.markdown(f"- **Model Score Component:** `{round(result['model_prob'] * 0.4, 1)}/40`")
        st.markdown(f"- **Combined Final Score:** `{result['bias_score']}/100`")
        st.markdown("---")
        st.markdown("Two models trained: **Logistic Regression** + **Random Forest**. Best model selected automatically.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center; color:#3a4560; font-size:0.78rem; padding:1rem 0;">
  RecruitLens · AI-Based Recruitment Bias Detection · BTech Final Year Project
</div>
""", unsafe_allow_html=True)
