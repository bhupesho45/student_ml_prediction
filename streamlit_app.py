from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("student_model.joblib")

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7f4ec 0%, #e7f1ed 100%);
    }
    .hero {
        padding: 2.5rem 0 1.5rem;
    }
    .eyebrow {
        color: #197268;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: #183b39;
        font-size: 3rem;
        line-height: 1.05;
        margin: 0.4rem 0 0.8rem;
    }
    .hero p {
        color: #526663;
        font-size: 1.05rem;
        margin: 0;
    }
    .result {
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
        text-align: center;
    }
    .result.pass {
        background: #d9f2e6;
        border: 1px solid #6fc39d;
        color: #145b3e;
    }
    .result.fail {
        background: #fbe2dc;
        border: 1px solid #e49787;
        color: #8a3027;
    }
    .result strong {
        display: block;
        font-size: 2rem;
        margin-top: 0.25rem;
    }
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
        --ink: #142b3a;
        --muted: #61727c;
        --line: #dbe4e7;
        --paper: #f5f8f7;
        --teal: #087f8c;
        --teal-dark: #075761;
    }
    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #102f3d; border-right: 0; }
    [data-testid="stSidebar"] * { color: #e6f0f1; }
    [data-testid="stSidebar"] hr { border-color: rgba(230, 240, 241, 0.16); }
    .block-container { max-width: 1240px; padding: 3.5rem 3rem 4rem; }
    .brand { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 2.5rem; }
    .brand-mark {
        display: grid; place-items: center; width: 2.3rem; height: 2.3rem;
        border-radius: 9px; background: #f1c46a; color: #102f3d;
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.15rem;
    }
    .brand-name { color: #fff; font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 700; }
    .side-kicker { color: #86cfd1; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; margin-bottom: 0.65rem; }
    .side-title { color: #fff; font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem; font-weight: 600; line-height: 1.15; margin-bottom: 0.75rem; }
    .side-copy { color: #b7cacc; font-size: 0.87rem; line-height: 1.55; }
    .side-status { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; padding: 0.85rem; margin-top: 1.5rem; }
    .side-status-label { color: #9cb8ba; font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .side-status-value { color: #fff; font-size: 0.95rem; font-weight: 600; margin-top: 0.3rem; }
    .side-footer { color: #7fa4a7; font-size: 0.72rem; line-height: 1.5; margin-top: 3rem; }
    .hero { align-items: end; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; gap: 2rem; padding: 0 0 2rem; }
    .hero h1 { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: clamp(2.4rem, 5vw, 4.4rem); letter-spacing: -0.04em; line-height: 0.98; margin: 0.5rem 0 0.8rem; }
    .hero p { color: var(--muted); font-size: 1.03rem; line-height: 1.5; margin: 0; max-width: 610px; }
    .hero-tag { border: 1px solid #c7dfe0; border-radius: 999px; color: var(--teal-dark); font-size: 0.78rem; font-weight: 600; padding: 0.65rem 0.9rem; white-space: nowrap; }
    .section-label { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 600; margin: 2rem 0 0.2rem; }
    .section-copy { color: var(--muted); font-size: 0.87rem; margin-bottom: 1rem; }
    .form-shell { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 1.25rem 1.5rem 1.5rem; }
    .field-group { color: var(--teal-dark); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; margin: 0.15rem 0 0.7rem; text-transform: uppercase; }
    [data-testid="stSlider"] label p { color: var(--ink); font-size: 0.86rem; font-weight: 600; }
    div[data-testid="stFormSubmitButton"] button { background: var(--teal); border: 0; border-radius: 8px; color: #fff; font-weight: 700; min-height: 2.8rem; }
    div[data-testid="stFormSubmitButton"] button:hover { background: var(--teal-dark); color: #fff; }
    .insight-panel { background: #e9f3f2; border-left: 4px solid var(--teal); border-radius: 0 12px 12px 0; margin-top: 2rem; padding: 1.25rem 1.35rem; }
    .insight-title { color: var(--teal-dark); font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700; margin-bottom: 0.45rem; }
    .insight-copy { color: #3b6266; font-size: 0.84rem; line-height: 1.55; }
    .result-card { border-radius: 14px; margin-top: 2rem; padding: 1.5rem; }
    .result-card.pass { background: #e6f4ed; border: 1px solid #a7d9c0; color: #165c45; }
    .result-card.fail { background: #fff0ed; border: 1px solid #e8b6a9; color: #873b31; }
    .result-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
    .result-value { font-family: 'Space Grotesk', sans-serif; font-size: 2.8rem; font-weight: 700; line-height: 1; margin: 0.45rem 0; }
    .result-note { font-size: 0.88rem; }
    .result-meta { background: rgba(255,255,255,0.55); border-radius: 8px; font-size: 0.82rem; margin-top: 1rem; padding: 0.7rem 0.8rem; }
    .metric-label { color: var(--muted); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; }
    .metric-value { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.65rem; font-weight: 700; margin-top: 0.25rem; }
    @media (max-width: 720px) { .block-container { padding: 2rem 1.1rem 3rem; } .hero { align-items: start; flex-direction: column; gap: 1rem; } .hero-tag { white-space: normal; } }
    </style>
    """,
    unsafe_allow_html=True,
)

model = joblib.load(MODEL_PATH)

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">S</div><div class="brand-name">Student Signal</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-kicker">Academic intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-title">A clearer view of student readiness.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-copy">A lightweight prediction workspace for exploring the signals behind a student outcome.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="side-status"><div class="side-status-label">Active model</div><div class="side-status-value">Logistic Regression</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="side-status"><div class="side-status-label">Training set</div><div class="side-status-value">25 labeled students</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="side-footer">For educational exploration only.<br>Predictions should support conversation, not replace it.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
        <div>
            <div class="eyebrow">Student success workspace</div>
            <h1>Pass Predictor</h1>
            <p>Turn everyday learning signals into a focused, explainable estimate of student readiness.</p>
        </div>
        <div class="hero-tag">5 signals &nbsp; / &nbsp; instant estimate</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Student profile</div>', unsafe_allow_html=True)
st.markdown('<div class="section-copy">Adjust the profile below to generate a model estimate.</div>', unsafe_allow_html=True)

with st.form("student_details"):
    st.markdown('<div class="form-shell">', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="field-group">Learning habits</div>', unsafe_allow_html=True)
        study_hours = st.slider("Study hours per day", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
        sleep_hours = st.slider("Average sleep hours", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
        assignments_completed = st.slider("Assignments completed (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    with right:
        st.markdown('<div class="field-group">Academic signals</div>', unsafe_allow_html=True)
        previous_marks = st.slider("Previous exam marks", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
        attendance = st.slider("Attendance percentage", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        st.markdown(
            '<div class="insight-panel"><div class="insight-title">A balanced signal set</div><div class="insight-copy">The estimate combines effort, consistency, past performance, and wellbeing signals instead of relying on a single score.</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div style="height: 0.7rem"></div>', unsafe_allow_html=True)
    submitted = st.form_submit_button("Generate student estimate", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    student_data = pd.DataFrame(
        [{
            "study_hours": study_hours,
            "previous_marks": previous_marks,
            "attendance": attendance,
            "assignments_completed": assignments_completed,
            "sleep_hours": sleep_hours,
        }]
    )
    prediction = int(model.predict(student_data)[0])
    result = "Pass" if prediction == 1 else "Fail"
    result_class = "pass" if prediction == 1 else "fail"
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(student_data)[0][prediction])
    else:
        confidence = 1.0
    confidence_percent = round(confidence * 100)
    result_message = (
        "The current profile shows a positive readiness signal."
        if prediction == 1
        else "The current profile may benefit from targeted academic support."
    )
    st.markdown(
        f'<div class="result-card {result_class}">'
        f'<div class="result-label">Predicted outcome</div>'
        f'<div class="result-value">{result}</div>'
        f'<div class="result-note">{result_message}</div>'
        f'<div class="result-meta">Model confidence: <strong>{confidence_percent}%</strong></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    metric_one, metric_two, metric_three = st.columns(3)
    with metric_one:
        st.markdown(f'<div class="metric-label">Study rhythm</div><div class="metric-value">{study_hours:g} hrs/day</div>', unsafe_allow_html=True)
    with metric_two:
        st.markdown(f'<div class="metric-label">Academic baseline</div><div class="metric-value">{previous_marks:g} marks</div>', unsafe_allow_html=True)
    with metric_three:
        st.markdown(f'<div class="metric-label">Attendance</div><div class="metric-value">{attendance:g}%</div>', unsafe_allow_html=True)
    st.caption(
        "This estimate is based on the five values entered above and is intended for educational exploration, not high-stakes decisions."
    )