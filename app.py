"""Streamlit app for predicting student exam scores."""

from __future__ import annotations

import json

import streamlit as st

from src.config import METRICS_PATH, PARENT_EDUCATION_LEVELS, PASS_THRESHOLD
from src.predict import load_model, predict_score

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("Student Performance Predictor")
st.write(
    "Enter a student's study and academic information to predict their exam score."
)


@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

metrics = None
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.caption(
        f"Model: {metrics['best_model'].replace('_', ' ')} · "
        f"Test MAE {metrics['test']['mae']:.2f} · "
        f"Test R² {metrics['test']['r2']:.3f}"
    )

with st.form("prediction_form"):
    hours_studied = st.slider("Hours studied per week", 1.0, 25.0, 10.0, 0.5)
    attendance_pct = st.slider("Attendance (%)", 40.0, 100.0, 85.0, 0.5)
    previous_grade = st.slider("Previous grade", 30.0, 100.0, 70.0, 0.5)
    sleep_hours = st.slider("Average sleep hours", 3.5, 10.0, 7.0, 0.5)
    extracurricular = st.selectbox(
        "Participates in extracurricular activities",
        options=[0, 1],
        format_func=lambda value: "Yes" if value == 1 else "No",
    )
    parent_education = st.selectbox("Parent education", PARENT_EDUCATION_LEVELS)
    submitted = st.form_submit_button("Predict score")

if submitted:
    result = predict_score(
        hours_studied,
        attendance_pct,
        previous_grade,
        sleep_hours,
        extracurricular,
        parent_education,
        model=model,
    )
    score = result["predicted_score"]
    outcome = result["predicted_outcome"]
    st.metric("Predicted exam score", f"{score}")
    if score >= PASS_THRESHOLD:
        st.success(f"Predicted outcome: {outcome} (threshold {PASS_THRESHOLD})")
    else:
        st.warning(f"Predicted outcome: {outcome} (threshold {PASS_THRESHOLD})")
