"""Streamlit frontend for the Lung Disease Management System.

Provides a two-column clinician interface:
  - Left panel:  WAV upload + patient metadata form + submit button
  - Right panel: diagnosis badge, confidence bar, Altair probability chart,
                 risk metrics, recommendations list, PDF download button

Requirements: 14.1–14.5
"""
from __future__ import annotations

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Lung Disease Management System",
    page_icon="\U0001fac1",
    layout="wide",
)

st.title("\U0001fac1 Personalized Lung Disease Management System")
st.markdown(
    "Upload a respiratory auscultation audio recording and enter patient "
    "information to receive an AI-powered diagnosis and management plan."
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "report_bytes" not in st.session_state:
    st.session_state.report_bytes = None

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 2])

RECORDING_LOCATIONS = [
    "Trachea",
    "Anterior left",
    "Anterior right",
    "Posterior left",
    "Posterior right",
    "Lateral left",
    "Lateral right",
]

# ---------------------------------------------------------------------------
# Left panel — input form
# ---------------------------------------------------------------------------
with col_left:
    st.subheader("Patient Information")

    audio_file = st.file_uploader(
        "Upload WAV recording", type=["wav"], key="audio_file"
    )
    patient_name = st.text_input("Patient Name", key="patient_name")
    patient_id = st.text_input("Patient ID", key="patient_id")
    age = st.number_input(
        "Age (years)", min_value=0, max_value=120, value=50, key="age"
    )
    sex = st.selectbox("Sex", ["M", "F"], key="sex")
    bmi = st.number_input(
        "BMI (kg/m\u00b2)", min_value=10.0, max_value=60.0, value=25.0, key="bmi"
    )
    pack_years = st.number_input(
        "Smoking Pack-Years",
        min_value=0.0,
        max_value=200.0,
        value=0.0,
        key="pack_years",
    )
    recording_location = st.selectbox(
        "Recording Location", RECORDING_LOCATIONS, key="recording_location"
    )

    if st.button("Analyze Recording", key="submit"):
        if audio_file is None:
            st.error("Please upload a WAV file before analyzing.")
        else:
            with st.spinner("Running analysis\u2026"):
                try:
                    files = {
                        "audio_file": (
                            audio_file.name,
                            audio_file.getvalue(),
                            "audio/wav",
                        )
                    }
                    form_data = {
                        "age": str(age),
                        "sex": sex,
                        "bmi": str(bmi),
                        "smoking_pack_years": str(pack_years),
                        "recording_location": recording_location,
                    }
                    resp = requests.post(
                        f"{API_BASE_URL}/predict",
                        files=files,
                        data=form_data,
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        st.session_state.prediction = resp.json()

                        # Also fetch the PDF report
                        report_data = dict(form_data)
                        report_data["patient_name"] = patient_name or "Unknown"
                        report_data["patient_id"] = patient_id or "N/A"
                        audio_file.seek(0)
                        report_files = {
                            "audio_file": (
                                audio_file.name,
                                audio_file.getvalue(),
                                "audio/wav",
                            )
                        }
                        rep_resp = requests.post(
                            f"{API_BASE_URL}/report",
                            files=report_files,
                            data=report_data,
                            timeout=60,
                        )
                        st.session_state.report_bytes = (
                            rep_resp.content if rep_resp.status_code == 200 else None
                        )
                    else:
                        st.error(
                            f"Prediction failed (HTTP {resp.status_code}): {resp.text}"
                        )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to the API server. "
                        f"Ensure it is running at {API_BASE_URL}"
                    )

# ---------------------------------------------------------------------------
# Right panel — results
# ---------------------------------------------------------------------------
with col_right:
    if st.session_state.prediction is not None:
        pred = st.session_state.prediction
        st.subheader("Analysis Results")

        # Diagnosis badge + confidence metric
        col_diag, col_conf = st.columns(2)
        with col_diag:
            st.metric("Diagnosis", pred["disease_class"])
        with col_conf:
            st.metric("Confidence", f"{pred['confidence']:.1%}")

        st.progress(float(pred["confidence"]))
        st.caption(f"Model confidence: {pred['confidence']:.1%}")

        # Altair probability bar chart
        prob_df = pd.DataFrame(
            list(pred["probabilities"].items()),
            columns=["Disease Class", "Probability"],
        )
        highlight_condition = alt.condition(
            alt.datum["Disease Class"] == pred["disease_class"],
            alt.value("#e74c3c"),
            alt.value("#3498db"),
        )
        chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Probability:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    title="Probability",
                ),
                y=alt.Y("Disease Class:N", sort="-x", title="Disease Class"),
                color=highlight_condition,
                tooltip=["Disease Class", "Probability"],
            )
            .properties(title="Class Probability Distribution", height=220)
        )
        st.altair_chart(chart, use_container_width=True)

        # Risk metrics
        st.subheader("Risk Assessment")
        col_tier, col_score = st.columns(2)
        with col_tier:
            st.metric("Risk Tier", pred["risk_tier"])
        with col_score:
            st.metric("Risk Score", f"{pred['risk_score']:.1f} / 100")

        # Download button
        if st.session_state.report_bytes is not None:
            st.download_button(
                label="\U0001f4e5 Download Clinical Report",
                data=st.session_state.report_bytes,
                file_name="lung_disease_report.pdf",
                mime="application/pdf",
                key="download_report",
            )
        else:
            st.info("Report not available — the API server may not be running.")
    else:
        st.info(
            "Upload a WAV recording and click **Analyze Recording** to see results."
        )
