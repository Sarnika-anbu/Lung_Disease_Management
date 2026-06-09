"""Streamlit frontend for the Lung Disease Management System.

Full-featured clinician interface with:
  - Left panel:  WAV upload + mel spectrogram preview + patient metadata form
  - Right panel: diagnosis badge, confidence bar, Altair probability chart,
                 Grad-CAM annotated spectrogram, risk metrics, recommendations,
                 progression timeline, confusion matrix, PDF download

Requirements: 14.1–14.5
"""
from __future__ import annotations

import base64
import io
import os
import time

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Lung Disease Management System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.diagnosis-badge {
    font-size: 1.4rem;
    font-weight: bold;
    padding: 8px 16px;
    border-radius: 8px;
    display: inline-block;
    margin-bottom: 8px;
}
.risk-low    { background: #d4edda; color: #155724; }
.risk-medium { background: #fff3cd; color: #856404; }
.risk-high   { background: #f8d7da; color: #721c24; }
.section-header { border-bottom: 2px solid #dee2e6; padding-bottom: 4px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("🫁 Personalized Lung Disease Management System")
st.markdown("AI-powered respiratory sound analysis · EfficientNetV2B0 + CBAM + Metadata Fusion")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key in ["prediction", "report_bytes", "spectrogram_b64", "mel_spec_data",
            "recommendations", "progression"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_mel_spectrogram(wav_bytes: bytes) -> np.ndarray | None:
    """Compute a quick mel spectrogram for display (not the full pipeline)."""
    try:
        import librosa
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(wav_bytes))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
    except Exception:
        return None


def _tier_color(tier: str) -> str:
    return {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}.get(tier, "risk-medium")


def _fetch_recommendations(disease_class: str, risk_tier: str) -> list:
    """Fetch recommendations from API or compute locally."""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.management.recommendations import RecommendationEngine
        from src.models.types import DiseaseClass, RiskTier
        engine = RecommendationEngine()
        dc = DiseaseClass(disease_class)
        rt = RiskTier(risk_tier)
        recs = engine.get_recommendations(dc, rt)
        return [{"icon": r.icon, "text": r.text, "sub_text": r.sub_text, "source": r.source} for r in recs]
    except Exception:
        return []


def _fetch_progression(disease_class: str, risk_tier: str) -> dict | None:
    """Fetch progression forecast locally."""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.management.progression import ProgressionModule
        from src.models.types import DiseaseClass, RiskTier, SmokingStatus
        pm = ProgressionModule()
        forecast = pm.get_trajectory(
            DiseaseClass(disease_class), RiskTier(risk_tier), SmokingStatus.NEVER
        )
        return {
            "3-Month":  forecast.month_3,
            "6-Month":  forecast.month_6,
            "12-Month": forecast.month_12,
        }
    except Exception:
        return None


def _check_api() -> dict | None:
    """Check if API is available."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Sidebar — API status & model info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Status")
    health = _check_api()
    if health:
        st.success(f"✅ API Online")
        st.metric("Model Version", health.get("model_version", "N/A"))
        st.metric("ICBHI Score", f"{health.get('icbhi_score', 0):.4f}")
    else:
        st.error("❌ API Offline")
        st.info(f"Expected at: {API_BASE_URL}")

    st.markdown("---")
    st.markdown("**About**")
    st.markdown("- EfficientNetV2B0 + CBAM")
    st.markdown("- MC Dropout uncertainty")
    st.markdown("- Grad-CAM explainability")
    st.markdown("- ICBHI 2017 dataset")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 2])

RECORDING_LOCATIONS = [
    "Trachea", "Anterior left", "Anterior right",
    "Posterior left", "Posterior right", "Lateral left", "Lateral right",
]

# ---------------------------------------------------------------------------
# Left panel — input form
# ---------------------------------------------------------------------------
with col_left:
    st.subheader("📋 Patient Information")

    audio_file = st.file_uploader("Upload WAV recording", type=["wav"], key="audio_file")

    # Show mel spectrogram preview when file is uploaded
    if audio_file is not None:
        wav_bytes = audio_file.getvalue()
        mel = _compute_mel_spectrogram(wav_bytes)
        if mel is not None:
            st.markdown("**🎵 Mel Spectrogram Preview**")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
            ax.set_xlabel("Time frames")
            ax.set_ylabel("Mel bins")
            ax.set_title("Input Audio Spectrogram")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with st.expander("Patient Details", expanded=True):
        patient_name = st.text_input("Patient Name", key="patient_name")
        patient_id   = st.text_input("Patient ID", key="patient_id")
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.number_input("Age", min_value=0, max_value=120, value=50, key="age")
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, key="bmi")
        with col_b:
            sex = st.selectbox("Sex", ["M", "F"], key="sex")
            pack_years = st.number_input("Pack-Years", min_value=0.0, max_value=200.0, value=0.0, key="pack_years")
        recording_location = st.selectbox("Recording Location", RECORDING_LOCATIONS, key="recording_location")

    if st.button("🔬 Analyze Recording", key="submit", use_container_width=True, type="primary"):
        if audio_file is None:
            st.error("Please upload a WAV file first.")
        elif health is None:
            st.error(f"API server is not running at {API_BASE_URL}. Start it with: `uvicorn src.api.main:app --port 8000`")
        else:
            with st.spinner("Running AI analysis…"):
                try:
                    audio_file.seek(0)
                    files = {"audio_file": (audio_file.name, audio_file.getvalue(), "audio/wav")}
                    form_data = {
                        "age": str(age), "sex": sex, "bmi": str(bmi),
                        "smoking_pack_years": str(pack_years),
                        "recording_location": recording_location,
                    }

                    # POST /predict
                    resp = requests.post(f"{API_BASE_URL}/predict", files=files, data=form_data, timeout=60)
                    if resp.status_code == 200:
                        pred = resp.json()
                        st.session_state.prediction = pred

                        # Fetch Grad-CAM explanation
                        try:
                            explain_resp = requests.get(f"{API_BASE_URL}/explain/current", timeout=30)
                            if explain_resp.status_code == 200:
                                st.session_state.spectrogram_b64 = explain_resp.json().get("spectrogram_base64")
                        except Exception:
                            st.session_state.spectrogram_b64 = None

                        # Fetch recommendations and progression locally
                        st.session_state.recommendations = _fetch_recommendations(
                            pred["disease_class"], pred["risk_tier"]
                        )
                        st.session_state.progression = _fetch_progression(
                            pred["disease_class"], pred["risk_tier"]
                        )

                        # POST /report
                        report_data = dict(form_data)
                        report_data["patient_name"] = patient_name or "Unknown"
                        report_data["patient_id"]   = patient_id or "N/A"
                        audio_file.seek(0)
                        rep_resp = requests.post(
                            f"{API_BASE_URL}/report",
                            files={"audio_file": (audio_file.name, audio_file.getvalue(), "audio/wav")},
                            data=report_data,
                            timeout=90,
                        )
                        st.session_state.report_bytes = rep_resp.content if rep_resp.status_code == 200 else None
                        st.success("Analysis complete!")
                    else:
                        st.error(f"Prediction failed (HTTP {resp.status_code}): {resp.text[:300]}")
                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to API at {API_BASE_URL}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ---------------------------------------------------------------------------
# Right panel — results
# ---------------------------------------------------------------------------
with col_right:
    pred = st.session_state.prediction
    if pred is None:
        st.info("👆 Upload a WAV recording and click **Analyze Recording** to see results.")
        st.markdown("---")
        st.markdown("**Expected outputs:**")
        st.markdown("- 🏥 Disease classification (7 classes)")
        st.markdown("- 📊 Confidence + probability distribution")
        st.markdown("- 🔥 Grad-CAM annotated spectrogram")
        st.markdown("- ⚠️ Risk tier + score (0–100)")
        st.markdown("- 💊 Evidence-based recommendations")
        st.markdown("- 📈 Disease progression forecast")
        st.markdown("- 📄 Downloadable clinical PDF report")
    else:
        # ----------------------------------------------------------------
        # Section 1 — Diagnosis + confidence
        # ----------------------------------------------------------------
        st.markdown('<div class="section-header"><h3>🏥 Diagnosis</h3></div>', unsafe_allow_html=True)
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            tier = pred.get("risk_tier", "Medium")
            css_class = _tier_color(tier)
            st.markdown(
                f'<div class="diagnosis-badge {css_class}">{pred["disease_class"]}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Predicted disease")
        with col_d2:
            conf = float(pred["confidence"])
            st.metric("Confidence", f"{conf:.1%}")
            st.progress(conf)
        with col_d3:
            st.metric("Risk Tier", tier)
            st.metric("Risk Score", f"{pred.get('risk_score', 0):.1f} / 100")

        # ----------------------------------------------------------------
        # Section 2 — Probability distribution chart
        # ----------------------------------------------------------------
        st.markdown('<div class="section-header"><h3>📊 Class Probabilities</h3></div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame(
            list(pred["probabilities"].items()),
            columns=["Disease Class", "Probability"],
        ).sort_values("Probability", ascending=False)

        chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Disease Class:N", sort="-x"),
                color=alt.condition(
                    alt.datum["Disease Class"] == pred["disease_class"],
                    alt.value("#e74c3c"),
                    alt.value("#3498db"),
                ),
                tooltip=["Disease Class", alt.Tooltip("Probability:Q", format=".3f")],
            )
            .properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)

        # Uncertainty info
        if pred.get("uncertainty"):
            top_unc = max(pred["uncertainty"].values())
            st.caption(f"MC Dropout uncertainty (max): {top_unc:.4f}")

        # ----------------------------------------------------------------
        # Section 3 — Grad-CAM spectrogram
        # ----------------------------------------------------------------
        st.markdown('<div class="section-header"><h3>🔥 Grad-CAM Explanation</h3></div>', unsafe_allow_html=True)
        if st.session_state.spectrogram_b64:
            try:
                img_bytes = base64.b64decode(st.session_state.spectrogram_b64)
                st.image(img_bytes, caption="Grad-CAM highlighted spectrogram (red = high attention)", use_container_width=True)
            except Exception:
                st.info("Grad-CAM image unavailable.")
        else:
            st.info("Grad-CAM explanation not available (API /explain endpoint returned no data).")

        # ----------------------------------------------------------------
        # Section 4 — Recommendations
        # ----------------------------------------------------------------
        st.markdown('<div class="section-header"><h3>💊 Management Recommendations</h3></div>', unsafe_allow_html=True)
        recs = st.session_state.recommendations or []
        if recs:
            for rec in recs:
                with st.expander(f"{rec['icon']} {rec['text']}", expanded=False):
                    st.markdown(rec["sub_text"])
                    st.caption(f"Source: {rec['source']}")
        else:
            st.info("Recommendations not available.")

        # ----------------------------------------------------------------
        # Section 5 — Progression forecast
        # ----------------------------------------------------------------
        st.markdown('<div class="section-header"><h3>📈 Disease Progression Forecast</h3></div>', unsafe_allow_html=True)
        prog = st.session_state.progression
        if prog:
            prog_rows = []
            for horizon, values in prog.items():
                for outcome, prob in values.items():
                    prog_rows.append({
                        "Horizon": horizon,
                        "Outcome": outcome.replace("_", " ").title(),
                        "Probability": prob,
                    })
            prog_df = pd.DataFrame(prog_rows)

            prog_chart = (
                alt.Chart(prog_df)
                .mark_bar()
                .encode(
                    x=alt.X("Horizon:N", title="Time Horizon"),
                    y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Outcome:N"),
                    tooltip=["Horizon", "Outcome", alt.Tooltip("Probability:Q", format=".2f")],
                )
                .properties(height=200, title="Trajectory Probabilities")
            )
            st.altair_chart(prog_chart, use_container_width=True)
        else:
            st.info("Progression forecast not available.")

        # ----------------------------------------------------------------
        # Section 6 — Confusion matrix (from outputs/ folder)
        # ----------------------------------------------------------------
        st.markdown('<div class="section-header"><h3>📉 Model Evaluation</h3></div>', unsafe_allow_html=True)
        col_cm, col_rd = st.columns(2)
        with col_cm:
            cm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix (test set)", use_container_width=True)
            else:
                st.info("Run evaluation to generate confusion matrix: `python src/training/evaluate.py`")
        with col_rd:
            rd_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "reliability_diagram.png")
            if os.path.exists(rd_path):
                st.image(rd_path, caption="Reliability Diagram (ECE calibration)", use_container_width=True)
            else:
                st.info("Reliability diagram not yet generated.")

        # ----------------------------------------------------------------
        # Section 7 — Download PDF report
        # ----------------------------------------------------------------
        st.markdown('<div class="section-header"><h3>📄 Clinical Report</h3></div>', unsafe_allow_html=True)
        if st.session_state.report_bytes:
            st.download_button(
                label="📥 Download Clinical PDF Report",
                data=st.session_state.report_bytes,
                file_name=f"lung_report_{patient_id or 'patient'}.pdf",
                mime="application/pdf",
                key="download_report",
                use_container_width=True,
                type="primary",
            )
            st.caption("9-section clinical report including diagnosis, Grad-CAM, risk metrics, recommendations, and progression.")
        else:
            st.warning("PDF report could not be generated. Check that the API is running.")
