"""Tests for app/streamlit_app.py.

Validates that all required form fields, endpoints, and UI components
are present in the source code of the Streamlit frontend.

Requirements: 14.1, 14.2
"""
from __future__ import annotations

from pathlib import Path

_SOURCE = (Path(__file__).parent.parent / "app" / "streamlit_app.py").read_text(
    encoding="utf-8"
)


def test_all_required_form_fields_present() -> None:
    """All 8 required form input fields must appear in the Streamlit app source.

    **Validates: Requirements 14.1**
    """
    required_fields = [
        "patient_name",
        "patient_id",
        "age",
        "sex",
        "bmi",
        "pack_years",
        "recording_location",
        "audio_file",
    ]
    for field in required_fields:
        assert field in _SOURCE, (
            f"Required form field key '{field}' not found in streamlit_app.py"
        )


def test_submit_button_present() -> None:
    """A submit/analyze button must be defined in the app.

    **Validates: Requirements 14.2**
    """
    assert "st.button" in _SOURCE, "st.button not found in streamlit_app.py"


def test_predict_endpoint_referenced() -> None:
    """The app must call POST /predict.

    **Validates: Requirements 14.2**
    """
    assert "/predict" in _SOURCE, "/predict endpoint not referenced"


def test_report_endpoint_referenced() -> None:
    """The app must call POST /report.

    **Validates: Requirements 14.2**
    """
    assert "/report" in _SOURCE, "/report endpoint not referenced"


def test_download_button_present() -> None:
    """A download button for the PDF report must be present.

    **Validates: Requirements 14.4**
    """
    assert "download_button" in _SOURCE, "st.download_button not found"


def test_altair_chart_present() -> None:
    """An Altair chart for probability distribution must be present.

    **Validates: Requirements 14.3**
    """
    assert "altair_chart" in _SOURCE or "alt.Chart" in _SOURCE, (
        "Altair chart not found in streamlit_app.py"
    )


def test_risk_metrics_displayed() -> None:
    """Risk tier and risk score must be displayed.

    **Validates: Requirements 14.3**
    """
    assert "risk_tier" in _SOURCE or "Risk Tier" in _SOURCE, (
        "Risk Tier metric not found"
    )
    assert "risk_score" in _SOURCE or "Risk Score" in _SOURCE, (
        "Risk Score metric not found"
    )
