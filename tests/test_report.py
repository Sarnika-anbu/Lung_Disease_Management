"""Tests for src/management/report_generator.py.

Covers:
  - Property 29: PDF Report Structure Invariant
  - Property 30: Disclaimer Footer Content
  - Unit tests: bytes output, PDF magic bytes, invalid base64 handling
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.management.report_generator import ReportGenerator
from src.models.types import (
    DiseaseClass,
    EncounterData,
    ModelPrediction,
    ProgressionForecast,
    Recommendation,
    RiskResult,
    RiskTier,
    SmokingStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_png_b64() -> str:
    """Return a base64-encoded 1x1 white PNG (minimal valid PNG)."""
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (1, 1), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_encounter(
    icbhi_score: float = 0.85,
    dataset: str = "ICBHI 2017",
) -> EncounterData:
    """Build a minimal valid EncounterData for testing."""
    probs = {dc.value: 1.0 / 7 for dc in DiseaseClass}
    unc = {dc.value: 0.05 for dc in DiseaseClass}

    prediction = ModelPrediction(
        disease_class=DiseaseClass.COPD,
        confidence=0.72,
        probabilities=probs,
        uncertainty=unc,
    )
    risk_result = RiskResult(tier=RiskTier.MEDIUM, score=45.0)
    progression = ProgressionForecast(
        month_3={"stable": 0.5, "mild_progression": 0.3, "moderate_progression": 0.15, "severe_progression": 0.05},
        month_6={"stable": 0.45, "mild_progression": 0.32, "moderate_progression": 0.17, "severe_progression": 0.06},
        month_12={"stable": 0.4, "mild_progression": 0.35, "moderate_progression": 0.18, "severe_progression": 0.07},
    )
    recommendations = [
        Recommendation(icon="💊", text="Initiate bronchodilator therapy", sub_text="First-line treatment.", source="GOLD 2024"),
        Recommendation(icon="🔬", text="Refer for spirometry to confirm diagnosis", sub_text="Gold standard.", source="NICE Guidelines"),
    ]

    return EncounterData(
        patient_name="Test Patient",
        patient_id="TEST001",
        age=65.0,
        sex="M",
        bmi=27.5,
        smoking_pack_years=20.0,
        recording_location="Trachea",
        prediction=prediction,
        risk_result=risk_result,
        progression=progression,
        recommendations=recommendations,
        spectrogram_base64=_make_tiny_png_b64(),
        model_icbhi_score=icbhi_score,
        training_dataset=dataset,
    )


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        # Fallback: search raw bytes for ASCII section markers
        return pdf_bytes.decode("latin-1", errors="replace")


_SECTION_MARKERS = [
    "SECTION_1_PATIENT_HEADER",
    "SECTION_2_DIAGNOSIS",
    "SECTION_3_PROBABILITIES",
    "SECTION_4_GRADCAM",
    "SECTION_5_RISK_METRICS",
    "SECTION_6_PROGRESSION",
    "SECTION_7_RECOMMENDATIONS",
    "SECTION_8_MODEL_QUALITY",
    "SECTION_9_DISCLAIMER",
]


# ---------------------------------------------------------------------------
# Property 29: PDF Report Structure Invariant
# Feature: lung-disease-management, Property 29: PDF Report Structure Invariant
# ---------------------------------------------------------------------------

@settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.just(None))
def test_pdf_report_structure_invariant(dummy: None) -> None:
    """Generated PDF must contain all 9 section markers in the correct order.

    **Validates: Requirements 12.1**
    """
    encounter = _make_encounter()
    pdf_bytes = ReportGenerator().generate(encounter)
    text = _extract_pdf_text(pdf_bytes)

    # All 9 markers must be present
    for marker in _SECTION_MARKERS:
        assert marker in text, f"Missing section marker: {marker}"

    # Markers must appear in order
    positions = [text.find(m) for m in _SECTION_MARKERS]
    assert positions == sorted(positions), (
        f"Sections are not in order. Positions: {list(zip(_SECTION_MARKERS, positions))}"
    )


# ---------------------------------------------------------------------------
# Property 30: Disclaimer Footer Content
# Feature: lung-disease-management, Property 30: Disclaimer Footer Content
# ---------------------------------------------------------------------------

@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    icbhi_score=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    dataset_name=st.sampled_from(["ICBHI 2017", "Arashnic Lung Sounds", "ICBHI+Arashnic"]),
)
def test_disclaimer_footer_content(icbhi_score: float, dataset_name: str) -> None:
    """Disclaimer section must contain both the ICBHI score and training dataset name.

    **Validates: Requirements 12.3**
    """
    encounter = _make_encounter(icbhi_score=icbhi_score, dataset=dataset_name)
    pdf_bytes = ReportGenerator().generate(encounter)
    text = _extract_pdf_text(pdf_bytes)

    # Find disclaimer section
    disc_idx = text.find("SECTION_9_DISCLAIMER")
    assert disc_idx >= 0, "SECTION_9_DISCLAIMER marker not found in PDF"
    disclaimer_text = text[disc_idx:]

    # ICBHI score must appear
    score_str = f"{icbhi_score:.4f}"
    score_str2 = f"{icbhi_score:.2f}"
    assert score_str in disclaimer_text or score_str2 in disclaimer_text, (
        f"ICBHI score '{icbhi_score}' not found in disclaimer. "
        f"Checked '{score_str}' and '{score_str2}'. "
        f"Disclaimer text: {disclaimer_text[:300]}"
    )

    # Dataset name must appear
    assert dataset_name in disclaimer_text, (
        f"Training dataset name '{dataset_name}' not found in disclaimer."
    )


# ---------------------------------------------------------------------------
# Unit tests — Task 14.4
# ---------------------------------------------------------------------------

def test_generate_returns_bytes() -> None:
    """generate() must return non-empty bytes."""
    encounter = _make_encounter()
    result = ReportGenerator().generate(encounter)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_generate_returns_valid_pdf_magic_bytes() -> None:
    """Output must start with the PDF magic bytes '%PDF'."""
    encounter = _make_encounter()
    result = ReportGenerator().generate(encounter)
    assert result[:4] == b"%PDF", f"Not a PDF: starts with {result[:4]!r}"


def test_generate_with_invalid_base64_does_not_raise() -> None:
    """Invalid spectrogram_base64 must not raise — show placeholder instead."""
    encounter = _make_encounter()
    encounter.spectrogram_base64 = "not!!valid!!base64"
    result = ReportGenerator().generate(encounter)
    assert isinstance(result, bytes)
    assert result[:4] == b"%PDF"


def test_all_9_section_markers_present() -> None:
    """All 9 section markers must be present in the generated PDF."""
    encounter = _make_encounter()
    pdf_bytes = ReportGenerator().generate(encounter)
    text = _extract_pdf_text(pdf_bytes)
    for marker in _SECTION_MARKERS:
        assert marker in text, f"Missing: {marker}"


def test_disclaimer_contains_dataset_name() -> None:
    """The disclaimer must contain the training dataset name."""
    dataset = "ICBHI 2017 Respiratory Sound Database"
    encounter = _make_encounter(dataset=dataset)
    pdf_bytes = ReportGenerator().generate(encounter)
    text = _extract_pdf_text(pdf_bytes)
    assert dataset in text, f"Dataset name not found in PDF"


def test_disclaimer_contains_icbhi_score() -> None:
    """The disclaimer must contain the model ICBHI score."""
    score = 0.9123
    encounter = _make_encounter(icbhi_score=score)
    pdf_bytes = ReportGenerator().generate(encounter)
    text = _extract_pdf_text(pdf_bytes)
    assert f"{score:.4f}" in text, f"ICBHI score {score} not found in PDF"
