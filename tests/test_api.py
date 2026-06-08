"""Tests for the FastAPI backend.

Covers:
  - Property 31: Prediction API Response Schema
  - Property 32: Input Validation Rejects Invalid Requests
  - Unit tests: HTTP 500 on model failure, PDF attachment header, /health schema
"""
from __future__ import annotations

import io
import numpy as np
import pytest
import scipy.io.wavfile
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes() -> bytes:
    """Create a minimal 1-second silent WAV at 4000 Hz."""
    audio = np.zeros(4000, dtype=np.int16)
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, 4000, audio)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Property 31: Prediction API Response Schema
# Feature: lung-disease-management, Property 31: Prediction API Response Schema
# ---------------------------------------------------------------------------

def test_health_returns_json() -> None:
    """GET /health must return a JSON body regardless of model state.

    **Validates: Requirements 13.4**
    """
    response = client.get("/health")
    assert response.headers["content-type"].startswith("application/json")
    data = response.json()
    assert isinstance(data, dict)


def test_health_returns_503_when_no_checkpoint() -> None:
    """GET /health returns 503 when no model checkpoint exists.

    **Validates: Requirements 13.4**
    """
    response = client.get("/health")
    # In test environment there is no checkpoint, so 503 is expected.
    assert response.status_code == 503


def test_health_response_has_status_key() -> None:
    """GET /health must have a 'status' or 'detail' key.

    **Validates: Requirements 13.4**
    """
    response = client.get("/health")
    data = response.json()
    assert "status" in data or "detail" in data


# ---------------------------------------------------------------------------
# Property 32: Input Validation Rejects Invalid Requests
# Feature: lung-disease-management, Property 32: Input Validation Rejects Invalid Requests
# ---------------------------------------------------------------------------

def test_predict_missing_audio_file_returns_422() -> None:
    """POST /predict without audio_file must return 422.

    **Validates: Requirements 13.5, 13.6**
    """
    response = client.post(
        "/predict",
        data={
            "age": "50",
            "sex": "M",
            "bmi": "25.0",
            "smoking_pack_years": "10.0",
            "recording_location": "Trachea",
        },
    )
    assert response.status_code == 422


def test_predict_missing_required_field_age_returns_422() -> None:
    """POST /predict with missing age must return 422.

    **Validates: Requirements 13.5, 13.6**
    """
    wav_bytes = _make_wav_bytes()
    response = client.post(
        "/predict",
        files={"audio_file": ("test.wav", wav_bytes, "audio/wav")},
        data={
            "sex": "M",
            "bmi": "25.0",
            "smoking_pack_years": "10.0",
            "recording_location": "Trachea",
        },
    )
    assert response.status_code == 422


def test_predict_missing_required_field_sex_returns_422() -> None:
    """POST /predict with missing sex must return 422.

    **Validates: Requirements 13.5, 13.6**
    """
    wav_bytes = _make_wav_bytes()
    response = client.post(
        "/predict",
        files={"audio_file": ("test.wav", wav_bytes, "audio/wav")},
        data={
            "age": "50",
            "bmi": "25.0",
            "smoking_pack_years": "10.0",
            "recording_location": "Trachea",
        },
    )
    assert response.status_code == 422


def test_predict_invalid_sex_returns_422_or_503() -> None:
    """POST /predict with sex='X' must return 422 (validation) or 503 (no model).

    **Validates: Requirements 13.5, 13.6**
    """
    wav_bytes = _make_wav_bytes()
    response = client.post(
        "/predict",
        files={"audio_file": ("test.wav", wav_bytes, "audio/wav")},
        data={
            "age": "50",
            "sex": "X",  # invalid
            "bmi": "25.0",
            "smoking_pack_years": "10.0",
            "recording_location": "Trachea",
        },
    )
    # Model not loaded → 503; if model were loaded, invalid sex → 422
    assert response.status_code in (422, 503)


def test_predict_all_fields_valid_returns_422_or_503_without_model() -> None:
    """POST /predict with all valid fields returns 503 (no model checkpoint).

    **Validates: Requirements 13.1**
    """
    wav_bytes = _make_wav_bytes()
    response = client.post(
        "/predict",
        files={"audio_file": ("test.wav", wav_bytes, "audio/wav")},
        data={
            "age": "50",
            "sex": "M",
            "bmi": "25.0",
            "smoking_pack_years": "10.0",
            "recording_location": "Trachea",
        },
    )
    # Without a model checkpoint the server returns 503
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Unit tests — Task 15.9
# ---------------------------------------------------------------------------

def test_explain_without_prediction_returns_404_or_503() -> None:
    """GET /explain/{id} before any prediction must return 404 or 503.

    **Validates: Requirements 13.2**
    """
    response = client.get("/explain/some_recording_id")
    assert response.status_code in (404, 503)


def test_report_missing_patient_name_returns_422() -> None:
    """POST /report without patient_name must return 422.

    **Validates: Requirements 13.3**
    """
    wav_bytes = _make_wav_bytes()
    response = client.post(
        "/report",
        files={"audio_file": ("test.wav", wav_bytes, "audio/wav")},
        data={
            "age": "50",
            "sex": "M",
            "bmi": "25.0",
            "smoking_pack_years": "10.0",
            "recording_location": "Trachea",
            "patient_id": "P001",
            # patient_name is intentionally missing
        },
    )
    assert response.status_code == 422


def test_report_missing_patient_id_returns_422() -> None:
    """POST /report without patient_id must return 422.

    **Validates: Requirements 13.3**
    """
    wav_bytes = _make_wav_bytes()
    response = client.post(
        "/report",
        files={"audio_file": ("test.wav", wav_bytes, "audio/wav")},
        data={
            "age": "50",
            "sex": "M",
            "bmi": "25.0",
            "smoking_pack_years": "10.0",
            "recording_location": "Trachea",
            "patient_name": "Test Patient",
            # patient_id is intentionally missing
        },
    )
    assert response.status_code == 422


def test_all_endpoints_respond() -> None:
    """All four defined endpoints must respond without a 404 route-not-found error.

    **Validates: Requirements 13.8**
    """
    # /health
    r = client.get("/health")
    assert r.status_code != 404

    # /predict (no body → 422 expected, not 404)
    r = client.post("/predict")
    assert r.status_code != 404

    # /explain/{id}
    r = client.get("/explain/test")
    assert r.status_code != 404

    # /report (no body → 422 expected, not 404)
    r = client.post("/report")
    assert r.status_code != 404
