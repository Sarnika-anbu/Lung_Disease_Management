"""Pydantic schemas for the FastAPI backend.

Validation failures on any request body automatically return HTTP 422.
"""
from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field

from src.models.types import RecordingLocation


class PredictResponse(BaseModel):
    """Response schema for POST /predict."""

    disease_class: str
    confidence: float
    probabilities: Dict[str, float]
    uncertainty: Dict[str, float]
    risk_tier: str
    risk_score: float


class ExplainResponse(BaseModel):
    """Response schema for GET /explain/{recording_id}."""

    spectrogram_base64: str
    highlighted_regions: List[str]


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    model_version: str
    icbhi_score: float
