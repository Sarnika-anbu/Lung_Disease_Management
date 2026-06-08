"""Shared enums and dataclasses for the Lung Disease Management System.

This module defines the canonical data types used across the prediction
pipeline, risk scoring, management recommendation, and report generation
components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DiseaseClass(str, Enum):
    """Seven-class lung disease taxonomy derived from the ICBHI schema."""

    COPD = "COPD"
    HEALTHY = "Healthy"
    URTI = "URTI"
    BRONCHIECTASIS = "Bronchiectasis"
    PNEUMONIA = "Pneumonia"
    BRONCHIOLITIS = "Bronchiolitis"
    ASTHMA = "Asthma"


class RiskTier(str, Enum):
    """Patient risk stratification tier.

    Score ranges:
        LOW    0–33
        MEDIUM 34–66
        HIGH   67–100
    """

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class SmokingStatus(str, Enum):
    """Patient smoking history category."""

    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"


class RecordingLocation(str, Enum):
    """Body location where the stethoscope was placed during audio capture."""

    TRACHEA = "Trachea"
    ANTERIOR_LEFT = "Anterior left"
    ANTERIOR_RIGHT = "Anterior right"
    POSTERIOR_LEFT = "Posterior left"
    POSTERIOR_RIGHT = "Posterior right"
    LATERAL_LEFT = "Lateral left"
    LATERAL_RIGHT = "Lateral right"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelPrediction:
    """Output of the lung disease classification model.

    Attributes:
        disease_class: Predicted disease class (argmax of mean probability vector).
        confidence: Maximum value of the mean probability vector (0.0–1.0).
        probabilities: Per-class mean softmax probabilities from MC Dropout.
        uncertainty: Per-class standard deviation from MC Dropout forward passes.
    """

    disease_class: DiseaseClass
    confidence: float
    probabilities: Dict[str, float]
    uncertainty: Dict[str, float]


@dataclass
class RiskResult:
    """Output of the rule-based risk scorer.

    Attributes:
        tier: Categorical risk level (Low / Medium / High).
        score: Numeric risk score in the range 0–100.
    """

    tier: RiskTier
    score: float


@dataclass
class ProgressionForecast:
    """Probabilistic disease trajectory forecast at three time horizons.

    Each attribute maps outcome labels to probabilities in [0.0, 1.0].
    Values are sourced from GOLD natural-history lookup tables.

    Attributes:
        month_3: Trajectory probabilities at the 3-month horizon.
        month_6: Trajectory probabilities at the 6-month horizon.
        month_12: Trajectory probabilities at the 12-month horizon.
    """

    month_3: Dict[str, float] = field(default_factory=dict)
    month_6: Dict[str, float] = field(default_factory=dict)
    month_12: Dict[str, float] = field(default_factory=dict)


@dataclass
class Recommendation:
    """A single evidence-based management recommendation.

    Attributes:
        icon: Unicode or emoji icon representing the recommendation category.
        text: Short primary recommendation text.
        sub_text: Longer explanatory sub-text or clinical rationale.
        source: Clinical guideline source (e.g., "GOLD 2024", "GINA").
    """

    icon: str
    text: str
    sub_text: str
    source: str


@dataclass
class EncounterData:
    """Aggregated per-encounter data passed to the report generator.

    Attributes:
        patient_name: Full name of the patient.
        patient_id: Unique patient identifier.
        age: Patient age in years.
        sex: Patient biological sex ("M" or "F").
        bmi: Body mass index (kg/m²).
        smoking_pack_years: Cumulative smoking exposure in pack-years.
        recording_location: Stethoscope placement location during audio capture.
        prediction: Model classification output including probabilities and uncertainty.
        risk_result: Risk tier and numeric score from the risk scorer.
        progression: Probabilistic disease trajectory forecast.
        recommendations: Ordered list of evidence-based management recommendations.
        spectrogram_base64: Base64-encoded PNG of the Grad-CAM annotated spectrogram.
        model_icbhi_score: ICBHI score of the deployed model on the test split.
        training_dataset: Human-readable name of the dataset used for training.
    """

    patient_name: str
    patient_id: str
    age: float
    sex: str
    bmi: float
    smoking_pack_years: float
    recording_location: str
    prediction: ModelPrediction
    risk_result: RiskResult
    progression: ProgressionForecast
    recommendations: List[Recommendation]
    spectrogram_base64: str
    model_icbhi_score: float
    training_dataset: str
