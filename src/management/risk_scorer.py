"""Rule-based risk scoring for the Lung Disease Management System.

Provides :class:`RiskScorer` which maps a disease prediction and patient
metadata to a numeric risk score in ``[0, 100]`` and a categorical
:class:`~src.models.types.RiskTier`.

Typical usage::

    scorer = RiskScorer()
    result = scorer.score(
        disease_class=DiseaseClass.COPD,
        age=72.0,
        pack_years=35.0,
        bmi=28.0,
        confidence=0.85,
        uncertainty=0.25,
    )
    print(result.tier, result.score)
"""

from __future__ import annotations

from src.models.types import DiseaseClass, RiskResult, RiskTier

# ---------------------------------------------------------------------------
# Base score table by disease class
# ---------------------------------------------------------------------------

_BASE_SCORES: dict[DiseaseClass, float] = {
    DiseaseClass.COPD: 30.0,
    DiseaseClass.PNEUMONIA: 25.0,
    DiseaseClass.BRONCHIECTASIS: 25.0,
    DiseaseClass.ASTHMA: 20.0,
    DiseaseClass.URTI: 10.0,
    DiseaseClass.BRONCHIOLITIS: 15.0,
    DiseaseClass.HEALTHY: 0.0,
}


def _tier_from_score(score: float) -> RiskTier:
    """Map a numeric score to a :class:`RiskTier`.

    Args:
        score: Numeric risk score in ``[0, 100]``.

    Returns:
        ``LOW`` for 0–33, ``MEDIUM`` for 34–66, ``HIGH`` for 67–100.
    """
    if score <= 33.0:
        return RiskTier.LOW
    if score <= 66.0:
        return RiskTier.MEDIUM
    return RiskTier.HIGH


class RiskScorer:
    """Rule-based composite risk scorer for lung disease patients.

    Combines disease classification with patient risk factors to produce
    a numeric score in ``[0, 100]`` and a categorical risk tier.

    Scoring rules (additive, then clamped to ``[0, 100]``):

    **Base score by disease:**
        COPD→30, Pneumonia→25, Bronchiectasis→25, Asthma→20,
        URTI→10, Bronchiolitis→15, Healthy→0

    **Age adjustments:**
        +5 if age > 60, +10 if age > 75 (mutually exclusive; higher threshold wins)

    **Pack-years adjustments:**
        +5 if pack_years > 10, +10 if pack_years > 30

    **BMI adjustments:**
        +5 if bmi > 30 (obese) or bmi < 18.5 (underweight)

    **Confidence adjustments:**
        −5 if confidence > 0.9, +5 if confidence < 0.5

    **Uncertainty adjustments:**
        +5 if uncertainty > 0.2, +10 if uncertainty > 0.4

    **Tier thresholds:**
        0–33 → Low, 34–66 → Medium, 67–100 → High

    Example::

        scorer = RiskScorer()
        result = scorer.score(DiseaseClass.COPD, 72.0, 35.0, 28.0, 0.85, 0.25)
        assert result.tier == RiskTier.HIGH
    """

    def score(
        self,
        disease_class: DiseaseClass,
        age: float,
        pack_years: float,
        bmi: float,
        confidence: float,
        uncertainty: float,
    ) -> RiskResult:
        """Compute a rule-based risk score and tier for a patient encounter.

        Args:
            disease_class: Predicted :class:`~src.models.types.DiseaseClass`.
            age: Patient age in years (non-negative).
            pack_years: Cumulative smoking exposure in pack-years (non-negative).
            bmi: Body mass index in kg/m² (positive).
            confidence: Model confidence (max softmax probability) in ``[0, 1]``.
            uncertainty: MC Dropout uncertainty (mean per-class std) in ``[0, 1]``.

        Returns:
            :class:`~src.models.types.RiskResult` with ``tier`` and ``score``
            attributes where ``score`` is in ``[0, 100]``.
        """
        raw_score: float = _BASE_SCORES[disease_class]

        # Age adjustment (higher threshold wins; additive to lower tier)
        if age > 75.0:
            raw_score += 10.0
        elif age > 60.0:
            raw_score += 5.0

        # Pack-years adjustment (both thresholds additive)
        if pack_years > 30.0:
            raw_score += 10.0
        if pack_years > 10.0:
            raw_score += 5.0

        # BMI adjustment (obese or underweight)
        if bmi > 30.0 or bmi < 18.5:
            raw_score += 5.0

        # Confidence adjustment
        if confidence > 0.9:
            raw_score -= 5.0
        elif confidence < 0.5:
            raw_score += 5.0

        # Uncertainty adjustment (higher threshold wins; additive to lower tier)
        if uncertainty > 0.4:
            raw_score += 10.0
        if uncertainty > 0.2:
            raw_score += 5.0

        # Clamp to [0, 100]
        clamped = max(0.0, min(100.0, raw_score))

        return RiskResult(tier=_tier_from_score(clamped), score=clamped)
