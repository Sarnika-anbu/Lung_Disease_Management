"""Tests for src/management/risk_scorer.py.

Covers:
  - Property 21: Risk Score Range and Tier Validity
  - Property 22: Risk Score Monotonicity (pack_years)
  - Unit tests: base score by disease, age adjustments, BMI adjustments,
    confidence/uncertainty effects, tier boundaries
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.management.risk_scorer import RiskScorer
from src.models.types import DiseaseClass, RiskResult, RiskTier


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _score(
    disease_class: DiseaseClass = DiseaseClass.COPD,
    age: float = 50.0,
    pack_years: float = 0.0,
    bmi: float = 24.0,
    confidence: float = 0.7,
    uncertainty: float = 0.1,
) -> RiskResult:
    """Convenience wrapper around RiskScorer.score with sensible defaults."""
    return RiskScorer().score(disease_class, age, pack_years, bmi, confidence, uncertainty)


# ---------------------------------------------------------------------------
# Property 21: Risk Score Range and Tier Validity
# Feature: lung-disease-management, Property 21: Risk Score Range and Tier Validity
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    disease_class=st.sampled_from(list(DiseaseClass)),
    age=st.floats(0, 120, allow_nan=False, allow_infinity=False),
    pack_years=st.floats(0, 100, allow_nan=False, allow_infinity=False),
    bmi=st.floats(10, 60, allow_nan=False, allow_infinity=False),
    confidence=st.floats(0, 1, allow_nan=False, allow_infinity=False),
    uncertainty=st.floats(0, 1, allow_nan=False, allow_infinity=False),
)
def test_risk_score_range_and_tier(
    disease_class: DiseaseClass,
    age: float,
    pack_years: float,
    bmi: float,
    confidence: float,
    uncertainty: float,
) -> None:
    """Score must be in [0, 100] and tier must be a valid RiskTier for all inputs.

    **Validates: Requirements 9.2**
    """
    result = RiskScorer().score(disease_class, age, pack_years, bmi, confidence, uncertainty)

    assert 0 <= result.score <= 100, (
        f"Score {result.score} out of [0, 100] for "
        f"disease={disease_class}, age={age}, pack_years={pack_years}, "
        f"bmi={bmi}, confidence={confidence}, uncertainty={uncertainty}"
    )
    assert result.tier in (RiskTier.LOW, RiskTier.MEDIUM, RiskTier.HIGH), (
        f"Invalid tier '{result.tier}'"
    )


# ---------------------------------------------------------------------------
# Property 22: Risk Score Monotonicity (pack_years)
# Feature: lung-disease-management, Property 22: Risk Score Monotonicity
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    disease_class=st.sampled_from(list(DiseaseClass)),
    age=st.floats(0, 120, allow_nan=False, allow_infinity=False),
    bmi=st.floats(10, 60, allow_nan=False, allow_infinity=False),
    confidence=st.floats(0, 1, allow_nan=False, allow_infinity=False),
    uncertainty=st.floats(0, 1, allow_nan=False, allow_infinity=False),
    pack_years_low=st.floats(0, 50, allow_nan=False, allow_infinity=False),
    pack_years_high=st.floats(0, 100, allow_nan=False, allow_infinity=False),
)
def test_risk_score_monotonicity_pack_years(
    disease_class: DiseaseClass,
    age: float,
    bmi: float,
    confidence: float,
    uncertainty: float,
    pack_years_low: float,
    pack_years_high: float,
) -> None:
    """Increasing pack_years must not decrease the risk score.

    Holds all other factors constant. Scores are compared on the same disease
    class with higher vs lower pack_years.

    **Validates: Requirements 9.3**
    """
    # We want pack_years_high >= pack_years_low; swap if needed
    py_low = min(pack_years_low, pack_years_high)
    py_high = max(pack_years_low, pack_years_high)

    result_low = RiskScorer().score(disease_class, age, py_low, bmi, confidence, uncertainty)
    result_high = RiskScorer().score(disease_class, age, py_high, bmi, confidence, uncertainty)

    assert result_high.score >= result_low.score - 1e-9, (
        f"Score decreased when pack_years increased from {py_low} to {py_high}: "
        f"{result_low.score} → {result_high.score}"
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestBaseScoreByDisease:
    """Base score correctness for each disease class."""

    def test_healthy_base_score(self) -> None:
        """Healthy patients with no risk factors should have a low score."""
        result = _score(DiseaseClass.HEALTHY, age=30.0, confidence=0.95, uncertainty=0.05)
        # 0 base + no adjustments (age<=60, confidence>0.9 gives -5) = -5 → clamped to 0
        assert result.score == 0.0
        assert result.tier == RiskTier.LOW

    def test_copd_base_score(self) -> None:
        """COPD with no extra risk factors should start at 30."""
        result = _score(DiseaseClass.COPD, age=50.0, pack_years=0.0, bmi=24.0,
                        confidence=0.7, uncertainty=0.1)
        # 30 (base) + 0 (age) + 0 (py) + 0 (bmi) + 0 (conf) + 0 (uncert) = 30
        assert result.score == 30.0
        assert result.tier == RiskTier.LOW

    def test_pneumonia_base_score(self) -> None:
        """Pneumonia with no extra risk factors starts at 25."""
        result = _score(DiseaseClass.PNEUMONIA, age=50.0, pack_years=0.0, bmi=24.0,
                        confidence=0.7, uncertainty=0.1)
        assert result.score == 25.0

    def test_asthma_base_score(self) -> None:
        """Asthma with no extra risk factors starts at 20."""
        result = _score(DiseaseClass.ASTHMA, age=50.0, pack_years=0.0, bmi=24.0,
                        confidence=0.7, uncertainty=0.1)
        assert result.score == 20.0

    def test_urti_base_score(self) -> None:
        """URTI with no extra risk factors starts at 10."""
        result = _score(DiseaseClass.URTI, age=50.0, pack_years=0.0, bmi=24.0,
                        confidence=0.7, uncertainty=0.1)
        assert result.score == 10.0

    def test_bronchiolitis_base_score(self) -> None:
        """Bronchiolitis with no extra risk factors starts at 15."""
        result = _score(DiseaseClass.BRONCHIOLITIS, age=50.0, pack_years=0.0, bmi=24.0,
                        confidence=0.7, uncertainty=0.1)
        assert result.score == 15.0

    def test_bronchiectasis_base_score(self) -> None:
        """Bronchiectasis with no extra risk factors starts at 25."""
        result = _score(DiseaseClass.BRONCHIECTASIS, age=50.0, pack_years=0.0, bmi=24.0,
                        confidence=0.7, uncertainty=0.1)
        assert result.score == 25.0


class TestAgeAdjustments:
    """Age-based scoring adjustments."""

    def test_age_under_60_no_bonus(self) -> None:
        """Age ≤ 60 adds no points."""
        r1 = _score(DiseaseClass.COPD, age=60.0)
        assert r1.score == 30.0  # base only

    def test_age_over_60_adds_5(self) -> None:
        """Age > 60 and ≤ 75 adds +5."""
        r1 = _score(DiseaseClass.COPD, age=65.0)
        assert r1.score == 35.0

    def test_age_over_75_adds_10(self) -> None:
        """Age > 75 adds +10 (not +15)."""
        r1 = _score(DiseaseClass.COPD, age=76.0)
        assert r1.score == 40.0


class TestPackYearsAdjustments:
    """Pack-years scoring adjustments."""

    def test_pack_years_under_10_no_bonus(self) -> None:
        """Pack_years ≤ 10 adds no points."""
        r = _score(DiseaseClass.COPD, pack_years=10.0)
        assert r.score == 30.0

    def test_pack_years_over_10_adds_5(self) -> None:
        """Pack_years > 10 adds +5."""
        r = _score(DiseaseClass.COPD, pack_years=15.0)
        assert r.score == 35.0

    def test_pack_years_over_30_adds_10_plus_5(self) -> None:
        """Pack_years > 30 adds +10 (and +5 for >10 also applies)."""
        r = _score(DiseaseClass.COPD, pack_years=35.0)
        # base 30 + 5 (>10) + 10 (>30) = 45
        assert r.score == 45.0


class TestBMIAdjustments:
    """BMI scoring adjustments."""

    def test_normal_bmi_no_bonus(self) -> None:
        """Normal BMI (18.5–30) adds no points."""
        r = _score(DiseaseClass.COPD, bmi=24.0)
        assert r.score == 30.0

    def test_obese_bmi_adds_5(self) -> None:
        """BMI > 30 (obese) adds +5."""
        r = _score(DiseaseClass.COPD, bmi=35.0)
        assert r.score == 35.0

    def test_underweight_bmi_adds_5(self) -> None:
        """BMI < 18.5 (underweight) adds +5."""
        r = _score(DiseaseClass.COPD, bmi=17.0)
        assert r.score == 35.0


class TestConfidenceUncertaintyAdjustments:
    """Confidence and uncertainty adjustments."""

    def test_high_confidence_reduces_score(self) -> None:
        """Confidence > 0.9 subtracts 5."""
        r = _score(DiseaseClass.COPD, confidence=0.95)
        assert r.score == 25.0  # 30 - 5

    def test_low_confidence_increases_score(self) -> None:
        """Confidence < 0.5 adds 5."""
        r = _score(DiseaseClass.COPD, confidence=0.3)
        assert r.score == 35.0  # 30 + 5

    def test_moderate_uncertainty_adds_5(self) -> None:
        """Uncertainty > 0.2 adds 5."""
        r = _score(DiseaseClass.COPD, uncertainty=0.25)
        assert r.score == 35.0  # 30 + 5

    def test_high_uncertainty_adds_10_plus_5(self) -> None:
        """Uncertainty > 0.4 adds +10 (and +5 for >0.2 also applies)."""
        r = _score(DiseaseClass.COPD, uncertainty=0.5)
        # 30 + 5 (>0.2) + 10 (>0.4) = 45
        assert r.score == 45.0


class TestTierBoundaries:
    """Risk tier boundary mapping."""

    def test_score_33_is_low(self) -> None:
        """Score of 33 maps to LOW tier."""
        from src.management.risk_scorer import _tier_from_score
        assert _tier_from_score(33.0) == RiskTier.LOW

    def test_score_34_is_medium(self) -> None:
        """Score of 34 maps to MEDIUM tier."""
        from src.management.risk_scorer import _tier_from_score
        assert _tier_from_score(34.0) == RiskTier.MEDIUM

    def test_score_66_is_medium(self) -> None:
        """Score of 66 maps to MEDIUM tier."""
        from src.management.risk_scorer import _tier_from_score
        assert _tier_from_score(66.0) == RiskTier.MEDIUM

    def test_score_67_is_high(self) -> None:
        """Score of 67 maps to HIGH tier."""
        from src.management.risk_scorer import _tier_from_score
        assert _tier_from_score(67.0) == RiskTier.HIGH

    def test_score_100_is_high(self) -> None:
        """Score of 100 maps to HIGH tier."""
        from src.management.risk_scorer import _tier_from_score
        assert _tier_from_score(100.0) == RiskTier.HIGH

    def test_score_clamped_at_zero(self) -> None:
        """Score cannot drop below 0."""
        # Healthy + high confidence (−5) → 0, clamped
        r = _score(DiseaseClass.HEALTHY, age=30.0, pack_years=0.0, bmi=24.0,
                   confidence=0.95, uncertainty=0.05)
        assert r.score == 0.0
        assert r.tier == RiskTier.LOW

    def test_score_clamped_at_100(self) -> None:
        """Score cannot exceed 100."""
        # COPD(30) + age>75(10) + pack_years>30(10+5) + obese(5) + low_conf(5)
        # + high_uncertainty(10+5) = 80 → still ≤ 100
        r = RiskScorer().score(
            DiseaseClass.COPD, age=80.0, pack_years=40.0, bmi=35.0,
            confidence=0.3, uncertainty=0.5
        )
        assert r.score <= 100.0
