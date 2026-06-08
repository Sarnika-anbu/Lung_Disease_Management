"""Tests for src/management/progression.py.

Covers:
  - Property 27: Progression Forecast Schema and Value Range
  - Property 28: Progression Forecast Determinism
  - Unit tests: key presence, probability bounds, Healthy stability, all combinations
"""

from __future__ import annotations

from itertools import product

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.management.progression import ProgressionModule
from src.models.types import DiseaseClass, ProgressionForecast, RiskTier, SmokingStatus

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_ALL_INPUTS = list(product(DiseaseClass, RiskTier, SmokingStatus))  # 63 combinations
_EXPECTED_KEYS = {"stable", "mild_progression", "moderate_progression", "severe_progression"}
_module = ProgressionModule()


# ---------------------------------------------------------------------------
# Property 27: Progression Forecast Schema and Value Range
# Feature: lung-disease-management, Property 27: Progression Forecast Schema and Value Range
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    disease_class=st.sampled_from(list(DiseaseClass)),
    risk_tier=st.sampled_from(list(RiskTier)),
    smoking_status=st.sampled_from(list(SmokingStatus)),
)
def test_progression_forecast_schema_and_range(
    disease_class: DiseaseClass,
    risk_tier: RiskTier,
    smoking_status: SmokingStatus,
) -> None:
    """All inputs must return a ProgressionForecast with correct keys and value range.

    Forecast must contain exactly the keys month_3, month_6, month_12.
    Every probability value must be in [0.0, 1.0].

    **Validates: Requirements 11.2**
    """
    forecast = _module.get_trajectory(disease_class, risk_tier, smoking_status)

    assert isinstance(forecast, ProgressionForecast), (
        f"Expected ProgressionForecast, got {type(forecast)}"
    )

    # Check top-level attributes
    assert hasattr(forecast, "month_3"), "Forecast missing month_3"
    assert hasattr(forecast, "month_6"), "Forecast missing month_6"
    assert hasattr(forecast, "month_12"), "Forecast missing month_12"

    # Validate each horizon
    for horizon_name, horizon_dict in [
        ("month_3", forecast.month_3),
        ("month_6", forecast.month_6),
        ("month_12", forecast.month_12),
    ]:
        assert isinstance(horizon_dict, dict), (
            f"{horizon_name} must be a dict, got {type(horizon_dict)}"
        )
        for key, val in horizon_dict.items():
            assert isinstance(val, float), (
                f"{horizon_name}[{key!r}] must be a float, got {type(val)}"
            )
            assert 0.0 <= val <= 1.0, (
                f"{horizon_name}[{key!r}] = {val} is outside [0.0, 1.0] "
                f"for ({disease_class}, {risk_tier}, {smoking_status})"
            )


# ---------------------------------------------------------------------------
# Property 28: Progression Forecast Determinism
# Feature: lung-disease-management, Property 28: Progression Forecast Determinism
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    disease_class=st.sampled_from(list(DiseaseClass)),
    risk_tier=st.sampled_from(list(RiskTier)),
    smoking_status=st.sampled_from(list(SmokingStatus)),
)
def test_progression_forecast_determinism(
    disease_class: DiseaseClass,
    risk_tier: RiskTier,
    smoking_status: SmokingStatus,
) -> None:
    """Identical inputs must return identical results on repeated calls.

    **Validates: Requirements 11.3**
    """
    forecast1 = _module.get_trajectory(disease_class, risk_tier, smoking_status)
    forecast2 = _module.get_trajectory(disease_class, risk_tier, smoking_status)

    assert forecast1.month_3 == forecast2.month_3, (
        f"month_3 differs on repeated calls for ({disease_class}, {risk_tier}, {smoking_status})"
    )
    assert forecast1.month_6 == forecast2.month_6, (
        f"month_6 differs on repeated calls for ({disease_class}, {risk_tier}, {smoking_status})"
    )
    assert forecast1.month_12 == forecast2.month_12, (
        f"month_12 differs on repeated calls for ({disease_class}, {risk_tier}, {smoking_status})"
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_all_63_combinations_covered() -> None:
    """All 63 (DiseaseClass × RiskTier × SmokingStatus) combinations must be valid.

    **Validates: Requirements 11.1, 11.2**
    """
    for disease_class, risk_tier, smoking_status in _ALL_INPUTS:
        try:
            forecast = _module.get_trajectory(disease_class, risk_tier, smoking_status)
        except KeyError:
            pytest.fail(
                f"Missing lookup table entry for ({disease_class}, {risk_tier}, {smoking_status})"
            )
        assert forecast is not None


def test_forecast_has_expected_outcome_keys() -> None:
    """Each horizon dict must contain the four expected outcome keys.

    **Validates: Requirements 11.2**
    """
    forecast = _module.get_trajectory(
        DiseaseClass.COPD, RiskTier.HIGH, SmokingStatus.CURRENT
    )
    for horizon_dict in [forecast.month_3, forecast.month_6, forecast.month_12]:
        assert set(horizon_dict.keys()) == _EXPECTED_KEYS, (
            f"Expected keys {_EXPECTED_KEYS}, got {set(horizon_dict.keys())}"
        )


def test_healthy_never_smoker_high_stable_probability() -> None:
    """Healthy + Low + Never smoker should have high stable probability.

    **Validates: Requirements 11.2**
    """
    forecast = _module.get_trajectory(
        DiseaseClass.HEALTHY, RiskTier.LOW, SmokingStatus.NEVER
    )
    assert forecast.month_3["stable"] >= 0.9, (
        f"Expected stable ≥ 0.9 for Healthy/Low/Never, got {forecast.month_3['stable']}"
    )
    assert forecast.month_12["stable"] >= 0.9, (
        f"Expected stable ≥ 0.9 (12-month) for Healthy/Low/Never, "
        f"got {forecast.month_12['stable']}"
    )


def test_all_probabilities_in_range() -> None:
    """Every probability across all 63 combinations must be in [0.0, 1.0].

    **Validates: Requirements 11.2**
    """
    for disease_class, risk_tier, smoking_status in _ALL_INPUTS:
        forecast = _module.get_trajectory(disease_class, risk_tier, smoking_status)
        for horizon_name, horizon_dict in [
            ("month_3", forecast.month_3),
            ("month_6", forecast.month_6),
            ("month_12", forecast.month_12),
        ]:
            for key, val in horizon_dict.items():
                assert 0.0 <= val <= 1.0, (
                    f"({disease_class}, {risk_tier}, {smoking_status}) "
                    f"{horizon_name}[{key!r}] = {val} out of [0, 1]"
                )


def test_forecast_returns_copy() -> None:
    """Mutating the returned ProgressionForecast must not affect subsequent calls.

    **Validates: Requirements 11.3**
    """
    forecast1 = _module.get_trajectory(
        DiseaseClass.COPD, RiskTier.LOW, SmokingStatus.NEVER
    )
    original_stable = forecast1.month_3["stable"]
    forecast1.month_3["stable"] = -999.0  # mutate

    forecast2 = _module.get_trajectory(
        DiseaseClass.COPD, RiskTier.LOW, SmokingStatus.NEVER
    )
    assert forecast2.month_3["stable"] == original_stable, (
        "Mutation of returned forecast must not affect lookup table"
    )


def test_copd_high_current_smoker_forecast() -> None:
    """COPD + High risk + current smoker should have low stable probability.

    **Validates: Requirements 11.2**
    """
    forecast = _module.get_trajectory(
        DiseaseClass.COPD, RiskTier.HIGH, SmokingStatus.CURRENT
    )
    # Per the spec example: stable=0.3 at month_3
    assert forecast.month_3["stable"] <= 0.5, (
        f"Expected stable ≤ 0.5 for COPD/High/Current, got {forecast.month_3['stable']}"
    )
    assert forecast.month_3["mild_progression"] > 0.0, (
        f"Expected mild_progression > 0 for COPD/High/Current"
    )
