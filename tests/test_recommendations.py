"""Tests for src/management/recommendations.py.

Covers:
  - Property 23: Recommendation Coverage and Schema Validity
  - Property 24: COPD GOLD 2024 Citation
  - Property 25: Asthma GINA Citation
  - Property 26: Spirometry Referral Always Present
  - Unit tests: lookup completeness, field non-emptiness, spirometry last entry
"""

from __future__ import annotations

from itertools import product
from typing import List

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.management.recommendations import RecommendationEngine, SPIROMETRY_REFERRAL
from src.models.types import DiseaseClass, Recommendation, RiskTier


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ALL_PAIRS = list(product(DiseaseClass, RiskTier))  # 21 combinations
_engine = RecommendationEngine()


def _all_fields_non_empty(rec: Recommendation) -> bool:
    """Return True iff all four fields are non-empty strings."""
    return (
        bool(rec.icon)
        and bool(rec.text)
        and bool(rec.sub_text)
        and bool(rec.source)
    )


# ---------------------------------------------------------------------------
# Property 23: Recommendation Coverage and Schema Validity
# Feature: lung-disease-management, Property 23: Recommendation Coverage and Schema Validity
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    disease_class=st.sampled_from(list(DiseaseClass)),
    risk_tier=st.sampled_from(list(RiskTier)),
)
def test_recommendations_coverage_and_schema(
    disease_class: DiseaseClass,
    risk_tier: RiskTier,
) -> None:
    """All 21 (DiseaseClass × RiskTier) pairs must return a valid non-empty list.

    Every recommendation must have non-null, non-empty icon, text,
    sub_text, and source fields.

    **Validates: Requirements 10.1, 10.4**
    """
    recs = _engine.get_recommendations(disease_class, risk_tier)

    assert isinstance(recs, list), (
        f"get_recommendations must return a list, got {type(recs)}"
    )
    assert len(recs) > 0, (
        f"No recommendations returned for ({disease_class}, {risk_tier})"
    )

    for i, rec in enumerate(recs):
        assert isinstance(rec, Recommendation), (
            f"Entry {i} is not a Recommendation instance for ({disease_class}, {risk_tier})"
        )
        assert bool(rec.icon), (
            f"Entry {i} has empty icon for ({disease_class}, {risk_tier})"
        )
        assert bool(rec.text), (
            f"Entry {i} has empty text for ({disease_class}, {risk_tier})"
        )
        assert bool(rec.sub_text), (
            f"Entry {i} has empty sub_text for ({disease_class}, {risk_tier})"
        )
        assert bool(rec.source), (
            f"Entry {i} has empty source for ({disease_class}, {risk_tier})"
        )


# ---------------------------------------------------------------------------
# Property 24: COPD GOLD 2024 Citation
# Feature: lung-disease-management, Property 24: COPD GOLD 2024 Citation
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(risk_tier=st.sampled_from(list(RiskTier)))
def test_copd_gold_2024_citation(risk_tier: RiskTier) -> None:
    """COPD recommendations must include at least one source citing 'GOLD 2024'.

    **Validates: Requirements 10.2**
    """
    recs = _engine.get_recommendations(DiseaseClass.COPD, risk_tier)
    has_gold = any("GOLD 2024" in rec.source for rec in recs)
    assert has_gold, (
        f"No recommendation cites 'GOLD 2024' for COPD + {risk_tier}. "
        f"Sources: {[r.source for r in recs]}"
    )


# ---------------------------------------------------------------------------
# Property 25: Asthma GINA Citation
# Feature: lung-disease-management, Property 25: Asthma GINA Citation
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(risk_tier=st.sampled_from(list(RiskTier)))
def test_asthma_gina_citation(risk_tier: RiskTier) -> None:
    """Asthma recommendations must include at least one source citing 'GINA'.

    **Validates: Requirements 10.3**
    """
    recs = _engine.get_recommendations(DiseaseClass.ASTHMA, risk_tier)
    has_gina = any("GINA" in rec.source for rec in recs)
    assert has_gina, (
        f"No recommendation cites 'GINA' for Asthma + {risk_tier}. "
        f"Sources: {[r.source for r in recs]}"
    )


# ---------------------------------------------------------------------------
# Property 26: Spirometry Referral Always Present
# Feature: lung-disease-management, Property 26: Spirometry Referral Always Present
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    disease_class=st.sampled_from(list(DiseaseClass)),
    risk_tier=st.sampled_from(list(RiskTier)),
)
def test_spirometry_always_present(
    disease_class: DiseaseClass,
    risk_tier: RiskTier,
) -> None:
    """All (DiseaseClass × RiskTier) pairs must include a spirometry recommendation.

    At least one recommendation's text or sub_text must contain "spirometry"
    (case-insensitive).

    **Validates: Requirements 10.5**
    """
    recs = _engine.get_recommendations(disease_class, risk_tier)
    has_spirometry = any(
        "spirometry" in rec.text.lower() or "spirometry" in rec.sub_text.lower()
        for rec in recs
    )
    assert has_spirometry, (
        f"No spirometry recommendation found for ({disease_class}, {risk_tier})"
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_all_21_pairs_have_recommendations() -> None:
    """Exhaustive check: all 21 (DiseaseClass × RiskTier) pairs return lists.

    **Validates: Requirements 10.1**
    """
    for disease_class, risk_tier in _ALL_PAIRS:
        recs = _engine.get_recommendations(disease_class, risk_tier)
        assert len(recs) > 0, (
            f"Empty recommendation list for ({disease_class}, {risk_tier})"
        )


def test_all_fields_non_empty_exhaustive() -> None:
    """All recommendation fields must be non-empty across all 21 pairs.

    **Validates: Requirements 10.4**
    """
    for disease_class, risk_tier in _ALL_PAIRS:
        recs = _engine.get_recommendations(disease_class, risk_tier)
        for rec in recs:
            assert _all_fields_non_empty(rec), (
                f"Empty field in Recommendation(icon={rec.icon!r}, text={rec.text!r}, "
                f"sub_text={rec.sub_text!r}, source={rec.source!r}) "
                f"for ({disease_class}, {risk_tier})"
            )


def test_spirometry_referral_is_last_entry() -> None:
    """SPIROMETRY_REFERRAL should be the last entry in every recommendation list.

    **Validates: Requirements 10.5**
    """
    for disease_class, risk_tier in _ALL_PAIRS:
        recs = _engine.get_recommendations(disease_class, risk_tier)
        last = recs[-1]
        assert last.text == SPIROMETRY_REFERRAL.text, (
            f"Last recommendation is not spirometry referral for ({disease_class}, {risk_tier}): "
            f"got '{last.text}'"
        )


def test_recommendations_returns_copy() -> None:
    """get_recommendations must return a fresh list (mutations don't affect lookup).

    **Validates: Requirements 10.1**
    """
    recs1 = _engine.get_recommendations(DiseaseClass.COPD, RiskTier.HIGH)
    recs1.clear()  # mutate the returned list
    recs2 = _engine.get_recommendations(DiseaseClass.COPD, RiskTier.HIGH)
    assert len(recs2) > 0, (
        "Mutating the returned list must not affect subsequent calls"
    )


def test_copd_all_tiers_have_gold_citation() -> None:
    """COPD across all three tiers must cite GOLD 2024.

    **Validates: Requirements 10.2**
    """
    for tier in RiskTier:
        recs = _engine.get_recommendations(DiseaseClass.COPD, tier)
        assert any("GOLD 2024" in r.source for r in recs), (
            f"No GOLD 2024 citation for COPD + {tier}"
        )


def test_asthma_all_tiers_have_gina_citation() -> None:
    """Asthma across all three tiers must cite GINA.

    **Validates: Requirements 10.3**
    """
    for tier in RiskTier:
        recs = _engine.get_recommendations(DiseaseClass.ASTHMA, tier)
        assert any("GINA" in r.source for r in recs), (
            f"No GINA citation for Asthma + {tier}"
        )
