"""Disease progression forecasting for the Lung Disease Management System.

Provides :class:`ProgressionModule` which encodes GOLD natural-history lookup
tables mapping (DiseaseClass × RiskTier × SmokingStatus) to probabilistic
disease trajectory forecasts at 3, 6, and 12-month horizons.

No runtime model inference is performed — all values are static look-up tables.

Typical usage::

    module = ProgressionModule()
    forecast = module.get_trajectory(
        DiseaseClass.COPD, RiskTier.HIGH, SmokingStatus.CURRENT
    )
    print(forecast.month_3)  # {"stable": 0.3, "mild_progression": 0.4, ...}
"""

from __future__ import annotations

from typing import Dict, Tuple

from src.models.types import DiseaseClass, ProgressionForecast, RiskTier, SmokingStatus

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

_OutcomeDict = Dict[str, float]
_ForecastTuple = Tuple[_OutcomeDict, _OutcomeDict, _OutcomeDict]

# ---------------------------------------------------------------------------
# Lookup table
# Keys: (DiseaseClass, RiskTier, SmokingStatus)
# Values: (month_3_dict, month_6_dict, month_12_dict)
# Each dict has keys: stable, mild_progression, moderate_progression, severe_progression
# Values are in [0.0, 1.0]. Sum ≤ 1.0 per spec (not required to sum to exactly 1.0).
# ---------------------------------------------------------------------------

_TABLE: Dict[Tuple[DiseaseClass, RiskTier, SmokingStatus], _ForecastTuple] = {
    # =========================================================================
    # COPD
    # =========================================================================
    (DiseaseClass.COPD, RiskTier.LOW, SmokingStatus.NEVER): (
        {"stable": 0.70, "mild_progression": 0.20, "moderate_progression": 0.07, "severe_progression": 0.03},
        {"stable": 0.65, "mild_progression": 0.22, "moderate_progression": 0.09, "severe_progression": 0.04},
        {"stable": 0.58, "mild_progression": 0.25, "moderate_progression": 0.12, "severe_progression": 0.05},
    ),
    (DiseaseClass.COPD, RiskTier.LOW, SmokingStatus.FORMER): (
        {"stable": 0.65, "mild_progression": 0.22, "moderate_progression": 0.09, "severe_progression": 0.04},
        {"stable": 0.60, "mild_progression": 0.25, "moderate_progression": 0.10, "severe_progression": 0.05},
        {"stable": 0.52, "mild_progression": 0.28, "moderate_progression": 0.14, "severe_progression": 0.06},
    ),
    (DiseaseClass.COPD, RiskTier.LOW, SmokingStatus.CURRENT): (
        {"stable": 0.55, "mild_progression": 0.28, "moderate_progression": 0.12, "severe_progression": 0.05},
        {"stable": 0.48, "mild_progression": 0.30, "moderate_progression": 0.15, "severe_progression": 0.07},
        {"stable": 0.40, "mild_progression": 0.32, "moderate_progression": 0.20, "severe_progression": 0.08},
    ),
    (DiseaseClass.COPD, RiskTier.MEDIUM, SmokingStatus.NEVER): (
        {"stable": 0.55, "mild_progression": 0.28, "moderate_progression": 0.12, "severe_progression": 0.05},
        {"stable": 0.48, "mild_progression": 0.30, "moderate_progression": 0.15, "severe_progression": 0.07},
        {"stable": 0.40, "mild_progression": 0.33, "moderate_progression": 0.18, "severe_progression": 0.09},
    ),
    (DiseaseClass.COPD, RiskTier.MEDIUM, SmokingStatus.FORMER): (
        {"stable": 0.48, "mild_progression": 0.30, "moderate_progression": 0.15, "severe_progression": 0.07},
        {"stable": 0.42, "mild_progression": 0.32, "moderate_progression": 0.17, "severe_progression": 0.09},
        {"stable": 0.35, "mild_progression": 0.33, "moderate_progression": 0.22, "severe_progression": 0.10},
    ),
    (DiseaseClass.COPD, RiskTier.MEDIUM, SmokingStatus.CURRENT): (
        {"stable": 0.38, "mild_progression": 0.32, "moderate_progression": 0.20, "severe_progression": 0.10},
        {"stable": 0.32, "mild_progression": 0.33, "moderate_progression": 0.22, "severe_progression": 0.13},
        {"stable": 0.25, "mild_progression": 0.35, "moderate_progression": 0.25, "severe_progression": 0.15},
    ),
    (DiseaseClass.COPD, RiskTier.HIGH, SmokingStatus.NEVER): (
        {"stable": 0.40, "mild_progression": 0.30, "moderate_progression": 0.20, "severe_progression": 0.10},
        {"stable": 0.33, "mild_progression": 0.32, "moderate_progression": 0.22, "severe_progression": 0.13},
        {"stable": 0.25, "mild_progression": 0.33, "moderate_progression": 0.27, "severe_progression": 0.15},
    ),
    (DiseaseClass.COPD, RiskTier.HIGH, SmokingStatus.FORMER): (
        {"stable": 0.35, "mild_progression": 0.32, "moderate_progression": 0.22, "severe_progression": 0.11},
        {"stable": 0.28, "mild_progression": 0.33, "moderate_progression": 0.25, "severe_progression": 0.14},
        {"stable": 0.20, "mild_progression": 0.35, "moderate_progression": 0.28, "severe_progression": 0.17},
    ),
    (DiseaseClass.COPD, RiskTier.HIGH, SmokingStatus.CURRENT): (
        {"stable": 0.30, "mild_progression": 0.40, "moderate_progression": 0.20, "severe_progression": 0.10},
        {"stable": 0.22, "mild_progression": 0.38, "moderate_progression": 0.25, "severe_progression": 0.15},
        {"stable": 0.15, "mild_progression": 0.35, "moderate_progression": 0.30, "severe_progression": 0.20},
    ),
    # =========================================================================
    # Asthma
    # =========================================================================
    (DiseaseClass.ASTHMA, RiskTier.LOW, SmokingStatus.NEVER): (
        {"stable": 0.75, "mild_progression": 0.18, "moderate_progression": 0.05, "severe_progression": 0.02},
        {"stable": 0.72, "mild_progression": 0.20, "moderate_progression": 0.06, "severe_progression": 0.02},
        {"stable": 0.68, "mild_progression": 0.22, "moderate_progression": 0.07, "severe_progression": 0.03},
    ),
    (DiseaseClass.ASTHMA, RiskTier.LOW, SmokingStatus.FORMER): (
        {"stable": 0.70, "mild_progression": 0.20, "moderate_progression": 0.07, "severe_progression": 0.03},
        {"stable": 0.66, "mild_progression": 0.22, "moderate_progression": 0.09, "severe_progression": 0.03},
        {"stable": 0.60, "mild_progression": 0.25, "moderate_progression": 0.11, "severe_progression": 0.04},
    ),
    (DiseaseClass.ASTHMA, RiskTier.LOW, SmokingStatus.CURRENT): (
        {"stable": 0.60, "mild_progression": 0.25, "moderate_progression": 0.10, "severe_progression": 0.05},
        {"stable": 0.55, "mild_progression": 0.27, "moderate_progression": 0.13, "severe_progression": 0.05},
        {"stable": 0.48, "mild_progression": 0.30, "moderate_progression": 0.16, "severe_progression": 0.06},
    ),
    (DiseaseClass.ASTHMA, RiskTier.MEDIUM, SmokingStatus.NEVER): (
        {"stable": 0.60, "mild_progression": 0.25, "moderate_progression": 0.10, "severe_progression": 0.05},
        {"stable": 0.55, "mild_progression": 0.27, "moderate_progression": 0.13, "severe_progression": 0.05},
        {"stable": 0.48, "mild_progression": 0.30, "moderate_progression": 0.16, "severe_progression": 0.06},
    ),
    (DiseaseClass.ASTHMA, RiskTier.MEDIUM, SmokingStatus.FORMER): (
        {"stable": 0.55, "mild_progression": 0.27, "moderate_progression": 0.13, "severe_progression": 0.05},
        {"stable": 0.48, "mild_progression": 0.30, "moderate_progression": 0.15, "severe_progression": 0.07},
        {"stable": 0.40, "mild_progression": 0.33, "moderate_progression": 0.19, "severe_progression": 0.08},
    ),
    (DiseaseClass.ASTHMA, RiskTier.MEDIUM, SmokingStatus.CURRENT): (
        {"stable": 0.45, "mild_progression": 0.30, "moderate_progression": 0.17, "severe_progression": 0.08},
        {"stable": 0.38, "mild_progression": 0.32, "moderate_progression": 0.20, "severe_progression": 0.10},
        {"stable": 0.30, "mild_progression": 0.35, "moderate_progression": 0.23, "severe_progression": 0.12},
    ),
    (DiseaseClass.ASTHMA, RiskTier.HIGH, SmokingStatus.NEVER): (
        {"stable": 0.45, "mild_progression": 0.30, "moderate_progression": 0.17, "severe_progression": 0.08},
        {"stable": 0.38, "mild_progression": 0.32, "moderate_progression": 0.20, "severe_progression": 0.10},
        {"stable": 0.30, "mild_progression": 0.35, "moderate_progression": 0.23, "severe_progression": 0.12},
    ),
    (DiseaseClass.ASTHMA, RiskTier.HIGH, SmokingStatus.FORMER): (
        {"stable": 0.38, "mild_progression": 0.32, "moderate_progression": 0.20, "severe_progression": 0.10},
        {"stable": 0.32, "mild_progression": 0.33, "moderate_progression": 0.23, "severe_progression": 0.12},
        {"stable": 0.25, "mild_progression": 0.35, "moderate_progression": 0.27, "severe_progression": 0.13},
    ),
    (DiseaseClass.ASTHMA, RiskTier.HIGH, SmokingStatus.CURRENT): (
        {"stable": 0.30, "mild_progression": 0.35, "moderate_progression": 0.22, "severe_progression": 0.13},
        {"stable": 0.25, "mild_progression": 0.35, "moderate_progression": 0.25, "severe_progression": 0.15},
        {"stable": 0.18, "mild_progression": 0.37, "moderate_progression": 0.28, "severe_progression": 0.17},
    ),
    # =========================================================================
    # Bronchiectasis
    # =========================================================================
    (DiseaseClass.BRONCHIECTASIS, RiskTier.LOW, SmokingStatus.NEVER): (
        {"stable": 0.65, "mild_progression": 0.22, "moderate_progression": 0.10, "severe_progression": 0.03},
        {"stable": 0.60, "mild_progression": 0.25, "moderate_progression": 0.12, "severe_progression": 0.03},
        {"stable": 0.55, "mild_progression": 0.27, "moderate_progression": 0.14, "severe_progression": 0.04},
    ),
    (DiseaseClass.BRONCHIECTASIS, RiskTier.LOW, SmokingStatus.FORMER): (
        {"stable": 0.60, "mild_progression": 0.25, "moderate_progression": 0.11, "severe_progression": 0.04},
        {"stable": 0.55, "mild_progression": 0.27, "moderate_progression": 0.13, "severe_progression": 0.05},
        {"stable": 0.48, "mild_progression": 0.30, "moderate_progression": 0.17, "severe_progression": 0.05},
    ),
    (DiseaseClass.BRONCHIECTASIS, RiskTier.LOW, SmokingStatus.CURRENT): (
        {"stable": 0.50, "mild_progression": 0.28, "moderate_progression": 0.15, "severe_progression": 0.07},
        {"stable": 0.44, "mild_progression": 0.30, "moderate_progression": 0.18, "severe_progression": 0.08},
        {"stable": 0.38, "mild_progression": 0.32, "moderate_progression": 0.22, "severe_progression": 0.08},
    ),
    (DiseaseClass.BRONCHIECTASIS, RiskTier.MEDIUM, SmokingStatus.NEVER): (
        {"stable": 0.50, "mild_progression": 0.28, "moderate_progression": 0.15, "severe_progression": 0.07},
        {"stable": 0.44, "mild_progression": 0.30, "moderate_progression": 0.18, "severe_progression": 0.08},
        {"stable": 0.37, "mild_progression": 0.33, "moderate_progression": 0.22, "severe_progression": 0.08},
    ),
    (DiseaseClass.BRONCHIECTASIS, RiskTier.MEDIUM, SmokingStatus.FORMER): (
        {"stable": 0.45, "mild_progression": 0.30, "moderate_progression": 0.17, "severe_progression": 0.08},
        {"stable": 0.38, "mild_progression": 0.32, "moderate_progression": 0.20, "severe_progression": 0.10},
        {"stable": 0.32, "mild_progression": 0.33, "moderate_progression": 0.25, "severe_progression": 0.10},
    ),
    (DiseaseClass.BRONCHIECTASIS, RiskTier.MEDIUM, SmokingStatus.CURRENT): (
        {"stable": 0.38, "mild_progression": 0.32, "moderate_progression": 0.20, "severe_progression": 0.10},
        {"stable": 0.32, "mild_progression": 0.33, "moderate_progression": 0.23, "severe_progression": 0.12},
        {"stable": 0.25, "mild_progression": 0.35, "moderate_progression": 0.27, "severe_progression": 0.13},
    ),
    (DiseaseClass.BRONCHIECTASIS, RiskTier.HIGH, SmokingStatus.NEVER): (
        {"stable": 0.38, "mild_progression": 0.32, "moderate_progression": 0.20, "severe_progression": 0.10},
        {"stable": 0.32, "mild_progression": 0.33, "moderate_progression": 0.23, "severe_progression": 0.12},
        {"stable": 0.25, "mild_progression": 0.35, "moderate_progression": 0.28, "severe_progression": 0.12},
    ),
    (DiseaseClass.BRONCHIECTASIS, RiskTier.HIGH, SmokingStatus.FORMER): (
        {"stable": 0.33, "mild_progression": 0.33, "moderate_progression": 0.22, "severe_progression": 0.12},
        {"stable": 0.27, "mild_progression": 0.33, "moderate_progression": 0.25, "severe_progression": 0.15},
        {"stable": 0.20, "mild_progression": 0.35, "moderate_progression": 0.30, "severe_progression": 0.15},
    ),
    (DiseaseClass.BRONCHIECTASIS, RiskTier.HIGH, SmokingStatus.CURRENT): (
        {"stable": 0.28, "mild_progression": 0.35, "moderate_progression": 0.25, "severe_progression": 0.12},
        {"stable": 0.22, "mild_progression": 0.35, "moderate_progression": 0.28, "severe_progression": 0.15},
        {"stable": 0.15, "mild_progression": 0.37, "moderate_progression": 0.32, "severe_progression": 0.16},
    ),
    # =========================================================================
    # Pneumonia
    # =========================================================================
    (DiseaseClass.PNEUMONIA, RiskTier.LOW, SmokingStatus.NEVER): (
        {"stable": 0.80, "mild_progression": 0.15, "moderate_progression": 0.04, "severe_progression": 0.01},
        {"stable": 0.85, "mild_progression": 0.11, "moderate_progression": 0.03, "severe_progression": 0.01},
        {"stable": 0.90, "mild_progression": 0.07, "moderate_progression": 0.02, "severe_progression": 0.01},
    ),
    (DiseaseClass.PNEUMONIA, RiskTier.LOW, SmokingStatus.FORMER): (
        {"stable": 0.75, "mild_progression": 0.18, "moderate_progression": 0.05, "severe_progression": 0.02},
        {"stable": 0.80, "mild_progression": 0.14, "moderate_progression": 0.04, "severe_progression": 0.02},
        {"stable": 0.85, "mild_progression": 0.10, "moderate_progression": 0.03, "severe_progression": 0.02},
    ),
    (DiseaseClass.PNEUMONIA, RiskTier.LOW, SmokingStatus.CURRENT): (
        {"stable": 0.68, "mild_progression": 0.22, "moderate_progression": 0.07, "severe_progression": 0.03},
        {"stable": 0.73, "mild_progression": 0.18, "moderate_progression": 0.07, "severe_progression": 0.02},
        {"stable": 0.78, "mild_progression": 0.14, "moderate_progression": 0.06, "severe_progression": 0.02},
    ),
    (DiseaseClass.PNEUMONIA, RiskTier.MEDIUM, SmokingStatus.NEVER): (
        {"stable": 0.60, "mild_progression": 0.25, "moderate_progression": 0.10, "severe_progression": 0.05},
        {"stable": 0.68, "mild_progression": 0.20, "moderate_progression": 0.09, "severe_progression": 0.03},
        {"stable": 0.75, "mild_progression": 0.16, "moderate_progression": 0.07, "severe_progression": 0.02},
    ),
    (DiseaseClass.PNEUMONIA, RiskTier.MEDIUM, SmokingStatus.FORMER): (
        {"stable": 0.55, "mild_progression": 0.28, "moderate_progression": 0.12, "severe_progression": 0.05},
        {"stable": 0.62, "mild_progression": 0.22, "moderate_progression": 0.12, "severe_progression": 0.04},
        {"stable": 0.70, "mild_progression": 0.18, "moderate_progression": 0.09, "severe_progression": 0.03},
    ),
    (DiseaseClass.PNEUMONIA, RiskTier.MEDIUM, SmokingStatus.CURRENT): (
        {"stable": 0.48, "mild_progression": 0.30, "moderate_progression": 0.15, "severe_progression": 0.07},
        {"stable": 0.55, "mild_progression": 0.25, "moderate_progression": 0.14, "severe_progression": 0.06},
        {"stable": 0.63, "mild_progression": 0.22, "moderate_progression": 0.11, "severe_progression": 0.04},
    ),
    (DiseaseClass.PNEUMONIA, RiskTier.HIGH, SmokingStatus.NEVER): (
        {"stable": 0.40, "mild_progression": 0.32, "moderate_progression": 0.18, "severe_progression": 0.10},
        {"stable": 0.48, "mild_progression": 0.28, "moderate_progression": 0.17, "severe_progression": 0.07},
        {"stable": 0.58, "mild_progression": 0.23, "moderate_progression": 0.13, "severe_progression": 0.06},
    ),
    (DiseaseClass.PNEUMONIA, RiskTier.HIGH, SmokingStatus.FORMER): (
        {"stable": 0.35, "mild_progression": 0.33, "moderate_progression": 0.20, "severe_progression": 0.12},
        {"stable": 0.43, "mild_progression": 0.30, "moderate_progression": 0.18, "severe_progression": 0.09},
        {"stable": 0.52, "mild_progression": 0.25, "moderate_progression": 0.15, "severe_progression": 0.08},
    ),
    (DiseaseClass.PNEUMONIA, RiskTier.HIGH, SmokingStatus.CURRENT): (
        {"stable": 0.28, "mild_progression": 0.35, "moderate_progression": 0.25, "severe_progression": 0.12},
        {"stable": 0.36, "mild_progression": 0.32, "moderate_progression": 0.22, "severe_progression": 0.10},
        {"stable": 0.45, "mild_progression": 0.28, "moderate_progression": 0.18, "severe_progression": 0.09},
    ),
    # =========================================================================
    # URTI
    # =========================================================================
    (DiseaseClass.URTI, RiskTier.LOW, SmokingStatus.NEVER): (
        {"stable": 0.88, "mild_progression": 0.10, "moderate_progression": 0.02, "severe_progression": 0.00},
        {"stable": 0.93, "mild_progression": 0.06, "moderate_progression": 0.01, "severe_progression": 0.00},
        {"stable": 0.96, "mild_progression": 0.03, "moderate_progression": 0.01, "severe_progression": 0.00},
    ),
    (DiseaseClass.URTI, RiskTier.LOW, SmokingStatus.FORMER): (
        {"stable": 0.83, "mild_progression": 0.13, "moderate_progression": 0.03, "severe_progression": 0.01},
        {"stable": 0.89, "mild_progression": 0.08, "moderate_progression": 0.02, "severe_progression": 0.01},
        {"stable": 0.93, "mild_progression": 0.05, "moderate_progression": 0.01, "severe_progression": 0.01},
    ),
    (DiseaseClass.URTI, RiskTier.LOW, SmokingStatus.CURRENT): (
        {"stable": 0.76, "mild_progression": 0.17, "moderate_progression": 0.05, "severe_progression": 0.02},
        {"stable": 0.82, "mild_progression": 0.12, "moderate_progression": 0.04, "severe_progression": 0.02},
        {"stable": 0.87, "mild_progression": 0.09, "moderate_progression": 0.03, "severe_progression": 0.01},
    ),
    (DiseaseClass.URTI, RiskTier.MEDIUM, SmokingStatus.NEVER): (
        {"stable": 0.72, "mild_progression": 0.20, "moderate_progression": 0.06, "severe_progression": 0.02},
        {"stable": 0.78, "mild_progression": 0.15, "moderate_progression": 0.05, "severe_progression": 0.02},
        {"stable": 0.83, "mild_progression": 0.12, "moderate_progression": 0.04, "severe_progression": 0.01},
    ),
    (DiseaseClass.URTI, RiskTier.MEDIUM, SmokingStatus.FORMER): (
        {"stable": 0.67, "mild_progression": 0.22, "moderate_progression": 0.08, "severe_progression": 0.03},
        {"stable": 0.73, "mild_progression": 0.18, "moderate_progression": 0.07, "severe_progression": 0.02},
        {"stable": 0.78, "mild_progression": 0.14, "moderate_progression": 0.06, "severe_progression": 0.02},
    ),
    (DiseaseClass.URTI, RiskTier.MEDIUM, SmokingStatus.CURRENT): (
        {"stable": 0.60, "mild_progression": 0.25, "moderate_progression": 0.11, "severe_progression": 0.04},
        {"stable": 0.66, "mild_progression": 0.22, "moderate_progression": 0.09, "severe_progression": 0.03},
        {"stable": 0.72, "mild_progression": 0.18, "moderate_progression": 0.08, "severe_progression": 0.02},
    ),
    (DiseaseClass.URTI, RiskTier.HIGH, SmokingStatus.NEVER): (
        {"stable": 0.55, "mild_progression": 0.28, "moderate_progression": 0.12, "severe_progression": 0.05},
        {"stable": 0.62, "mild_progression": 0.23, "moderate_progression": 0.11, "severe_progression": 0.04},
        {"stable": 0.69, "mild_progression": 0.19, "moderate_progression": 0.09, "severe_progression": 0.03},
    ),
    (DiseaseClass.URTI, RiskTier.HIGH, SmokingStatus.FORMER): (
        {"stable": 0.50, "mild_progression": 0.30, "moderate_progression": 0.14, "severe_progression": 0.06},
        {"stable": 0.57, "mild_progression": 0.26, "moderate_progression": 0.13, "severe_progression": 0.04},
        {"stable": 0.64, "mild_progression": 0.21, "moderate_progression": 0.11, "severe_progression": 0.04},
    ),
    (DiseaseClass.URTI, RiskTier.HIGH, SmokingStatus.CURRENT): (
        {"stable": 0.43, "mild_progression": 0.32, "moderate_progression": 0.17, "severe_progression": 0.08},
        {"stable": 0.50, "mild_progression": 0.28, "moderate_progression": 0.16, "severe_progression": 0.06},
        {"stable": 0.58, "mild_progression": 0.24, "moderate_progression": 0.13, "severe_progression": 0.05},
    ),
    # =========================================================================
    # Bronchiolitis
    # =========================================================================
    (DiseaseClass.BRONCHIOLITIS, RiskTier.LOW, SmokingStatus.NEVER): (
        {"stable": 0.78, "mild_progression": 0.16, "moderate_progression": 0.05, "severe_progression": 0.01},
        {"stable": 0.83, "mild_progression": 0.12, "moderate_progression": 0.04, "severe_progression": 0.01},
        {"stable": 0.88, "mild_progression": 0.09, "moderate_progression": 0.02, "severe_progression": 0.01},
    ),
    (DiseaseClass.BRONCHIOLITIS, RiskTier.LOW, SmokingStatus.FORMER): (
        {"stable": 0.72, "mild_progression": 0.20, "moderate_progression": 0.06, "severe_progression": 0.02},
        {"stable": 0.78, "mild_progression": 0.16, "moderate_progression": 0.05, "severe_progression": 0.01},
        {"stable": 0.83, "mild_progression": 0.13, "moderate_progression": 0.03, "severe_progression": 0.01},
    ),
    (DiseaseClass.BRONCHIOLITIS, RiskTier.LOW, SmokingStatus.CURRENT): (
        {"stable": 0.65, "mild_progression": 0.24, "moderate_progression": 0.08, "severe_progression": 0.03},
        {"stable": 0.70, "mild_progression": 0.20, "moderate_progression": 0.08, "severe_progression": 0.02},
        {"stable": 0.75, "mild_progression": 0.17, "moderate_progression": 0.06, "severe_progression": 0.02},
    ),
    (DiseaseClass.BRONCHIOLITIS, RiskTier.MEDIUM, SmokingStatus.NEVER): (
        {"stable": 0.60, "mild_progression": 0.26, "moderate_progression": 0.10, "severe_progression": 0.04},
        {"stable": 0.66, "mild_progression": 0.22, "moderate_progression": 0.09, "severe_progression": 0.03},
        {"stable": 0.72, "mild_progression": 0.18, "moderate_progression": 0.08, "severe_progression": 0.02},
    ),
    (DiseaseClass.BRONCHIOLITIS, RiskTier.MEDIUM, SmokingStatus.FORMER): (
        {"stable": 0.55, "mild_progression": 0.28, "moderate_progression": 0.12, "severe_progression": 0.05},
        {"stable": 0.62, "mild_progression": 0.24, "moderate_progression": 0.11, "severe_progression": 0.03},
        {"stable": 0.68, "mild_progression": 0.21, "moderate_progression": 0.09, "severe_progression": 0.02},
    ),
    (DiseaseClass.BRONCHIOLITIS, RiskTier.MEDIUM, SmokingStatus.CURRENT): (
        {"stable": 0.48, "mild_progression": 0.30, "moderate_progression": 0.15, "severe_progression": 0.07},
        {"stable": 0.54, "mild_progression": 0.27, "moderate_progression": 0.14, "severe_progression": 0.05},
        {"stable": 0.60, "mild_progression": 0.23, "moderate_progression": 0.13, "severe_progression": 0.04},
    ),
    (DiseaseClass.BRONCHIOLITIS, RiskTier.HIGH, SmokingStatus.NEVER): (
        {"stable": 0.42, "mild_progression": 0.32, "moderate_progression": 0.18, "severe_progression": 0.08},
        {"stable": 0.48, "mild_progression": 0.28, "moderate_progression": 0.17, "severe_progression": 0.07},
        {"stable": 0.55, "mild_progression": 0.25, "moderate_progression": 0.15, "severe_progression": 0.05},
    ),
    (DiseaseClass.BRONCHIOLITIS, RiskTier.HIGH, SmokingStatus.FORMER): (
        {"stable": 0.37, "mild_progression": 0.33, "moderate_progression": 0.20, "severe_progression": 0.10},
        {"stable": 0.43, "mild_progression": 0.30, "moderate_progression": 0.19, "severe_progression": 0.08},
        {"stable": 0.50, "mild_progression": 0.27, "moderate_progression": 0.17, "severe_progression": 0.06},
    ),
    (DiseaseClass.BRONCHIOLITIS, RiskTier.HIGH, SmokingStatus.CURRENT): (
        {"stable": 0.32, "mild_progression": 0.35, "moderate_progression": 0.23, "severe_progression": 0.10},
        {"stable": 0.38, "mild_progression": 0.32, "moderate_progression": 0.22, "severe_progression": 0.08},
        {"stable": 0.45, "mild_progression": 0.28, "moderate_progression": 0.20, "severe_progression": 0.07},
    ),
    # =========================================================================
    # Healthy
    # =========================================================================
    (DiseaseClass.HEALTHY, RiskTier.LOW, SmokingStatus.NEVER): (
        {"stable": 0.96, "mild_progression": 0.03, "moderate_progression": 0.01, "severe_progression": 0.00},
        {"stable": 0.95, "mild_progression": 0.04, "moderate_progression": 0.01, "severe_progression": 0.00},
        {"stable": 0.93, "mild_progression": 0.05, "moderate_progression": 0.01, "severe_progression": 0.01},
    ),
    (DiseaseClass.HEALTHY, RiskTier.LOW, SmokingStatus.FORMER): (
        {"stable": 0.92, "mild_progression": 0.06, "moderate_progression": 0.01, "severe_progression": 0.01},
        {"stable": 0.90, "mild_progression": 0.07, "moderate_progression": 0.02, "severe_progression": 0.01},
        {"stable": 0.87, "mild_progression": 0.09, "moderate_progression": 0.03, "severe_progression": 0.01},
    ),
    (DiseaseClass.HEALTHY, RiskTier.LOW, SmokingStatus.CURRENT): (
        {"stable": 0.85, "mild_progression": 0.11, "moderate_progression": 0.03, "severe_progression": 0.01},
        {"stable": 0.82, "mild_progression": 0.13, "moderate_progression": 0.04, "severe_progression": 0.01},
        {"stable": 0.78, "mild_progression": 0.16, "moderate_progression": 0.05, "severe_progression": 0.01},
    ),
    (DiseaseClass.HEALTHY, RiskTier.MEDIUM, SmokingStatus.NEVER): (
        {"stable": 0.88, "mild_progression": 0.09, "moderate_progression": 0.02, "severe_progression": 0.01},
        {"stable": 0.85, "mild_progression": 0.11, "moderate_progression": 0.03, "severe_progression": 0.01},
        {"stable": 0.81, "mild_progression": 0.14, "moderate_progression": 0.04, "severe_progression": 0.01},
    ),
    (DiseaseClass.HEALTHY, RiskTier.MEDIUM, SmokingStatus.FORMER): (
        {"stable": 0.82, "mild_progression": 0.13, "moderate_progression": 0.04, "severe_progression": 0.01},
        {"stable": 0.78, "mild_progression": 0.16, "moderate_progression": 0.05, "severe_progression": 0.01},
        {"stable": 0.73, "mild_progression": 0.20, "moderate_progression": 0.06, "severe_progression": 0.01},
    ),
    (DiseaseClass.HEALTHY, RiskTier.MEDIUM, SmokingStatus.CURRENT): (
        {"stable": 0.73, "mild_progression": 0.18, "moderate_progression": 0.07, "severe_progression": 0.02},
        {"stable": 0.68, "mild_progression": 0.22, "moderate_progression": 0.08, "severe_progression": 0.02},
        {"stable": 0.62, "mild_progression": 0.26, "moderate_progression": 0.10, "severe_progression": 0.02},
    ),
    (DiseaseClass.HEALTHY, RiskTier.HIGH, SmokingStatus.NEVER): (
        {"stable": 0.78, "mild_progression": 0.16, "moderate_progression": 0.05, "severe_progression": 0.01},
        {"stable": 0.73, "mild_progression": 0.19, "moderate_progression": 0.06, "severe_progression": 0.02},
        {"stable": 0.67, "mild_progression": 0.23, "moderate_progression": 0.08, "severe_progression": 0.02},
    ),
    (DiseaseClass.HEALTHY, RiskTier.HIGH, SmokingStatus.FORMER): (
        {"stable": 0.70, "mild_progression": 0.20, "moderate_progression": 0.07, "severe_progression": 0.03},
        {"stable": 0.65, "mild_progression": 0.23, "moderate_progression": 0.09, "severe_progression": 0.03},
        {"stable": 0.58, "mild_progression": 0.27, "moderate_progression": 0.12, "severe_progression": 0.03},
    ),
    (DiseaseClass.HEALTHY, RiskTier.HIGH, SmokingStatus.CURRENT): (
        {"stable": 0.60, "mild_progression": 0.24, "moderate_progression": 0.12, "severe_progression": 0.04},
        {"stable": 0.55, "mild_progression": 0.27, "moderate_progression": 0.14, "severe_progression": 0.04},
        {"stable": 0.48, "mild_progression": 0.30, "moderate_progression": 0.18, "severe_progression": 0.04},
    ),
}


class ProgressionModule:
    """Static lookup-table disease trajectory forecaster.

    Encodes GOLD natural-history and clinical-guideline probability estimates
    for disease progression at 3, 6, and 12-month horizons.

    All 63 combinations of
    (7 DiseaseClass × 3 RiskTier × 3 SmokingStatus) are covered.

    No runtime model inference is performed; results come purely from the
    static lookup table.

    Example::

        module = ProgressionModule()
        forecast = module.get_trajectory(
            DiseaseClass.HEALTHY, RiskTier.LOW, SmokingStatus.NEVER
        )
        assert forecast.month_3["stable"] > 0.9
    """

    def get_trajectory(
        self,
        disease_class: DiseaseClass,
        risk_tier: RiskTier,
        smoking_status: SmokingStatus,
    ) -> ProgressionForecast:
        """Return probabilistic disease trajectory forecast for the given inputs.

        Args:
            disease_class: Predicted :class:`~src.models.types.DiseaseClass`.
            risk_tier: Patient :class:`~src.models.types.RiskTier`.
            smoking_status: Patient :class:`~src.models.types.SmokingStatus`.

        Returns:
            :class:`~src.models.types.ProgressionForecast` with ``month_3``,
            ``month_6``, and ``month_12`` dicts.  Each dict maps outcome labels
            (``"stable"``, ``"mild_progression"``, ``"moderate_progression"``,
            ``"severe_progression"``) to probabilities in ``[0.0, 1.0]``.

        Raises:
            KeyError: If the combination is not found in the lookup table
                (should never happen for valid enum values).
        """
        m3, m6, m12 = _TABLE[(disease_class, risk_tier, smoking_status)]
        return ProgressionForecast(
            month_3=dict(m3),
            month_6=dict(m6),
            month_12=dict(m12),
        )
