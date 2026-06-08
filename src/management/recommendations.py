"""Evidence-based management recommendations for the Lung Disease Management System.

Provides :class:`RecommendationEngine` which maps each
(DiseaseClass × RiskTier) combination to a list of structured
:class:`~src.models.types.Recommendation` objects sourced from
GOLD 2024, GINA, and NICE clinical guidelines.

Typical usage::

    engine = RecommendationEngine()
    recs = engine.get_recommendations(DiseaseClass.COPD, RiskTier.HIGH)
    for rec in recs:
        print(rec.icon, rec.text)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.models.types import DiseaseClass, Recommendation, RiskTier

# ---------------------------------------------------------------------------
# Shared recommendations
# ---------------------------------------------------------------------------

SPIROMETRY_REFERRAL = Recommendation(
    icon="🔬",
    text="Refer for spirometry to confirm diagnosis",
    sub_text=(
        "Spirometry is the gold standard for diagnosing and staging obstructive "
        "lung disease. FEV1/FVC ratio < 0.70 post-bronchodilator confirms airflow "
        "limitation consistent with COPD or asthma."
    ),
    source="NICE Guidelines",
)

_SMOKING_CESSATION = Recommendation(
    icon="🚭",
    text="Advise smoking cessation",
    sub_text=(
        "Smoking cessation is the single most effective intervention to slow disease "
        "progression in obstructive lung disease. Refer to a specialist cessation "
        "service and consider NRT or pharmacotherapy."
    ),
    source="NICE Guidelines",
)

_INFLUENZA_VACCINE = Recommendation(
    icon="💉",
    text="Ensure annual influenza vaccination",
    sub_text=(
        "Annual influenza vaccination reduces the risk of acute exacerbations and "
        "hospitalisations in patients with chronic lung disease."
    ),
    source="NICE Guidelines",
)

_PULMONARY_REHAB = Recommendation(
    icon="🏃",
    text="Refer for pulmonary rehabilitation",
    sub_text=(
        "Pulmonary rehabilitation improves exercise tolerance, quality of life, and "
        "reduces hospital admissions in moderate-to-severe COPD."
    ),
    source="GOLD 2024",
)

_SABA_RELIEF = Recommendation(
    icon="💊",
    text="Prescribe short-acting bronchodilator (SABA) for symptom relief",
    sub_text=(
        "A SABA (e.g., salbutamol) is the first-line reliever medication for acute "
        "bronchospasm in obstructive lung disease."
    ),
    source="GOLD 2024",
)

_LAMA_INITIATION = Recommendation(
    icon="💊",
    text="Initiate long-acting muscarinic antagonist (LAMA) therapy",
    sub_text=(
        "LAMA monotherapy is recommended as initial maintenance treatment for "
        "symptomatic COPD patients (CAT ≥ 10 or mMRC ≥ 2)."
    ),
    source="GOLD 2024",
)

_TRIPLE_THERAPY = Recommendation(
    icon="💊",
    text="Consider escalation to triple inhaler therapy (ICS/LABA/LAMA)",
    sub_text=(
        "Triple therapy (ICS + LABA + LAMA) is indicated for frequent exacerbators "
        "or persistently symptomatic patients with blood eosinophils ≥ 300 cells/μL."
    ),
    source="GOLD 2024",
)

_ICS_LABA = Recommendation(
    icon="💊",
    text="Prescribe ICS/LABA combination inhaler",
    sub_text=(
        "ICS/LABA combination inhalers reduce exacerbation frequency and improve "
        "lung function in COPD with eosinophilic inflammation."
    ),
    source="GOLD 2024",
)

_OXYGEN_ASSESSMENT = Recommendation(
    icon="🫁",
    text="Assess for long-term oxygen therapy (LTOT)",
    sub_text=(
        "LTOT is indicated if resting SaO2 ≤ 88% or PaO2 ≤ 7.3 kPa. Improves "
        "survival in hypoxaemic COPD."
    ),
    source="GOLD 2024",
)

_GINA_STEP1 = Recommendation(
    icon="💊",
    text="Initiate low-dose ICS-formoterol as reliever and controller (GINA Step 1–2)",
    sub_text=(
        "Low-dose ICS-formoterol is the preferred reliever and initial controller "
        "therapy for mild asthma, reducing exacerbations compared to SABA alone."
    ),
    source="GINA 2024",
)

_GINA_STEP3 = Recommendation(
    icon="💊",
    text="Escalate to medium-dose ICS-LABA (GINA Step 3–4)",
    sub_text=(
        "For uncontrolled asthma on low-dose ICS, step up to medium-dose ICS plus "
        "LABA. Review inhaler technique and adherence before escalation."
    ),
    source="GINA 2024",
)

_GINA_STEP5 = Recommendation(
    icon="🏥",
    text="Refer to respiratory specialist for severe/refractory asthma (GINA Step 5)",
    sub_text=(
        "Severe asthma should be assessed for add-on biologic therapy (anti-IL-5, "
        "anti-IL-4Rα, anti-IgE) by a specialist respiratory team."
    ),
    source="GINA 2024",
)

_ASTHMA_ACTION_PLAN = Recommendation(
    icon="📋",
    text="Provide written asthma action plan",
    sub_text=(
        "A personalised written action plan improves self-management and reduces "
        "emergency department visits in asthma patients."
    ),
    source="GINA 2024",
)

_CHEST_PHYSIO = Recommendation(
    icon="🫁",
    text="Refer for chest physiotherapy and airway clearance techniques",
    sub_text=(
        "Regular airway clearance (e.g., active cycle of breathing, oscillatory PEP) "
        "reduces exacerbations and improves quality of life in bronchiectasis."
    ),
    source="BTS Bronchiectasis Guideline 2019",
)

_SPUTUM_CULTURE = Recommendation(
    icon="🧫",
    text="Send sputum for culture and sensitivity",
    sub_text=(
        "Regular sputum microbiological surveillance guides antibiotic choice during "
        "exacerbations in bronchiectasis."
    ),
    source="BTS Bronchiectasis Guideline 2019",
)

_ANTIBIOTIC_REVIEW = Recommendation(
    icon="💊",
    text="Review antibiotic prophylaxis strategy",
    sub_text=(
        "Long-term macrolide prophylaxis (e.g., azithromycin) may be considered for "
        "patients with ≥3 exacerbations per year."
    ),
    source="BTS Bronchiectasis Guideline 2019",
)

_REST_HYDRATION = Recommendation(
    icon="💧",
    text="Advise rest, adequate hydration, and symptom monitoring",
    sub_text=(
        "Most upper respiratory tract infections are self-limiting. Advise rest, "
        "adequate fluid intake, and return if symptoms worsen or persist > 10 days."
    ),
    source="NICE NG120 Cough (Acute)",
)

_URTI_ANTIBIOTICS = Recommendation(
    icon="⚠️",
    text="Avoid routine antibiotic prescribing for URTI",
    sub_text=(
        "Antibiotics are not recommended for uncomplicated URTI. Consider delayed "
        "prescribing for high-risk patients or if bacterial superinfection is suspected."
    ),
    source="NICE NG120 Cough (Acute)",
)

_URTI_RED_FLAGS = Recommendation(
    icon="🚨",
    text="Educate on red-flag symptoms requiring urgent review",
    sub_text=(
        "Patients should seek urgent review if they develop high fever (>39°C), "
        "shortness of breath, chest pain, or symptoms lasting > 3 weeks."
    ),
    source="NICE NG120 Cough (Acute)",
)

_BRONCHIOLITIS_SUPPORTIVE = Recommendation(
    icon="💧",
    text="Supportive care: ensure adequate hydration and oxygenation",
    sub_text=(
        "Bronchiolitis management is primarily supportive. Monitor oxygen saturation "
        "and provide supplemental oxygen if SpO2 < 92%."
    ),
    source="NICE NG9 Bronchiolitis",
)

_BRONCHIOLITIS_ADMIT = Recommendation(
    icon="🏥",
    text="Consider hospital admission for moderate-to-severe bronchiolitis",
    sub_text=(
        "Admission criteria include SpO2 < 92%, respiratory rate > 70 breaths/min, "
        "or poor feeding (< 50% of normal intake)."
    ),
    source="NICE NG9 Bronchiolitis",
)

_PNEUMONIA_ANTIBIOTIC = Recommendation(
    icon="💊",
    text="Initiate antibiotic therapy per local CAP guidelines",
    sub_text=(
        "Community-acquired pneumonia should be treated promptly with antibiotics. "
        "Use CURB-65 score to guide severity assessment and admission decision."
    ),
    source="BTS CAP Guideline 2009 (updated 2023)",
)

_PNEUMONIA_ADMIT = Recommendation(
    icon="🏥",
    text="Consider hospital admission based on CURB-65 severity score",
    sub_text=(
        "CURB-65 score ≥ 2 indicates moderate-to-severe CAP requiring hospital "
        "admission. Score ≥ 3 suggests ICU assessment."
    ),
    source="BTS CAP Guideline 2009 (updated 2023)",
)

_PNEUMONIA_FOLLOW_UP = Recommendation(
    icon="📋",
    text="Arrange 6-week follow-up chest X-ray",
    sub_text=(
        "A follow-up chest radiograph at 6 weeks is recommended to confirm "
        "radiological resolution and exclude underlying malignancy."
    ),
    source="BTS CAP Guideline 2009 (updated 2023)",
)

_HEALTHY_LIFESTYLE = Recommendation(
    icon="🌱",
    text="Encourage healthy lifestyle and annual lung health review",
    sub_text=(
        "Advise on smoking cessation, regular physical activity, and maintain "
        "up-to-date vaccinations. Consider annual spirometry if at risk."
    ),
    source="NICE Preventive Care",
)

_HEALTHY_REASSURE = Recommendation(
    icon="✅",
    text="Reassure and provide respiratory health education",
    sub_text=(
        "Lung sounds are within normal limits. Educate the patient on triggers for "
        "common respiratory conditions and when to seek review."
    ),
    source="NICE Preventive Care",
)

# ---------------------------------------------------------------------------
# Lookup table: (DiseaseClass, RiskTier) → List[Recommendation]
# ---------------------------------------------------------------------------
# Every entry ends with SPIROMETRY_REFERRAL appended last.

_LOOKUP: Dict[Tuple[DiseaseClass, RiskTier], List[Recommendation]] = {
    # -----------------------------------------------------------------------
    # COPD
    # -----------------------------------------------------------------------
    (DiseaseClass.COPD, RiskTier.LOW): [
        _SABA_RELIEF,
        _SMOKING_CESSATION,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.COPD, RiskTier.MEDIUM): [
        _LAMA_INITIATION,
        _SABA_RELIEF,
        _SMOKING_CESSATION,
        _INFLUENZA_VACCINE,
        _PULMONARY_REHAB,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.COPD, RiskTier.HIGH): [
        _TRIPLE_THERAPY,
        _OXYGEN_ASSESSMENT,
        _PULMONARY_REHAB,
        _SMOKING_CESSATION,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    # -----------------------------------------------------------------------
    # Asthma
    # -----------------------------------------------------------------------
    (DiseaseClass.ASTHMA, RiskTier.LOW): [
        _GINA_STEP1,
        _ASTHMA_ACTION_PLAN,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.ASTHMA, RiskTier.MEDIUM): [
        _GINA_STEP3,
        _ASTHMA_ACTION_PLAN,
        _SMOKING_CESSATION,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.ASTHMA, RiskTier.HIGH): [
        _GINA_STEP5,
        _GINA_STEP3,
        _ASTHMA_ACTION_PLAN,
        _SMOKING_CESSATION,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    # -----------------------------------------------------------------------
    # Bronchiectasis
    # -----------------------------------------------------------------------
    (DiseaseClass.BRONCHIECTASIS, RiskTier.LOW): [
        _CHEST_PHYSIO,
        _SPUTUM_CULTURE,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.BRONCHIECTASIS, RiskTier.MEDIUM): [
        _CHEST_PHYSIO,
        _SPUTUM_CULTURE,
        _ANTIBIOTIC_REVIEW,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.BRONCHIECTASIS, RiskTier.HIGH): [
        _ANTIBIOTIC_REVIEW,
        _CHEST_PHYSIO,
        _SPUTUM_CULTURE,
        _PULMONARY_REHAB,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    # -----------------------------------------------------------------------
    # Pneumonia
    # -----------------------------------------------------------------------
    (DiseaseClass.PNEUMONIA, RiskTier.LOW): [
        _PNEUMONIA_ANTIBIOTIC,
        _REST_HYDRATION,
        _PNEUMONIA_FOLLOW_UP,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.PNEUMONIA, RiskTier.MEDIUM): [
        _PNEUMONIA_ANTIBIOTIC,
        _PNEUMONIA_ADMIT,
        _PNEUMONIA_FOLLOW_UP,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.PNEUMONIA, RiskTier.HIGH): [
        _PNEUMONIA_ADMIT,
        _PNEUMONIA_ANTIBIOTIC,
        _OXYGEN_ASSESSMENT,
        _PNEUMONIA_FOLLOW_UP,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    # -----------------------------------------------------------------------
    # URTI
    # -----------------------------------------------------------------------
    (DiseaseClass.URTI, RiskTier.LOW): [
        _REST_HYDRATION,
        _URTI_ANTIBIOTICS,
        _URTI_RED_FLAGS,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.URTI, RiskTier.MEDIUM): [
        _REST_HYDRATION,
        _URTI_ANTIBIOTICS,
        _URTI_RED_FLAGS,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.URTI, RiskTier.HIGH): [
        _URTI_RED_FLAGS,
        _REST_HYDRATION,
        _URTI_ANTIBIOTICS,
        _INFLUENZA_VACCINE,
        _PNEUMONIA_FOLLOW_UP,
        SPIROMETRY_REFERRAL,
    ],
    # -----------------------------------------------------------------------
    # Bronchiolitis
    # -----------------------------------------------------------------------
    (DiseaseClass.BRONCHIOLITIS, RiskTier.LOW): [
        _BRONCHIOLITIS_SUPPORTIVE,
        _REST_HYDRATION,
        _URTI_RED_FLAGS,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.BRONCHIOLITIS, RiskTier.MEDIUM): [
        _BRONCHIOLITIS_SUPPORTIVE,
        _BRONCHIOLITIS_ADMIT,
        _REST_HYDRATION,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.BRONCHIOLITIS, RiskTier.HIGH): [
        _BRONCHIOLITIS_ADMIT,
        _BRONCHIOLITIS_SUPPORTIVE,
        _OXYGEN_ASSESSMENT,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    # -----------------------------------------------------------------------
    # Healthy
    # -----------------------------------------------------------------------
    (DiseaseClass.HEALTHY, RiskTier.LOW): [
        _HEALTHY_REASSURE,
        _HEALTHY_LIFESTYLE,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.HEALTHY, RiskTier.MEDIUM): [
        _HEALTHY_REASSURE,
        _HEALTHY_LIFESTYLE,
        _SMOKING_CESSATION,
        _INFLUENZA_VACCINE,
        SPIROMETRY_REFERRAL,
    ],
    (DiseaseClass.HEALTHY, RiskTier.HIGH): [
        _HEALTHY_LIFESTYLE,
        _SMOKING_CESSATION,
        _INFLUENZA_VACCINE,
        _URTI_RED_FLAGS,
        SPIROMETRY_REFERRAL,
    ],
}


class RecommendationEngine:
    """Evidence-based management recommendation engine.

    Maps each (DiseaseClass × RiskTier) combination to an ordered list of
    :class:`~src.models.types.Recommendation` objects derived from GOLD 2024,
    GINA, NICE, and BTS clinical guidelines.

    Every returned list:
      - Contains at least one :class:`~src.models.types.Recommendation` with
        non-empty ``icon``, ``text``, ``sub_text``, and ``source`` fields.
      - Includes a spirometry referral recommendation as the final entry.
      - For COPD disease class: includes at least one source citing "GOLD 2024".
      - For Asthma disease class: includes at least one source citing "GINA".

    Example::

        engine = RecommendationEngine()
        recs = engine.get_recommendations(DiseaseClass.COPD, RiskTier.HIGH)
        assert any("GOLD 2024" in r.source for r in recs)
    """

    def get_recommendations(
        self,
        disease_class: DiseaseClass,
        risk_tier: RiskTier,
    ) -> List[Recommendation]:
        """Return ordered clinical recommendations for the given disease and tier.

        Args:
            disease_class: Predicted :class:`~src.models.types.DiseaseClass`.
            risk_tier: Patient :class:`~src.models.types.RiskTier`.

        Returns:
            Non-empty list of :class:`~src.models.types.Recommendation` objects.
            The last entry is always a spirometry referral.

        Raises:
            KeyError: If the (disease_class, risk_tier) combination is not found
                in the lookup table (should never happen for valid enum values).
        """
        return list(_LOOKUP[(disease_class, risk_tier)])
