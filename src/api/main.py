"""FastAPI backend for the Lung Disease Management System.

Exposes four endpoints:
  - POST /predict   — audio + metadata → prediction + risk
  - GET  /explain/{recording_id} — Grad-CAM explanation of last prediction
  - POST /report    — full pipeline → downloadable PDF
  - GET  /health    — service health

Requirements: 13.1–13.8
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.api.schemas import ExplainResponse, HealthResponse, PredictResponse
from src.config import Config
from src.management.progression import ProgressionModule
from src.management.recommendations import RecommendationEngine
from src.management.risk_scorer import RiskScorer
from src.models.types import DiseaseClass, RecordingLocation, SmokingStatus

logger = logging.getLogger(__name__)
config = Config()

MODEL_VERSION = "1.0.0"
METADATA_DIM = 11  # age(1) + sex(1) + bmi(1) + pack_years(1) + location(7)

app = FastAPI(title="Lung Disease Management System", version=MODEL_VERSION)

# ---------------------------------------------------------------------------
# Application-level singletons (populated at startup)
# ---------------------------------------------------------------------------
_model = None
_explainer = None
_model_loaded: bool = False
_icbhi_score: float = 0.0
_last_spectrogram: Optional[torch.Tensor] = None


@app.on_event("startup")
async def startup_event() -> None:
    """Load model checkpoint and instantiate service components at startup."""
    global _model, _explainer, _model_loaded, _icbhi_score
    checkpoint_path = config.checkpoints_dir / "best.pth"
    if checkpoint_path.exists():
        try:
            from src.explainability.explainer import GradCAMExplainer
            from src.models.model import LungDiseaseModel

            checkpoint = torch.load(
                str(checkpoint_path), map_location="cpu", weights_only=False
            )
            _model = LungDiseaseModel(metadata_input_dim=METADATA_DIM)
            _model.load_state_dict(checkpoint["model_state_dict"])
            _model.eval()
            _icbhi_score = float(checkpoint.get("score", 0.0))
            _model_loaded = True
            logger.info(
                "Model loaded from %s (ICBHI=%.4f)", checkpoint_path, _icbhi_score
            )
            # Explainer is optional — don't fail startup if it errors
            try:
                _explainer = GradCAMExplainer(_model, config)
                logger.info("GradCAM explainer ready")
            except Exception as exc_exp:  # noqa: BLE001
                logger.warning("GradCAM explainer failed to load: %s", exc_exp)
                _explainer = None
        except Exception as exc:  # noqa: BLE001
            logger.critical("Failed to load model: %s", exc)
            _model_loaded = False
    else:
        logger.warning(
            "Model checkpoint not found at %s — starting without model",
            checkpoint_path,
        )
        _model_loaded = False


# ---------------------------------------------------------------------------
# Helper: metadata encoding
# ---------------------------------------------------------------------------

def _build_metadata_tensor(
    age: float,
    sex: str,
    bmi: float,
    pack_years: float,
    recording_location: str,
) -> torch.Tensor:
    """Encode patient metadata into an 11-dim float32 tensor.

    Encoding matches :class:`~src.training.train.LungSoundDataset`:
      - age / 100
      - sex binary (M=1, F=0)
      - (bmi - 10) / 40
      - pack_years / 100
      - recording_location as 7-dim one-hot

    Args:
        age: Patient age in years.
        sex: "M" or "F".
        bmi: Body mass index.
        pack_years: Cumulative smoking exposure.
        recording_location: Stethoscope placement string.

    Returns:
        Float32 tensor of shape ``(1, 11)``.
    """
    features: List[float] = []
    features.append(max(0.0, min(1.0, age / 100.0)))
    features.append(1.0 if str(sex).strip().upper() == "M" else 0.0)
    features.append(max(0.0, min(1.0, (bmi - 10.0) / 40.0)))
    features.append(max(0.0, min(1.0, pack_years / 100.0)))
    locations = [loc.value for loc in RecordingLocation]
    loc_onehot = [0.0] * 7
    if recording_location in locations:
        loc_onehot[locations.index(recording_location)] = 1.0
    features.extend(loc_onehot)
    return torch.tensor([features], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health and model information.

    Returns:
        :class:`~src.api.schemas.HealthResponse` JSON.

    Raises:
        HTTPException: HTTP 503 if the model checkpoint was not loaded.

    Requirements: 13.4
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "model_version": MODEL_VERSION,
                "icbhi_score": 0.0,
            },
        )
    return HealthResponse(
        status="ok", model_version=MODEL_VERSION, icbhi_score=_icbhi_score
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    audio_file: UploadFile = File(...),
    age: float = Form(...),
    sex: str = Form(...),
    bmi: float = Form(...),
    smoking_pack_years: float = Form(...),
    recording_location: str = Form(...),
) -> PredictResponse:
    """Run inference on an uploaded WAV recording.

    Args:
        audio_file: WAV audio file (multipart upload).
        age: Patient age in years.
        sex: Biological sex — "M" or "F".
        bmi: Body mass index.
        smoking_pack_years: Cumulative smoking exposure.
        recording_location: Stethoscope placement location.

    Returns:
        :class:`~src.api.schemas.PredictResponse` JSON.

    Raises:
        HTTPException: 422 on invalid inputs, 500 on internal errors,
            503 when the model is not loaded.

    Requirements: 13.1
    """
    global _last_spectrogram

    if not _model_loaded or _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate sex
    if sex not in ("M", "F"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid sex '{sex}'. Must be 'M' or 'F'.",
        )

    # Validate recording location
    valid_locations = [loc.value for loc in RecordingLocation]
    if recording_location not in valid_locations:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid recording_location '{recording_location}'. "
                f"Must be one of {valid_locations}."
            ),
        )

    # Preprocess audio
    try:
        from src.data.preprocess import preprocess_audio

        suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await audio_file.read())
            tmp_path = Path(tmp.name)
        spec_np = preprocess_audio(tmp_path, config)
        tmp_path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Internal model error: {exc}"
        )

    # Inference
    try:
        spec_tensor = torch.from_numpy(spec_np).unsqueeze(0).float()
        meta_tensor = _build_metadata_tensor(
            age, sex, bmi, smoking_pack_years, recording_location
        )

        _last_spectrogram = spec_tensor.clone()

        mean, std = _model.predict_with_uncertainty(
            spec_tensor, meta_tensor, n_passes=20
        )
        mean_np = mean.detach().cpu().numpy()
        std_np = std.detach().cpu().numpy()

        class_names = [dc.value for dc in DiseaseClass]
        pred_idx = int(np.argmax(mean_np))
        pred_class = DiseaseClass(class_names[pred_idx])
        confidence = float(mean_np[pred_idx])

        probabilities = {
            class_names[i]: float(mean_np[i]) for i in range(len(class_names))
        }
        uncertainty = {
            class_names[i]: float(std_np[i]) for i in range(len(class_names))
        }

        risk_result = RiskScorer().score(
            pred_class,
            age,
            smoking_pack_years,
            bmi,
            confidence,
            float(std_np.mean()),
        )

        return PredictResponse(
            disease_class=pred_class.value,
            confidence=confidence,
            probabilities=probabilities,
            uncertainty=uncertainty,
            risk_tier=risk_result.tier.value,
            risk_score=risk_result.score,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Internal model error: {exc}"
        )


@app.get("/explain/{recording_id}", response_model=ExplainResponse)
async def explain(recording_id: str) -> ExplainResponse:
    """Return Grad-CAM explanation for the most recent prediction.

    Args:
        recording_id: Identifier for the recording (informational; the most
            recent cached spectrogram is always used).

    Returns:
        :class:`~src.api.schemas.ExplainResponse` JSON.

    Raises:
        HTTPException: 404 if no prior prediction exists, 503 if no model.

    Requirements: 13.2
    """
    if not _model_loaded or _explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _last_spectrogram is None:
        raise HTTPException(
            status_code=404, detail="No prediction has been made yet"
        )
    try:
        b64_png = _explainer.explain(_last_spectrogram)
        return ExplainResponse(
            spectrogram_base64=b64_png, highlighted_regions=[]
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Explanation failed: {exc}"
        )


@app.post("/report")
async def report(
    audio_file: UploadFile = File(...),
    age: float = Form(...),
    sex: str = Form(...),
    bmi: float = Form(...),
    smoking_pack_years: float = Form(...),
    recording_location: str = Form(...),
    patient_name: str = Form(...),
    patient_id: str = Form(...),
) -> StreamingResponse:
    """Generate a full 9-section clinical PDF report.

    Args:
        audio_file: WAV audio file.
        age: Patient age.
        sex: "M" or "F".
        bmi: Body mass index.
        smoking_pack_years: Cumulative smoking exposure.
        recording_location: Stethoscope placement.
        patient_name: Full patient name.
        patient_id: Unique patient identifier.

    Returns:
        PDF file as a streaming attachment with
        ``Content-Disposition: attachment; filename="report.pdf"``.

    Raises:
        HTTPException: 500 on any generation failure; 503 if no model.

    Requirements: 13.3, 12.4, 12.5
    """
    if not _model_loaded or _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from src.data.preprocess import preprocess_audio
        from src.management.report_generator import ReportGenerator
        from src.models.types import EncounterData, ModelPrediction

        suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await audio_file.read())
            tmp_path = Path(tmp.name)
        spec_np = preprocess_audio(tmp_path, config)
        tmp_path.unlink(missing_ok=True)

        spec_tensor = torch.from_numpy(spec_np).unsqueeze(0).float()
        meta_tensor = _build_metadata_tensor(
            age, sex, bmi, smoking_pack_years, recording_location
        )

        mean, std = _model.predict_with_uncertainty(spec_tensor, meta_tensor)
        mean_np = mean.detach().cpu().numpy()
        std_np = std.detach().cpu().numpy()

        class_names = [dc.value for dc in DiseaseClass]
        pred_idx = int(np.argmax(mean_np))
        pred_class = DiseaseClass(class_names[pred_idx])
        confidence = float(mean_np[pred_idx])

        probabilities = {
            class_names[i]: float(mean_np[i]) for i in range(len(class_names))
        }
        uncertainty_d = {
            class_names[i]: float(std_np[i]) for i in range(len(class_names))
        }

        prediction = ModelPrediction(
            disease_class=pred_class,
            confidence=confidence,
            probabilities=probabilities,
            uncertainty=uncertainty_d,
        )
        risk_result = RiskScorer().score(
            pred_class,
            age,
            smoking_pack_years,
            bmi,
            confidence,
            float(std_np.mean()),
        )

        b64_png = _explainer.explain(spec_tensor) if _explainer else ""

        progression = ProgressionModule().get_trajectory(
            pred_class, risk_result.tier, SmokingStatus.NEVER
        )
        recommendations = RecommendationEngine().get_recommendations(
            pred_class, risk_result.tier
        )

        encounter = EncounterData(
            patient_name=patient_name,
            patient_id=patient_id,
            age=age,
            sex=sex,
            bmi=bmi,
            smoking_pack_years=smoking_pack_years,
            recording_location=recording_location,
            prediction=prediction,
            risk_result=risk_result,
            progression=progression,
            recommendations=recommendations,
            spectrogram_base64=b64_png,
            model_icbhi_score=_icbhi_score,
            training_dataset="ICBHI 2017 + Arashnic Lung Sounds",
        )

        pdf_bytes = ReportGenerator().generate(encounter)

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="report.pdf"'
            },
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Report generation failed: {exc}"
        )
