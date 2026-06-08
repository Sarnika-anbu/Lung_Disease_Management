# Design Document: Personalized Lung Disease Management System

## Overview

The Personalized Lung Disease Management System is a clinical AI tool that classifies lung diseases from respiratory auscultation audio, estimates patient risk, generates evidence-based management recommendations, and produces downloadable clinical reports. The system is trained on the ICBHI 2017 Respiratory Sound Database and the Arashnic Lung Sounds dataset.

The system is composed of two main runtime processes:
- A **FastAPI backend** (`src/api/`) that exposes prediction, explanation, reporting, and health endpoints.
- A **Streamlit frontend** (`app/`) that provides a clinician-facing web interface.

A separate offline pipeline (`src/data/`, `src/training/`) handles data download, preprocessing, model training, and evaluation.

### Key Design Decisions

**EfficientNetV2B0 + CBAM + Metadata fusion**: EfficientNetV2B0 provides a strong, lightweight ImageNet-pretrained backbone for spectrogram classification. CBAM attention sharpens the model's focus on clinically relevant frequency-time regions. A metadata MLP branch fuses demographic and smoking history information to improve per-patient personalization.

**Monte Carlo Dropout (MC Dropout)**: Epistemic uncertainty is estimated by running 20 stochastic forward passes with dropout active. This provides clinically useful confidence bounds rather than a single point estimate.

**Grad-CAM explainability**: Clinicians need to understand why a prediction was made. Grad-CAM produces a spectrogram heatmap that maps model attention to frequency-time regions, with clinical annotations (wheeze region, crackle burst, low-frequency artifact).

**Rule-based risk scoring and recommendations**: Clinical guidelines (GOLD 2024, GINA) are encoded as deterministic rule tables, ensuring consistency, auditability, and ease of update as guidelines evolve.

**ReportLab PDF generation**: A structured 9-section PDF report is generated per encounter, providing a clinical record suitable for documentation and referral.

---

## Architecture

The system is composed of four high-level layers:

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Frontend (app/)               │
│  Audio Upload │ Metadata Form │ Results Panel │ PDF DL  │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (multipart/form-data, JSON)
┌────────────────────────▼────────────────────────────────┐
│                  FastAPI Backend (src/api/)              │
│  POST /predict │ GET /explain │ POST /report │ GET /health│
│  Pydantic validation │ Error handling                   │
└───┬───────────────┬───────────────┬──────────────────────┘
    │               │               │
    ▼               ▼               ▼
┌──────────┐  ┌──────────┐  ┌──────────────────────────┐
│Inference │  │Explainer │  │ Management Layer          │
│Pipeline  │  │(Grad-CAM)│  │ Risk Scorer               │
│          │  │          │  │ Recommendation Engine     │
│ Preproc  │  │          │  │ Progression Module        │
│ Model    │  │          │  │ Report Generator          │
│ MC Drop  │  │          │  │                           │
└──────────┘  └──────────┘  └──────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Offline Pipeline                        │
│  download.py │ preprocess.py │ train.py │ evaluate.py   │
└─────────────────────────────────────────────────────────┘
```

### Data Flow (Inference)

1. Clinician uploads a WAV file and fills in patient metadata on the Streamlit frontend.
2. Frontend sends `POST /predict` to the FastAPI backend.
3. API validates inputs with Pydantic, passes audio to the Preprocessor.
4. Preprocessor resamples → bandpass filters → noise-gates → segments → computes log-mel spectrogram → produces (3, 128, 216) tensor.
5. Model runs 20 MC Dropout forward passes; returns mean probability vector (len 7) and per-class std vector (len 7).
6. Risk Scorer maps (disease_class, age, pack_years, BMI, confidence, uncertainty) → Risk_Tier + numeric score.
7. API returns prediction JSON to frontend.
8. Frontend calls `POST /report`; API triggers Explainer (Grad-CAM) + Recommendation Engine + Progression Module + Report Generator.
9. PDF is streamed back as a downloadable attachment.

---

## Components and Interfaces

### 1. Data Pipeline (`src/data/`)

#### `download.py`
- `download_datasets() -> None`
  - Reads `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars or `~/.kaggle/kaggle.json`.
  - Downloads ICBHI and Arashnic datasets; skips if target directory is non-empty.
  - Validates archive size > 0 bytes before extraction.
  - Exits with non-zero status on credential, network, or validation failure.

#### `preprocess.py`
- `parse_icbhi_annotations(data_dir: Path) -> pd.DataFrame`
  - Parses annotation CSVs; cross-references `ICBHI_diagnosis.txt`; warns on unmatched patient IDs.
- `map_label(raw_label: str) -> Optional[DiseaseClass]`
  - Maps raw diagnosis strings to `DiseaseClass` enum; logs unrecognised labels.
- `preprocess_audio(audio_path: Path, config: Config) -> np.ndarray`
  - Resamples → bandpass → spectral gating → segments → log-mel → (3, 128, 216).
- `save_metadata(df: pd.DataFrame, output_path: Path) -> None`
  - Writes `metadata.csv`; records `null` for missing fields.
- `split_dataset(df: pd.DataFrame, split_def: Path) -> Tuple[pd.DataFrame, pd.DataFrame]`
  - Applies official ICBHI patient-level split; asserts no patient-ID overlap.

### 2. Model (`src/models/model.py`)

```python
class LungDiseaseModel(nn.Module):
    def __init__(self, metadata_input_dim: int, num_classes: int = 7): ...
    def forward(self, spectrogram: Tensor, metadata: Tensor) -> Tensor: ...
    def predict_with_uncertainty(
        self, spectrogram: Tensor, metadata: Tensor, n_passes: int = 20
    ) -> Tuple[Tensor, Tensor]: ...
```

- **Backbone**: EfficientNetV2B0 (ImageNet weights); classifier head replaced.
- **CBAM**: Applied after the final convolutional block, before global average pooling.
- **Metadata MLP**: `[input_dim → 64 → 32 → 16]` with ReLU activations; null fields zero-filled.
- **Fusion head**: Concatenate backbone output + metadata MLP output → `[concat_dim → 128 → 7]` + softmax.
- **MC Dropout**: `predict_with_uncertainty` sets `model.train()` for 20 forward passes, then returns mean and std of the resulting probability matrix.

#### CBAM Module (`src/models/cbam.py`)
```python
class CBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16): ...
    def forward(self, x: Tensor) -> Tensor: ...
```

### 3. Trainer (`src/training/train.py`)

```python
class Trainer:
    def __init__(self, model: LungDiseaseModel, config: TrainConfig): ...
    def train_stage1(self) -> None: ...  # 50 epochs, full model, CosineAnnealingLR
    def freeze_backbone(self) -> None: ...  # sets backbone requires_grad=False; asserts
    def train_stage2(self) -> None: ...  # 15 epochs, head only, balanced subset
    def _save_best_checkpoint(self, epoch: int, score: float) -> None: ...
```

- Focal Loss: `gamma=2`, `alpha=inverse_class_freq`, `label_smoothing=0.1`.
- WeightedRandomSampler: weights = 1 / class_count, normalized.
- SpecAugment: applied in `__getitem__` of training Dataset only.
- Mixup: applied on minority-class pairs in the training loop only.
- CosineAnnealingLR: `T_max=50`, Stage 1 only.
- AdamW: `lr=1e-4`, `weight_decay=1e-2`.

### 4. Evaluator (`src/training/evaluate.py`)

```python
class Evaluator:
    def __init__(self, model: LungDiseaseModel, test_loader: DataLoader): ...
    def compute_icbhi_score(self) -> float: ...
    def compute_metrics(self) -> EvaluationReport: ...
    def plot_confusion_matrix(self, save_path: Path) -> None: ...
    def plot_reliability_diagram(self, save_path: Path) -> None: ...
```

- Outputs `outputs/confusion_matrix.png` and `outputs/reliability_diagram.png`.
- Prints ICBHI score, macro-F1, per-class table to stdout.

### 5. Explainer (`src/explainability/explainer.py`)

```python
class GradCAMExplainer:
    def __init__(self, model: LungDiseaseModel): ...
    def explain(self, spectrogram: Tensor) -> str:
        """Returns base64-encoded annotated PNG."""
        ...
    def _annotate_heatmap(
        self, heatmap: np.ndarray, spectrogram: np.ndarray
    ) -> Image: ...
```

- Uses `pytorch-grad-cam` targeting the last conv layer of EfficientNetV2B0.
- Annotation logic:
  - Sustained activity 100–1000 Hz → `"wheeze region"` (if most prominent pattern).
  - Short transients 200–2000 Hz → `"crackle burst"` (if most prominent pattern).
  - Activity < 100 Hz → `"low-frequency artifact"` (if most prominent pattern).
  - No pattern → no annotation.
- Must complete within 3 s combined with model inference.

### 6. Risk Scorer (`src/management/risk_scorer.py`)

```python
class RiskScorer:
    def score(
        self,
        disease_class: DiseaseClass,
        age: float,
        pack_years: float,
        bmi: float,
        confidence: float,
        uncertainty: float,
    ) -> RiskResult: ...

@dataclass
class RiskResult:
    tier: RiskTier          # Low | Medium | High
    score: float            # 0–100
```

- Fully rule-based; no ML inference at runtime.
- Score mapped to tier: 0–33 → Low, 34–66 → Medium, 67–100 → High.

### 7. Recommendation Engine (`src/management/recommendations.py`)

```python
class RecommendationEngine:
    def get_recommendations(
        self, disease_class: DiseaseClass, risk_tier: RiskTier
    ) -> List[Recommendation]: ...

@dataclass
class Recommendation:
    icon: str
    text: str
    sub_text: str
    source: str
```

- Lookup table keyed by `(DiseaseClass, RiskTier)`.
- COPD recommendations cite GOLD 2024; Asthma cites GINA.
- Always appends a spirometry referral recommendation.

### 8. Progression Module (`src/management/progression.py`)

```python
class ProgressionModule:
    def get_trajectory(
        self,
        disease_class: DiseaseClass,
        risk_tier: RiskTier,
        smoking_status: SmokingStatus,
    ) -> ProgressionForecast: ...

@dataclass
class ProgressionForecast:
    month_3: Dict[str, float]
    month_6: Dict[str, float]
    month_12: Dict[str, float]
```

- Strict lookup table; no runtime model inference.
- Values sourced from GOLD natural-history tables.

### 9. Report Generator (`src/management/report_generator.py`)

```python
class ReportGenerator:
    def generate(self, encounter: EncounterData) -> bytes: ...
```

- Uses `reportlab==4.0`.
- Produces a 9-section PDF (see Requirement 12).
- Returns raw PDF bytes; API streams as `application/pdf` attachment.

### 10. FastAPI Backend (`src/api/`)

#### Endpoints

| Method | Path | Input | Output |
|--------|------|-------|--------|
| POST | `/predict` | multipart: `audio_file`, `age`, `sex`, `bmi`, `smoking_pack_years`, `recording_location` | JSON: `disease_class`, `confidence`, `probabilities`, `uncertainty`, `risk_tier`, `risk_score` |
| GET | `/explain/{recording_id}` | path param | JSON: `spectrogram_base64`, `highlighted_regions` |
| POST | `/report` | same as `/predict` + `patient_name`, `patient_id` | PDF attachment |
| GET | `/health` | — | JSON: `status`, `model_version`, `icbhi_score` |

#### Pydantic Schemas (`src/api/schemas.py`)
```python
class PredictRequest(BaseModel):
    age: float
    sex: Literal["M", "F"]
    bmi: float
    smoking_pack_years: float
    recording_location: RecordingLocation

class PredictResponse(BaseModel):
    disease_class: str
    confidence: float
    probabilities: Dict[str, float]
    uncertainty: Dict[str, float]
    risk_tier: str
    risk_score: float
```

- Validation failures → HTTP 422.
- Internal model errors → HTTP 500.

### 11. Streamlit Frontend (`app/streamlit_app.py`)

- **Left panel**: WAV uploader, patient metadata form (name, ID, age, sex, BMI, pack-years, recording location).
- **Submit button**: calls `POST /predict` then `POST /report`.
- **Right panel**: diagnosis badge, confidence bar, Altair probability chart, Grad-CAM image, risk metrics, recommendations list.
- **Download button**: delivers PDF to browser.

---

## Data Models

### Enums (`src/models/types.py`)

```python
class DiseaseClass(str, Enum):
    COPD = "COPD"
    HEALTHY = "Healthy"
    URTI = "URTI"
    BRONCHIECTASIS = "Bronchiectasis"
    PNEUMONIA = "Pneumonia"
    BRONCHIOLITIS = "Bronchiolitis"
    ASTHMA = "Asthma"

class RiskTier(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class SmokingStatus(str, Enum):
    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"

class RecordingLocation(str, Enum):
    TRACHEA = "Trachea"
    ANTERIOR_LEFT = "Anterior left"
    ANTERIOR_RIGHT = "Anterior right"
    POSTERIOR_LEFT = "Posterior left"
    POSTERIOR_RIGHT = "Posterior right"
    LATERAL_LEFT = "Lateral left"
    LATERAL_RIGHT = "Lateral right"
```

### Processed Data Schema

#### `data/processed/metadata.csv`
| Column | Type | Notes |
|--------|------|-------|
| patient_id | str | ICBHI patient identifier |
| age | float \| null | null if missing |
| sex | str \| null | "M" or "F"; null if missing |
| bmi | float \| null | null if missing |
| smoking_status | str \| null | null if missing |
| recording_location | str \| null | null if missing |

#### `data/processed/splits/train.csv` / `test.csv`
| Column | Type |
|--------|------|
| patient_id | str |
| recording_id | str |
| segment_id | int |
| spectrogram_path | str |
| disease_class | str |

### Spectrogram Tensor
- Shape: `(3, 128, 216)` — channels are [log-mel, delta, delta-delta]
- Time axis: 216 frames (zero-padded or truncated)
- Frequency axis: 128 mel bins
- dtype: `float32`

### Model Outputs
```python
@dataclass
class ModelPrediction:
    disease_class: DiseaseClass
    confidence: float           # max of mean probability vector
    probabilities: Dict[str, float]  # 7-class mean softmax
    uncertainty: Dict[str, float]    # 7-class std from MC dropout
```

### Configuration (`src/config.py`)
```python
@dataclass
class Config:
    # Data
    raw_icbhi_dir: Path = Path("data/raw/icbhi")
    raw_arashnic_dir: Path = Path("data/raw/arashnic")
    processed_dir: Path = Path("data/processed")
    splits_dir: Path = Path("data/processed/splits")
    metadata_path: Path = Path("data/processed/metadata.csv")
    checkpoints_dir: Path = Path("checkpoints")
    outputs_dir: Path = Path("outputs")

    # Preprocessing
    target_sample_rate: int = 4000
    bandpass_low: float = 50.0
    bandpass_high: float = 2000.0
    noise_gate_threshold: float = 0.5
    segment_duration: float = 5.0
    segment_overlap: float = 0.5
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 128
    n_time_frames: int = 216

    # Training
    batch_size: int = 32
    stage1_epochs: int = 50
    stage2_epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-2
    focal_gamma: float = 2.0
    focal_label_smoothing: float = 0.1
    mixup_alpha: float = 0.4
    spec_augment_time_mask: int = 80
    spec_augment_freq_mask: int = 30
    spec_augment_num_masks: int = 2
    mc_dropout_passes: int = 20
```

### EncounterData (Report Generator Input)
```python
@dataclass
class EncounterData:
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
```

---

## Correctness Properties

### Property 1: Annotation Parsing Round-Trip Integrity
For any valid ICBHI annotation CSV row, the parsed DataFrame must contain field values that exactly match the source CSV values (start_time, end_time, crackle_label, wheeze_label, patient_id, recording_id).
**Validates: Requirement 2.1**

### Property 2: Label Mapping Completeness
For any source diagnosis string in the official ICBHI vocabulary, `map_label` must return exactly one of the seven DiseaseClasses. For any unknown string, it must return `None`.
**Validates: Requirements 2.3, 2.4**

### Property 3: Null Metadata Substitution Completeness
For any patient record with any subset of fields missing, the output row in `metadata.csv` must record `null` for every missing field and the correct value for every present field.
**Validates: Requirement 2.5**

### Property 4: Resampling Invariant
For any audio recording at any input sample rate, after resampling, the output must have a sample rate of exactly 4000 Hz.
**Validates: Requirement 3.1**

### Property 5: Bandpass Filter Frequency Invariant
For sinusoids in the passband 50–2000 Hz, output power > input × 0.5. For sinusoids outside the passband, output power < input × 0.1.
**Validates: Requirement 3.2**

### Property 6: Segmentation Uniformity
For any audio recording of any length, all output segments must have exactly 20,000 samples. Recordings shorter than 5 s produce exactly one zero-padded segment.
**Validates: Requirement 3.4**

### Property 7: Spectrogram Output Shape Invariant
For any valid 5-second 4000 Hz segment, the preprocessing pipeline output must be a tensor of exactly shape (3, 128, 216).
**Validates: Requirements 3.5, 3.6**

### Property 8: Patient-Independent Split No-Overlap
The intersection of train and test patient ID sets must always be empty. If overlap exists, an AssertionError must be raised listing the overlapping IDs.
**Validates: Requirements 4.1, 4.2, 15.3**

### Property 9: Split CSV Column Schema
Every split CSV must contain exactly the columns: patient_id, recording_id, segment_id, spectrogram_path, disease_class.
**Validates: Requirement 4.5**

### Property 10: Model Input Channel Validation
For input tensors with channel count ≠ 3, the Model must raise a descriptive error. For channel count = 3, no error is raised.
**Validates: Requirement 5.1**

### Property 11: Metadata Null Zero-Fill
For any metadata input containing null/NaN values, the MLP branch must replace all null/NaN with 0.0. The MLP output must be a vector of length 16.
**Validates: Requirement 5.4**

### Property 12: Softmax Output Invariant
For any valid (spectrogram, metadata) input pair, the forward pass must return a vector of length 7 with all values in [0.0, 1.0] summing to 1.0 within 1e-5.
**Validates: Requirement 5.5**

### Property 13: MC Dropout Output Shape and Non-Negativity
`predict_with_uncertainty` must return two vectors of length 7. All std values must be ≥ 0. The mean vector must satisfy the softmax invariant.
**Validates: Requirement 5.6**

### Property 14: Sampler Weight Correctness
For any class-count mapping, weight_i = 1.0 / count_i. Minority classes must have strictly higher weights than majority classes.
**Validates: Requirement 6.2**

### Property 15: Augmentation Training-Only Invariant
In eval mode, the data pipeline returns deterministic output. In training mode, repeated calls on the same input have non-zero probability of different output.
**Validates: Requirements 6.3, 6.4**

### Property 16: Backbone Freeze Completeness
After `freeze_backbone()`, every backbone parameter must have `requires_grad = False`. If any remain True, an error must be raised.
**Validates: Requirement 6.8**

### Property 17: Checkpoint Save Monotonicity
The checkpoint is overwritten if and only if current_score > previous_best.
**Validates: Requirement 6.10**

### Property 18: ICBHI Score Formula Correctness
For any TP, TN, FP, FN inputs, `compute_icbhi_score` must return exactly `(sensitivity + specificity) / 2`.
**Validates: Requirement 7.1**

### Property 19: Evaluation Full-Class Coverage
Per-class metrics output must contain exactly seven entries — one per DiseaseClass — for both precision/recall/F1 and ROC-AUC.
**Validates: Requirements 7.3, 7.4**

### Property 20: Grad-CAM Annotation Correctness and Exclusivity
Each highlighted region receives at most one annotation. Correct label applied per frequency range. No annotation when no pattern is detected.
**Validates: Requirements 8.3–8.6**

### Property 21: Risk Score Range and Tier Validity
For any valid input combination, the score must be in [0, 100] and the tier must be exactly one of {Low, Medium, High}.
**Validates: Requirement 9.2**

### Property 22: Risk Score Monotonicity
Increasing a risk-elevating factor (pack_years, age, lower confidence, higher uncertainty) while holding others constant must not decrease the score.
**Validates: Requirement 9.3**

### Property 23: Recommendation Coverage and Schema Validity
For all 21 (DiseaseClass × RiskTier) pairs, the engine returns a non-empty list where every recommendation has non-null, non-empty values for all four fields.
**Validates: Requirements 10.1, 10.4**

### Property 24: COPD Recommendations Reference GOLD 2024
For any COPD query across all three risk tiers, at least one recommendation's `source` contains "GOLD 2024".
**Validates: Requirement 10.2**

### Property 25: Asthma Recommendations Reference GINA
For any Asthma query across all three risk tiers, at least one recommendation's `source` contains "GINA".
**Validates: Requirement 10.3**

### Property 26: Spirometry Referral Always Present
For any (DiseaseClass, RiskTier) combination, at least one recommendation's `text` or `sub_text` contains "spirometry".
**Validates: Requirement 10.5**

### Property 27: Progression Forecast Schema and Value Range
For any valid 3-factor input, the forecast contains exactly keys month_3, month_6, month_12 with all probability values in [0.0, 1.0].
**Validates: Requirement 11.2**

### Property 28: Progression Forecast Determinism
Calling `get_trajectory` twice with identical arguments returns identical results.
**Validates: Requirement 11.3**

### Property 29: PDF Report Structure Invariant
For any valid EncounterData, the generated PDF contains exactly 9 sections in the specified order.
**Validates: Requirement 12.1**

### Property 30: Disclaimer Footer Content
The PDF's section 9 must contain both the model's ICBHI score and the training dataset name.
**Validates: Requirement 12.2**

### Property 31: Prediction API Response Schema
For any valid POST /predict request, the JSON response contains all six required fields with correct types and value ranges.
**Validates: Requirement 13.1**

### Property 32: Input Validation Rejects Invalid Requests
For any POST /predict request with invalid input, the API returns HTTP 422 with a descriptive body identifying the invalid field(s).
**Validates: Requirements 13.5, 13.6**

---

## Error Handling

### Data Pipeline Errors

| Error Condition | Handling |
|-----------------|----------|
| Missing/invalid Kaggle credentials | Print to stderr with credential file path; exit non-zero |
| Network error during download | Print dataset name and failure reason to stderr; exit non-zero |
| 0-byte downloaded archive | Print error to stderr; exit non-zero; do not attempt extraction |
| Unmatched patient_id in diagnosis file | Log warning to stderr with patient_id; exclude patient's segments |
| Unknown diagnosis label | Log warning to stderr with label and source file; exclude affected segments only |
| Malformed annotation CSV row | Log warning with file name and row number; skip row; continue |
| Unassigned patient ID in split definition | Log warning with patient_id; exclude from both splits |
| Non-empty intersection in split | Raise AssertionError listing overlapping patient_ids |

### Model and Inference Errors

| Error Condition | Handling |
|-----------------|----------|
| Wrong channel count in spectrogram input | Raise descriptive error with received and expected counts |
| Backbone parameters still trainable after freeze | Halt training; raise error |
| Model file not found at startup | Log critical error; API returns 503 on all endpoints |

### API Errors

| Error Condition | HTTP Status | Body |
|-----------------|-------------|------|
| Pydantic validation failure | 422 | JSON with field-level error details |
| Internal model failure | 500 | JSON with descriptive error message |
| PDF generation failure | 500 | JSON error; no file attachment |

### Preprocessing Edge Cases

- Audio shorter than 5 seconds → single zero-padded segment (normal behavior, not an error).
- Trailing segment shorter than 5 seconds → zero-padded to exactly 5 seconds.
- Audio with sample rate ≠ 4000 Hz → always resampled (not an error).

---

## Testing Strategy

### Approach

The system uses both **unit/example-based tests** and **property-based tests** (PBT):

- **Unit tests**: Verify specific behaviors, error conditions, and integration points with concrete examples.
- **Property tests**: Verify universal invariants across large randomly-generated input spaces using the `hypothesis` library.

### Property-Based Testing

**Library**: `hypothesis` with `@settings(max_examples=100)` on every property test.

**Tag format**: Each property test must include:
```python
# Feature: lung-disease-management, Property N: <property_text>
```

| Property | Test File | Hypothesis Strategy |
|----------|-----------|---------------------|
| P1: Annotation Parsing Round-Trip | `tests/test_preprocess.py` | `st.from_regex` for CSV rows |
| P2: Label Mapping Completeness | `tests/test_preprocess.py` | `st.sampled_from` + `st.text()` |
| P3: Null Metadata Substitution | `tests/test_preprocess.py` | `st.fixed_dictionaries` with optional fields |
| P4: Resampling Invariant | `tests/test_preprocess.py` | `st.integers(min_value=4000, max_value=48000)` |
| P5: Bandpass Filter Frequency | `tests/test_preprocess.py` | `st.floats` for frequency; generate sinusoids |
| P6: Segmentation Uniformity | `tests/test_preprocess.py` | `st.integers(min_value=1)` for audio length |
| P7: Spectrogram Output Shape | `tests/test_preprocess.py` | Float arrays of 20000 samples |
| P8: Patient-Independent Split | `tests/test_split.py` | `st.sets` of patient IDs |
| P9: Split CSV Column Schema | `tests/test_split.py` | Mock dataframes with `st.data` |
| P10: Model Input Channel Validation | `tests/test_model.py` | `st.integers(min_value=1, max_value=10)` |
| P11: Metadata Null Zero-Fill | `tests/test_model.py` | `st.lists` of nullable floats |
| P12: Softmax Output Invariant | `tests/test_model.py` | Random (spectrogram, metadata) tensors |
| P13: MC Dropout Output Shape | `tests/test_model.py` | Random valid inputs |
| P14: Sampler Weight Correctness | `tests/test_training.py` | `st.dictionaries` of class→count |
| P15: Augmentation Training-Only | `tests/test_training.py` | Fixed spectrogram; toggle mode |
| P16: Backbone Freeze Completeness | `tests/test_training.py` | Model instances |
| P17: Checkpoint Save Monotonicity | `tests/test_training.py` | `st.floats` for score pairs |
| P18: ICBHI Score Formula | `tests/test_evaluate.py` | `st.integers(min_value=0)` for TP/TN/FP/FN |
| P19: Evaluation Full-Class Coverage | `tests/test_evaluate.py` | `st.lists` of (label, prediction) pairs |
| P20: Grad-CAM Annotation Correctness | `tests/test_explainer.py` | Synthetic heatmaps at known frequency ranges |
| P21: Risk Score Range and Tier | `tests/test_risk_scorer.py` | `st.floats` + `st.sampled_from(DiseaseClass)` |
| P22: Risk Score Monotonicity | `tests/test_risk_scorer.py` | Vary one factor; hold others constant |
| P23: Recommendation Coverage | `tests/test_recommendations.py` | Exhaustive 21-pair enumeration |
| P24: COPD GOLD 2024 Citation | `tests/test_recommendations.py` | `st.sampled_from(RiskTier)` |
| P25: Asthma GINA Citation | `tests/test_recommendations.py` | `st.sampled_from(RiskTier)` |
| P26: Spirometry Referral Present | `tests/test_recommendations.py` | `st.sampled_from(DiseaseClass × RiskTier)` |
| P27: Progression Forecast Schema | `tests/test_progression.py` | `st.sampled_from` all 3-factor combinations |
| P28: Progression Forecast Determinism | `tests/test_progression.py` | Identical inputs; assert equal outputs |
| P29: PDF Report Structure | `tests/test_report.py` | `st.builds(EncounterData)` |
| P30: Disclaimer Footer Content | `tests/test_report.py` | `st.builds(EncounterData)` |
| P31: Prediction API Response Schema | `tests/test_api.py` | `st.builds(PredictRequest)` |
| P32: Input Validation Rejects Invalid | `tests/test_api.py` | `st.from_type` with invalid values |

### Unit Tests

| File | Focus |
|------|-------|
| `tests/test_download.py` | Credential loading, skip logic, archive verification, network errors |
| `tests/test_preprocess.py` | Malformed row skipping, unmatched patient_id warnings |
| `tests/test_model.py` | CBAM insertion position, head replacement |
| `tests/test_api.py` | HTTP 500 on model failure, PDF attachment header, health endpoint |
| `tests/test_report.py` | HTTP 500 on PDF failure, section ordering |

### Integration Tests

| Test | Description |
|------|-------------|
| End-to-end inference timing | Single WAV file through prediction + Grad-CAM; assert < 3 s |
| Full pipeline smoke test | Download → preprocess → split → train (1 epoch mock) → evaluate |
| API startup | All four endpoints return expected status codes |
| Output files | `confusion_matrix.png` and `reliability_diagram.png` exist after evaluation |

### Test Execution

```bash
pytest tests/ -v
pytest tests/ -k "property"
HYPOTHESIS_MAX_EXAMPLES=50 pytest tests/
```
