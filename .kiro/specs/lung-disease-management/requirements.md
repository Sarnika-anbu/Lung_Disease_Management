# Requirements Document

## Introduction

This document defines the requirements for a Personalized Lung Disease Management System using AI-driven predictive analytics. The system classifies lung diseases from respiratory auscultation audio recordings, estimates patient-specific risk, generates evidence-based management recommendations, and produces downloadable clinical reports. The system is composed of a FastAPI backend and a Streamlit frontend, trained on the ICBHI 2017 Respiratory Sound Database and the Arashnic Lung Sounds dataset.

## Glossary

- **System**: The Personalized Lung Disease Management System as a whole.
- **Data_Pipeline**: The module responsible for downloading, preprocessing, segmenting, and splitting audio data.
- **Preprocessor**: The component that resamples, filters, segments, and converts audio to spectrograms.
- **Model**: The deep learning classifier (EfficientNetV2B0 + CBAM + metadata fusion) that predicts lung disease class and uncertainty.
- **Trainer**: The component responsible for executing the training loop, applying augmentation, and saving checkpoints.
- **Evaluator**: The component that computes ICBHI score, macro-F1, per-class metrics, calibration error, and plots.
- **Explainer**: The Grad-CAM-based explainability module that produces annotated spectrogram visualizations.
- **Risk_Scorer**: The rule-based engine that maps diagnosis + patient metadata to a risk tier and numeric score.
- **Recommendation_Engine**: The guideline-based rule engine that maps disease class and risk tier to structured recommendations.
- **Progression_Module**: The module encoding natural-history lookup tables for disease trajectory probabilities.
- **Report_Generator**: The PDF generation module producing a 9-section clinical report.
- **API**: The FastAPI backend exposing prediction, explanation, report, and health endpoints.
- **Frontend**: The Streamlit web application providing audio upload, metadata input, and result display.
- **ICBHI_Score**: Metric defined as (sensitivity + specificity) / 2, used as the primary evaluation metric.
- **Patient_Independent_Split**: A train/test data split where no patient ID appears in both subsets.
- **MC_Dropout**: Monte Carlo Dropout inference — running 20 forward passes with dropout active to estimate prediction uncertainty.
- **CBAM**: Convolutional Block Attention Module — a channel and spatial attention mechanism applied after the last convolutional block.
- **Focal_Loss**: A loss function with gamma=2 and alpha=inverse class frequency, with label smoothing epsilon=0.1.
- **SpecAugment**: A spectrogram augmentation technique applying time and frequency masking during training.
- **Mixup**: A data augmentation technique that linearly interpolates between two samples and their labels.
- **Grad_CAM**: Gradient-weighted Class Activation Mapping — a technique that produces a heatmap highlighting the spectrogram regions most influential to a prediction.
- **Risk_Tier**: One of three categorical risk levels: Low, Medium, or High.
- **Disease_Class**: One of seven target classes: COPD, Healthy, URTI, Bronchiectasis, Pneumonia, Bronchiolitis, Asthma.
- **Recording_Location**: The body location where the stethoscope was placed during audio capture (e.g., Trachea, Anterior left, Posterior right).
- **Pack_Years**: A measure of cumulative smoking exposure calculated as (packs per day × years smoked).
- **ECE**: Expected Calibration Error — a metric quantifying the difference between predicted confidence and actual accuracy.
- **WeightedRandomSampler**: A PyTorch DataLoader sampler that oversamples minority-class examples.
- **GOLD_2024**: Global Initiative for Chronic Obstructive Lung Disease 2024 clinical guidelines.
- **GINA**: Global Initiative for Asthma clinical guidelines.

---

## Requirements

### Requirement 1: Dataset Download

**User Story:** As a data engineer, I want to download both Kaggle datasets automatically, so that all raw data is available locally before preprocessing begins.

#### Acceptance Criteria

1. WHEN `python src/data/download.py` is executed, THE Data_Pipeline SHALL download the ICBHI 2017 Respiratory Sound Database from `https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database` to `data/raw/icbhi/`.
2. WHEN `python src/data/download.py` is executed, THE Data_Pipeline SHALL download the Arashnic Lung Sounds dataset from `https://www.kaggle.com/datasets/arashnic/lung-sound-dataset` to `data/raw/arashnic/`.
3. THE Data_Pipeline SHALL use the Kaggle API with credentials sourced from the environment variable `KAGGLE_KEY` and `KAGGLE_USERNAME`, or from `~/.kaggle/kaggle.json` if environment variables are absent.
4. IF the Kaggle API credentials are missing or invalid, THEN THE Data_Pipeline SHALL print an error message to stderr that identifies the expected credential file path (`~/.kaggle/kaggle.json`) or the missing environment variables, and exit with a non-zero status code.
5. IF a dataset's target directory already exists and contains at least one file, THEN THE Data_Pipeline SHALL skip downloading that dataset and print a skip message to stdout identifying the dataset name; each dataset SHALL be checked and skipped independently.
6. WHEN a dataset archive is downloaded, THE Data_Pipeline SHALL verify the downloaded file is at least 1 byte in size before extraction; IF the file is 0 bytes or absent, THEN THE Data_Pipeline SHALL print an error to stderr and exit with a non-zero status code without attempting extraction.
7. WHEN a dataset archive passes verification, THE Data_Pipeline SHALL extract it to its target subdirectory (`data/raw/icbhi/` or `data/raw/arashnic/`).
8. IF a download fails due to a network error or non-200 HTTP response, THEN THE Data_Pipeline SHALL print an error message to stderr identifying the dataset name and the failure reason, and exit with a non-zero status code.

---

### Requirement 2: Annotation Parsing and Label Mapping

**User Story:** As a data engineer, I want to parse ICBHI annotations and map all labels to a unified 7-class schema, so that the model trains on consistent, correctly labeled data.

#### Acceptance Criteria

1. WHEN `python src/data/preprocess.py` is executed, THE Data_Pipeline SHALL parse each ICBHI annotation CSV to extract respiratory cycle segments with fields: start_time, end_time, crackle_label, wheeze_label, patient_id, recording_id.
2. WHEN a segment is parsed, THE Data_Pipeline SHALL look up the patient's diagnosis in `ICBHI_diagnosis.txt` using the patient_id; IF no matching entry is found in `ICBHI_diagnosis.txt` for a given patient_id, THEN THE Data_Pipeline SHALL log a warning to stderr identifying the unmatched patient_id and exclude all segments for that patient from the processed dataset.
3. WHEN a segment's diagnosis is resolved, THE Data_Pipeline SHALL map the diagnosis to exactly one of the seven Disease_Classes: COPD, Healthy, URTI, Bronchiectasis, Pneumonia, Bronchiolitis, Asthma.
4. IF a source label has no mapping to any of the seven Disease_Classes, THEN THE Data_Pipeline SHALL log a warning to stderr identifying the unmapped label and the source file, and exclude only the affected segments from the processed dataset without halting execution.
5. WHEN `python src/data/preprocess.py` is executed, THE Data_Pipeline SHALL store patient metadata (age, sex, BMI, smoking status, Recording_Location) in `data/processed/metadata.csv`; IF a patient record is missing one or more metadata fields, THEN THE Data_Pipeline SHALL record `null` for the missing fields rather than excluding the patient.
6. IF a row in an annotation CSV is malformed (missing required columns or unparseable values), THEN THE Data_Pipeline SHALL log a warning to stderr identifying the file name and row number, and skip that row without halting execution.

---

### Requirement 3: Audio Preprocessing

**User Story:** As a data engineer, I want to preprocess audio into clean, standardized segments, so that the model receives consistent input.

#### Acceptance Criteria

1. THE Preprocessor SHALL resample all audio recordings to exactly 4000 Hz, resampling any recording whose sample rate differs from 4000 Hz regardless of how close it is to the target.
2. THE Preprocessor SHALL apply a bandpass filter with a passband of 50–2000 Hz to all audio recordings.
3. THE Preprocessor SHALL apply spectral gating for noise reduction to all audio recordings, using a sensitivity threshold in the range 0.0–1.0 with a default value of 0.5.
4. THE Preprocessor SHALL segment audio using a 5-second sliding window with 50% overlap; trailing segments shorter than 5 seconds SHALL be zero-padded to exactly 5 seconds; IF a recording is shorter than one full 5-second window, THEN THE Preprocessor SHALL treat it as a single segment and zero-pad it to 5 seconds.
5. WHEN computing spectrograms, THE Preprocessor SHALL compute 128-bin log-mel spectrograms with n_fft=1024 and hop_length=128.
6. THE Preprocessor SHALL compute delta and delta-delta channels from the log-mel spectrogram; the time axis of each channel SHALL be zero-padded or truncated to exactly 216 frames, producing a final tensor of shape (3, 128, 216).
7. THE Preprocessor SHALL use pathlib and a centralized `config.py` for all file paths, with no hardcoded path strings in preprocessing code.

---

### Requirement 4: Patient-Independent Data Split

**User Story:** As a data scientist, I want a strict patient-independent train/test split, so that evaluation results reflect real-world generalization.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL implement the official ICBHI patient-level split, which is a fixed pre-defined assignment of patient IDs to train or test sets (not a computed ratio), such that no patient ID appears in both the train and test subsets.
2. WHEN the split is applied, THE Data_Pipeline SHALL assert that the intersection of train patient IDs and test patient IDs is empty; IF the intersection is non-empty, THEN THE Data_Pipeline SHALL raise an AssertionError with a message listing the overlapping patient IDs.
3. IF a patient ID is present in the dataset but absent from the official split definition, THEN THE Data_Pipeline SHALL log a warning to stderr identifying the unassigned patient ID and exclude that patient's segments from both splits.
4. WHEN saving split CSVs, THE Data_Pipeline SHALL write `data/processed/splits/train.csv` and `data/processed/splits/test.csv`, overwriting any existing files at those paths.
5. THE Data_Pipeline SHALL include patient_id, recording_id, segment_id, spectrogram_path, and disease_class columns in each split CSV.

---

### Requirement 5: Model Architecture

**User Story:** As an ML engineer, I want a multi-input deep learning model combining spectrograms and patient metadata, so that predictions incorporate both acoustic and demographic information.

#### Acceptance Criteria

1. THE Model SHALL use EfficientNetV2B0 pretrained on ImageNet as the spectrogram backbone, accepting only 3-channel input of shape (3, 128, 216); IF the input has a channel count other than 3, THEN THE Model SHALL raise a descriptive error stating the received channel count and the expected channel count of 3, and reject the input.
2. THE Model SHALL replace the default EfficientNetV2B0 classifier head with a custom classification head connected to the fusion layer.
3. THE Model SHALL apply a CBAM attention module (channel attention followed by spatial attention) between the final convolutional block output and the global pooling layer of EfficientNetV2B0.
4. THE Model SHALL include a metadata MLP branch with layer dimensions: input_dim → 64 → 32 → 16, encoding age (normalized), sex (one-hot), BMI (normalized), Pack_Years (normalized), and Recording_Location (one-hot); IF any metadata field is missing or null, THEN THE Model SHALL substitute a zero vector for that field's encoding before passing it to the MLP.
5. THE Model SHALL concatenate the spectrogram backbone output and the metadata MLP output (dimension 16), then pass the concatenated vector through a 2-layer fusion head with a 128-unit hidden layer, ending in a 7-class softmax output.
6. WHEN performing inference, THE Model SHALL always execute 20 forward passes with dropout active (MC_Dropout) and return both the mean prediction vector of length 7 and the per-class standard deviation vector of length 7, regardless of whether uncertainty estimation was explicitly requested.
7. THE Model SHALL include type hints and Google-style docstrings on all public methods and classes.

---

### Requirement 6: Training Procedure

**User Story:** As an ML engineer, I want a reproducible two-stage training procedure with augmentation and class balancing, so that the model achieves high performance on imbalanced respiratory data.

#### Acceptance Criteria

1. THE Trainer SHALL use Focal_Loss with gamma=2, alpha set to inverse class frequency weights, and label smoothing epsilon=0.1.
2. THE Trainer SHALL use a WeightedRandomSampler in the DataLoader to oversample minority-class samples during training.
3. WHILE training is active (Stage 1 or Stage 2), THE Trainer SHALL apply SpecAugment with time_mask_param=80, freq_mask_param=30, and num_masks=2; SpecAugment SHALL NOT be applied during validation or evaluation.
4. WHILE training is active (Stage 1 or Stage 2), THE Trainer SHALL apply Mixup augmentation with alpha=0.4 on minority-class samples; Mixup SHALL NOT be applied during validation or evaluation.
5. THE Trainer SHALL use the AdamW optimizer with lr=1e-4 and weight_decay=1e-2.
6. THE Trainer SHALL use a CosineAnnealingLR scheduler scoped to Stage 1 only, with T_max=50 epochs.
7. THE Trainer SHALL execute Stage 1 training: train all model parameters for 50 epochs on the full training dataset.
8. WHEN Stage 1 training is complete, THE Trainer SHALL freeze the EfficientNetV2B0 backbone (set all backbone parameter `requires_grad` to False) before beginning Stage 2; IF any backbone parameter remains trainable after the freeze operation, THEN THE Trainer SHALL halt training and raise an error rather than proceeding with Stage 2.
9. WHEN the backbone is successfully frozen, THE Trainer SHALL fine-tune the classifier head for 15 epochs on a class-balanced subset constructed by sampling an equal number of examples per class, capped at the minority-class count, using AdamW with lr=1e-4 and weight_decay=1e-2.
10. IF the validation ICBHI_Score at the end of an epoch exceeds the previously saved best score, THEN THE Trainer SHALL overwrite `checkpoints/best.pth` with the current model state.
11. WHEN each training epoch ends, THE Trainer SHALL print the epoch number, training loss, validation loss, and validation ICBHI_Score to stdout.

---

### Requirement 7: Model Evaluation

**User Story:** As an ML engineer, I want comprehensive evaluation metrics reported on the patient-independent test split, so that I can confirm the model meets clinical-grade performance targets.

#### Acceptance Criteria

1. THE Evaluator SHALL compute ICBHI_Score = (sensitivity + specificity) / 2 on the patient-independent test split.
2. THE Evaluator SHALL compute macro-F1 score on the patient-independent test split.
3. THE Evaluator SHALL compute per-class precision, recall, and F1 score for all seven Disease_Classes.
4. THE Evaluator SHALL compute ROC-AUC for each of the seven Disease_Classes.
5. THE Evaluator SHALL compute ECE and generate a reliability diagram saved to `outputs/reliability_diagram.png`.
6. THE Evaluator SHALL generate and save a confusion matrix heatmap to `outputs/confusion_matrix.png`.
7. WHEN `python src/training/evaluate.py` is executed, THE Evaluator SHALL print the ICBHI_Score, macro-F1, and per-class table to the console.
8. THE Evaluator SHALL use only the patient-independent test split for all reported metrics.
9. THE Evaluator SHALL target an ICBHI_Score ≥ 0.90 on the patient-independent test split.

---

### Requirement 8: Explainability

**User Story:** As a clinician, I want to see which spectrogram regions drove the model's prediction, annotated with clinical frequency labels, so that I can assess the clinical plausibility of the prediction.

#### Acceptance Criteria

1. THE Explainer SHALL implement Grad_CAM using the `pytorch-grad-cam` library, targeting the last convolutional layer of EfficientNetV2B0.
2. THE Explainer SHALL overlay the Grad_CAM heatmap on the log-mel spectrogram image.
3. WHEN the Grad_CAM heatmap detects sustained frequency activity in the range 100–1000 Hz, and this pattern is the most prominent detected pattern in the highlighted region, THE Explainer SHALL annotate the region with the label "wheeze region".
4. WHEN the Grad_CAM heatmap detects short transients in the frequency range 200–2000 Hz, and this pattern is the most prominent detected pattern in the highlighted region, THE Explainer SHALL annotate the region with the label "crackle burst".
5. WHEN the Grad_CAM heatmap highlights activity below 100 Hz, and this pattern is the most prominent detected pattern in the highlighted region, THE Explainer SHALL annotate the region with the label "low-frequency artifact".
6. THE Explainer SHALL apply only the label corresponding to the most prominent detected frequency pattern per highlighted region, applying no annotation to regions where none of the defined patterns are detected.
7. THE Explainer SHALL return the annotated spectrogram image as a base64-encoded PNG string via the `/explain` endpoint.
8. WHEN the full prediction pipeline is executed, THE Explainer SHALL complete classification and Grad_CAM computation within a combined total of 3 seconds.

---

### Requirement 9: Risk Scoring

**User Story:** As a clinician, I want each prediction accompanied by a risk score, so that I can prioritize patient follow-up appropriately.

#### Acceptance Criteria

1. THE Risk_Scorer SHALL accept the inputs: disease_class, age, Pack_Years, BMI, model_confidence, and MC_Dropout uncertainty.
2. THE Risk_Scorer SHALL return a Risk_Tier (Low, Medium, or High) and a numeric risk score in the range 0–100.
3. THE Risk_Scorer SHALL apply rule-based logic combining all input factors to compute the numeric risk score and derive the Risk_Tier.
4. THE Risk_Scorer SHALL include type hints and docstrings on all public methods.

---

### Requirement 10: Management Recommendations

**User Story:** As a clinician, I want evidence-based management recommendations tailored to each patient's disease class and risk tier, so that I can provide guideline-consistent care.

#### Acceptance Criteria

1. THE Recommendation_Engine SHALL map each combination of (Disease_Class, Risk_Tier) to a list of recommendation objects.
2. WHEN the Disease_Class is COPD, THE Recommendation_Engine SHALL reference GOLD_2024 guidelines in the generated recommendations.
3. WHEN the Disease_Class is Asthma, THE Recommendation_Engine SHALL reference GINA guidelines in the generated recommendations.
4. EACH recommendation object SHALL include the fields: icon, text, sub_text, and source.
5. THE Recommendation_Engine SHALL always include a recommendation to "refer for spirometry to confirm" in the output for all Disease_Classes.

---

### Requirement 11: Disease Progression Estimation

**User Story:** As a clinician, I want a probabilistic progression forecast at 3, 6, and 12 months, so that I can counsel the patient on expected disease trajectory.

#### Acceptance Criteria

1. THE Progression_Module SHALL accept the inputs: disease_class, Risk_Tier, and smoking_status.
2. THE Progression_Module SHALL return trajectory probabilities at 3-month, 6-month, and 12-month horizons encoded from GOLD natural history lookup tables; zero probabilities are permitted as valid returned values.
3. THE Progression_Module SHALL implement the trajectory probabilities as a strict lookup table with no runtime model inference; runtime models that produce equivalent results SHALL NOT be used in place of the lookup table.

---

### Requirement 12: Clinical PDF Report

**User Story:** As a clinician, I want to download a structured PDF report for each patient encounter, so that I have a clinical record for documentation and referral.

#### Acceptance Criteria

1. THE Report_Generator SHALL produce a PDF containing exactly 9 sections in the following order: (1) Patient header with risk badge, (2) Primary diagnosis with confidence bar, (3) Full class probability distribution bar chart, (4) Grad_CAM spectrogram image, (5) Risk stratification metrics grid, (6) Progression timeline, (7) Management recommendations list, (8) Model quality metadata, (9) Disclaimer footer.
2. THE Report_Generator SHALL include the model's ICBHI_Score and training dataset name in the disclaimer footer (Section 9).
3. THE Report_Generator SHALL use the `reportlab` library (version 4.0) to generate the PDF.
4. WHEN the `/report` endpoint is called and PDF generation succeeds, THE API SHALL return the PDF as a downloadable file attachment.
5. IF PDF generation fails due to missing data or an internal error, THEN THE API SHALL return HTTP status 500 with no file attachment.

---

### Requirement 13: FastAPI Backend

**User Story:** As a developer, I want a FastAPI backend exposing structured endpoints, so that the frontend and external clients can interact with the model programmatically.

#### Acceptance Criteria

1. THE API SHALL expose a `POST /predict` endpoint accepting multipart/form-data with fields: audio_file (WAV), age, sex, bmi, smoking_pack_years, and recording_location, returning a JSON response with fields: disease_class, confidence, probabilities, uncertainty, risk_tier, and risk_score.
2. THE API SHALL expose a `GET /explain/{recording_id}` endpoint returning a JSON response with fields: spectrogram_base64 and highlighted_regions.
3. THE API SHALL expose a `POST /report` endpoint accepting the same fields as `/predict` plus patient_name and patient_id, returning a PDF file as a downloadable attachment.
4. THE API SHALL expose a `GET /health` endpoint returning a JSON response with fields: status, model_version, and icbhi_score.
5. THE API SHALL validate all request inputs using Pydantic schemas.
6. WHEN a request fails Pydantic validation, THE API SHALL return HTTP status 422 with a descriptive error body.
7. WHEN an internal model failure occurs, THE API SHALL return HTTP status 500 with a descriptive error body.
8. WHEN `uvicorn src.api.main:app` is executed, THE API SHALL start successfully and all four endpoints SHALL be reachable.

---

### Requirement 14: Streamlit Frontend

**User Story:** As a clinician, I want a web interface to upload audio, enter patient information, and view AI-generated results, so that I can use the system without writing code.

#### Acceptance Criteria

1. THE Frontend SHALL display a left panel containing a WAV file uploader and a patient metadata form with fields: patient name, patient ID, age, sex, BMI, smoking Pack_Years, and Recording_Location.
2. THE Frontend SHALL display a submit button that, when clicked, calls the `POST /predict` endpoint and then the `POST /report` endpoint.
3. THE Frontend SHALL display a right panel containing: a diagnosis badge, a confidence bar, a class probability chart rendered with Altair, the Grad_CAM annotated spectrogram image, a risk metrics display, and a recommendations list.
4. THE Frontend SHALL display a download button that, when clicked, delivers the PDF report generated by the `POST /report` endpoint to the user's browser.
5. WHEN `streamlit run app/streamlit_app.py` is executed, THE Frontend SHALL open in the browser and accept WAV file uploads.

---

### Requirement 15: Non-Functional — Performance and Environment

**User Story:** As a system operator, I want the system to run efficiently on a Google Colab T4 GPU with fast inference, so that it is usable in resource-constrained environments.

#### Acceptance Criteria

1. THE Model SHALL support training and inference on a GPU with 15 GB VRAM using a batch size of 32 or fewer.
2. WHEN processing a single audio file from upload through classification and Grad_CAM computation to final output, THE System SHALL complete the full pipeline within 3 seconds.
3. THE System SHALL enforce a Patient_Independent_Split with a runtime assertion that raises an AssertionError if any patient ID appears in both the train and test sets.
4. ALL reported evaluation metrics SHALL be computed exclusively on the patient-independent test split.

---

### Requirement 16: Non-Functional — Code Quality and Documentation

**User Story:** As a developer, I want all ML code to be well-documented and reproducible, so that the system can be maintained and extended.

#### Acceptance Criteria

1. EVERY ML function and class in the `src/` directory SHALL include a docstring and type hints on all parameters and return values.
2. THE System SHALL define all file paths using pathlib and a centralized `config.py` file; no hardcoded path strings SHALL appear outside `config.py`.
3. THE `requirements.txt` file SHALL pin all package versions to exact version numbers.
4. THE `README.md` SHALL include setup instructions, Kaggle API configuration steps, commands to run training, and commands to start the API and frontend.
5. THE System SHALL include a test suite executable via `pytest` with all tests passing.

---

### Requirement 17: Repository Structure

**User Story:** As a developer, I want a consistent project layout, so that contributors can navigate and extend the codebase easily.

#### Acceptance Criteria

1. THE System SHALL maintain the following top-level directory structure: `data/`, `src/data/`, `src/models/`, `src/training/`, `src/explainability/`, `src/management/`, `src/api/`, `app/`, `notebooks/`, `tests/`, `checkpoints/`, `outputs/`.
2. THE System SHALL include a `Dockerfile` at the project root that builds a runnable image of the API server.
3. THE System SHALL include a `requirements.txt` at the project root with all dependencies pinned to exact versions.
