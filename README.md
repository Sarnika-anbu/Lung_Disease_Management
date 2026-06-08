# Personalized Lung Disease Management System

An AI-driven clinical decision-support tool that classifies lung diseases from respiratory auscultation audio, estimates patient-specific risk, generates evidence-based management recommendations, and produces downloadable clinical PDF reports.

## Architecture

```
Streamlit Frontend (app/)
       │ HTTP
FastAPI Backend (src/api/)
       │
┌──────┴────────────────────────┐
│ Inference Pipeline            │
│  preprocess_audio → model     │
│  MC Dropout uncertainty       │
├───────────────────────────────┤
│ Management Layer              │
│  RiskScorer                   │
│  RecommendationEngine         │
│  ProgressionModule            │
│  ReportGenerator (PDF)        │
├───────────────────────────────┤
│ Explainability (Grad-CAM)     │
└───────────────────────────────┘
       │
Offline Pipeline (src/data/, src/training/)
  download.py → preprocess.py → train.py → evaluate.py
```

## Setup Instructions

### 1. Prerequisites

- Python 3.11+
- pip 23+

### 2. Clone and install

```bash
git clone <repo-url>
cd Lung_Disease_Management
pip install -r requirements.txt
```

## Kaggle API Configuration

The data pipeline uses the Kaggle API to download datasets. Configure credentials using **one** of these methods:

### Option A — Environment variables

```bash
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_api_key
```

### Option B — kaggle.json file

Create `~/.kaggle/kaggle.json`:

```json
{"username": "your_kaggle_username", "key": "your_kaggle_api_key"}
```

Then restrict permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Your API key can be obtained from [https://www.kaggle.com/settings/account](https://www.kaggle.com/settings/account) → **Create New Token**.

## Running the Data Pipeline

```bash
# Step 1 — Download datasets from Kaggle
python src/data/download.py

# Step 2 — Preprocess audio, parse annotations, and create train/test splits
python src/data/preprocess.py
```

Processed data and split CSVs will be saved to `data/processed/`.

## Training the Model

```bash
python src/training/train.py
```

Runs Stage 1 (50 epochs, full model) then Stage 2 (15 epochs, frozen backbone). The best checkpoint is saved to `checkpoints/best.pth`.

## Evaluating the Model

```bash
python src/training/evaluate.py
```

Prints ICBHI Score, macro-F1, and per-class metrics. Saves `outputs/confusion_matrix.png` and `outputs/reliability_diagram.png`.

## Starting the API Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health and model ICBHI score |
| `POST` | `/predict` | Classify audio + return risk score |
| `GET` | `/explain/{recording_id}` | Grad-CAM explanation (last prediction) |
| `POST` | `/report` | Full pipeline → downloadable PDF |

## Starting the Frontend

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`. Upload a WAV file, fill in patient details, and click **Analyze Recording**.

To point the frontend at a non-default API URL:

```bash
API_BASE_URL=http://your-api-host:8000 streamlit run app/streamlit_app.py
```

## Running Tests

```bash
# Run full test suite
pytest tests/ -v

# Run only property-based tests
pytest tests/ -k "property or invariant or coverage or monotonicity"

# Run with reduced Hypothesis examples (faster CI)
HYPOTHESIS_MAX_EXAMPLES=25 pytest tests/ -v
```

## Docker

Build and run the API server in a container:

```bash
docker build -t lung-disease-api .
docker run -p 8000:8000 lung-disease-api
```

## Project Structure

```
.
├── app/
│   └── streamlit_app.py        # Streamlit frontend
├── checkpoints/                # Model checkpoints (best.pth)
├── data/
│   ├── raw/                    # Downloaded raw datasets
│   └── processed/              # Preprocessed spectrograms + splits
├── notebooks/                  # Jupyter notebooks for exploration
├── outputs/                    # Evaluation plots
├── src/
│   ├── api/                    # FastAPI backend
│   │   ├── main.py
│   │   └── schemas.py
│   ├── config.py               # Centralised Config dataclass
│   ├── data/
│   │   ├── download.py         # Kaggle dataset downloader
│   │   └── preprocess.py       # Audio preprocessing + splits
│   ├── explainability/
│   │   └── explainer.py        # Grad-CAM explainer
│   ├── management/
│   │   ├── progression.py      # Disease progression module
│   │   ├── recommendations.py  # Recommendation engine
│   │   ├── report_generator.py # PDF report generator
│   │   └── risk_scorer.py      # Rule-based risk scorer
│   ├── models/
│   │   ├── cbam.py             # CBAM attention module
│   │   ├── model.py            # LungDiseaseModel
│   │   └── types.py            # Shared enums and dataclasses
│   └── training/
│       ├── evaluate.py         # Evaluator
│       └── train.py            # Trainer + FocalLoss + Dataset
├── tests/                      # pytest test suite
├── Dockerfile
└── requirements.txt
```

## Requirements Summary

- **Model**: EfficientNetV2B0 + CBAM + Metadata MLP + MC Dropout
- **Training target**: ICBHI Score ≥ 0.90 on patient-independent test split
- **Inference budget**: ≤ 3 seconds (classification + Grad-CAM)
- **GPU requirement**: 15 GB VRAM, batch size ≤ 32 (Google Colab T4)
- **Datasets**: ICBHI 2017 Respiratory Sound Database + Arashnic Lung Sounds
