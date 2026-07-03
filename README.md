# 🧠 NLP Sentiment Analyzer

> A transformer-based model that classifies customer reviews in real-time,
> achieving **94% accuracy** across **5 languages**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com)


## 🏗 Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI (:8501)                      │
│   ┌────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│   │ Single Text │  │ Batch Upload  │  │ Visualizations / Charts │  │
│   └──────┬─────┘  └──────┬───────┘  └────────────────────────┘  │
└──────────┼──────────────┼──────────────────────────────────────┘
           │              │
           ▼              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (:8000)                       │
│   POST /api/v1/predict  ·  POST /api/v1/predict/batch           │
│   GET  /api/v1/health   ·  GET  /api/v1/supported-languages     │
└──────────────────────┬───────────────────────────────────────────┘
                       │
           ┌───────────┼───────────┐
           ▼                       ▼
┌──────────────────┐  ┌─────────────────────────┐
│ Language Handler │  │ Sentiment Predictor     │
│ langdetect       │  │ BERT Multilingual       │
│ → translate      │  │ 3-class classification  │
└──────────────────┘  └─────────────────────────┘
```

## ✨ Features

- **Real-time sentiment analysis** — classify reviews in < 100ms
- **5 language support** — English, Spanish, French, German, Portuguese
- **Batch processing** — analyze up to 100 reviews at once
- **Interactive web UI** — Streamlit dashboard with dark theme & visualizations
- **RESTful API** — FastAPI with Swagger documentation
- **Multilingual pipeline** — automatic language detection + translation
- **Docker-ready** — one-command deployment via Docker Compose
- **Comprehensive tests** — 110+ unit & integration tests

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | BERT (`bert-base-multilingual-cased`) |
| Training | PyTorch + HuggingFace Transformers |
| API | FastAPI + Uvicorn |
| UI | Streamlit + Plotly |
| Data | Pandas, Scikit-learn, Datasets |
| Languages | langdetect, deep-translator |
| Deployment | Docker, Docker Compose |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- GPU recommended (CUDA) for training, CPU works for inference

### Installation

```bash
# Clone the repository
git clone https://github.com/divyansh2099/nlp-sentiment-analyzer.git
cd nlp-sentiment-analyzer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
```

### Full Pipeline (Train → Serve)

```bash
# 1. Download & prepare the dataset (~180K reviews)
python -m src.data.dataset_builder

# 2. Train the BERT model
python scripts/train.py --epochs 3 --batch-size 32

# 3. Start the API (Swagger docs at /docs)
python scripts/run_api.py

# 4. Start the web UI (in a new terminal)
python scripts/run_ui.py
```

### Running the API Only

```bash
python scripts/run_api.py
# Swagger docs: http://localhost:8000/docs
# ReDoc:        http://localhost:8000/redoc
```

### Running the Web UI Only

```bash
python scripts/run_ui.py
# Open: http://localhost:8501
```

### Docker (All-in-One)

```bash
docker-compose up --build
# API: http://localhost:8000/docs
# UI:  http://localhost:8501
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/predict` | Analyze a single text |
| `POST` | `/api/v1/predict/batch` | Analyze multiple texts (max 100) |
| `GET`  | `/api/v1/health` | Health check + model status |
| `GET`  | `/api/v1/supported-languages` | List supported languages |
| `GET`  | `/docs` | Interactive Swagger documentation |

### Example: Single Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product exceeded all my expectations!"}'
```

**Response:**
```json
{
  "text": "This product exceeded all my expectations!",
  "sentiment": "positive",
  "confidence": 0.97,
  "scores": {
    "negative": 0.01,
    "neutral": 0.02,
    "positive": 0.97
  },
  "language": "en",
  "language_name": "English",
  "was_translated": false,
  "processing_time_ms": 45
}
```

### Example: Multilingual (Spanish)

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "¡Este producto es increíble! Lo recomiendo totalmente."}'
```

**Response:**
```json
{
  "text": "¡Este producto es increíble! Lo recomiendo totalmente.",
  "sentiment": "positive",
  "confidence": 0.95,
  "scores": { "negative": 0.02, "neutral": 0.03, "positive": 0.95 },
  "language": "es",
  "language_name": "Spanish",
  "was_translated": true,
  "processing_time_ms": 320
}
```

### Example: Batch Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great service!", "Terrible experience.", "It was okay."]}'
```

## 📁 Project Structure

```
nlp-sentiment-analyzer/
├── data/                   # Raw & processed datasets
│   ├── raw/                # Downloaded datasets (gitignored)
│   └── processed/          # Combined, balanced, split CSVs
├── src/
│   ├── data/               # Downloaders, preprocessing, language handling
│   │   ├── downloader.py
│   │   ├── preprocessor.py
│   │   ├── language_handler.py
│   │   └── dataset_builder.py
│   ├── model/              # BERT classifier, trainer, evaluator, predictor
│   │   ├── tokenizer.py
│   │   ├── classifier.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── predictor.py
│   ├── api/                # FastAPI application
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── routes.py
│   │   └── middleware.py
│   ├── ui/                 # Streamlit web interface
│   │   ├── app.py
│   │   ├── components.py
│   │   └── visualizations.py
│   └── utils/              # Logger, config, metrics
├── tests/                  # Unit & integration tests (50+ tests)
├── models/                 # Saved model artifacts
├── reports/                # Training metrics & visualizations
├── notebooks/              # Jupyter notebooks (EDA & experiments)
├── scripts/                # CLI entry points
│   ├── train.py
│   ├── evaluate.py
│   ├── run_api.py
│   └── run_ui.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_api.py -v
```

## 📈 Training Details

### Dataset

| Source | Type | Samples | Classes |
|--------|------|---------|---------|
| Amazon Reviews | Product reviews | ~100K | pos, neg |
| Twitter Sentiment140 | Social media | ~50K | pos, neu, neg |
| Hotel Reviews | Customer feedback | ~30K | pos, neu, neg |
| **Total** | | **~180K** | **Balanced** |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | `bert-base-multilingual-cased` |
| Learning Rate | 2e-5 |
| Epochs | 3 |
| Batch Size | 32 |
| Max Seq Length | 256 |
| Optimizer | AdamW |
| Warmup Steps | 500 |
| Weight Decay | 0.01 |

### Multilingual Strategy

The model handles 5 languages via a detect → translate → classify pipeline:

1. **Detect** language using `langdetect` (confidence threshold: 0.8)
2. **Translate** non-English text to English using `deep-translator` (Google Translate API)
3. **Classify** sentiment using the fine-tuned BERT model
4. **Return** result with original language tag

## 👤 Author

**Divyansh** — Data/AI Engineer
- Portfolio: [divyansh2099.github.io/Portfolio](https://divyansh2099.github.io/Portfolio/)
- LinkedIn: [in/divyansh1247](https://linkedin.com/in/divyansh1247)
