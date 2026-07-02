# Data Directory

This folder contains raw and processed datasets for the NLP Sentiment Analyzer.

## Dataset Generation

The project ships with a **self-contained synthetic generator** that produces
realistic customer reviews across 6 domains (electronics, restaurants, hotels,
movies, software, fashion) and 5 languages (EN, ES, FR, DE, PT). No external
downloads or API keys are required.

| Option | Source | Samples | Classes | Command |
|--------|--------|---------|---------|---------|
| **Generate** (default) | Built-in templates | ~96K | 3 | `python scripts/generate_data.py` |
| **Download** | HuggingFace Hub | ~180K | 3 | `python -m src.data.dataset_builder --source download` |

### Real (downloaded) datasets

| Source | Type | Samples | Classes | Module |
|--------|------|---------|---------|--------|
| Amazon Reviews | Product reviews | ~100K | positive, negative | `src/data/downloader.py` |
| Twitter Sentiment140 | Tweets | ~50K | positive, neutral, negative | `src/data/downloader.py` |
| Hotel Reviews | Reviews | ~30K | positive, neutral, negative | `src/data/downloader.py` |

## Directory Structure

```
data/
├── raw/                    # Downloaded raw datasets (gitignored)
├── processed/
│   ├── combined_dataset.csv     # Full combined & balanced dataset
│   ├── train_split.csv         # Training split (80%)
│   ├── val_split.csv           # Validation split (10%)
│   └── test_split.csv          # Test split (10%)
└── README.md              # This file
```

## Label Mapping

| Value | Label    |
|-------|----------|
| 0     | negative |
| 1     | neutral  |
| 2     | positive |

## How to Rebuild

```bash
# Option 1: Self-contained synthetic dataset (recommended, no network)
python scripts/generate_data.py --n-per-class 30000 --multilingual 500

# Option 2: Download real datasets from HuggingFace
python -m src.data.dataset_builder --source download
```

Both will:
1. Generate / fetch raw reviews
2. Clean and preprocess text
3. Combine, balance, and stratify-split
4. Save processed CSVs to `data/processed/`
