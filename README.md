# AI-Powered Fraud Mail Intelligence System

An end-to-end ML + GenAI system for detecting fraudulent emails with explainability, threat summaries, and severity scoring.

## Project Architecture

```
fraud-mail-detection/
├── data/
│   ├── raw/                    # Raw datasets (phishing + legitimate)
│   └── processed/              # Cleaned, feature-engineered CSVs
├── models/
│   ├── saved/                  # Trained .pkl / .pt model files
│   └── metrics/                # Performance reports (JSON/CSV)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_explainability.ipynb
├── src/
│   ├── preprocess.py           # Text cleaning, tokenization
│   ├── features.py             # Feature engineering (TF-IDF, fraud signals)
│   ├── train.py                # Multi-model training pipeline
│   ├── predict.py              # Inference engine
│   ├── explain.py              # SHAP + LIME explainability
│   └── genai_reasoning.py      # GenAI fraud explanation layer
├── app.py                      # Streamlit UI
├── requirements.txt
└── README.md
```

## Tech Stack

| Layer | Tools |
|---|---|
| NLP Processing | spaCy, NLTK, scikit-learn |
| ML Models | Logistic Regression, Random Forest, XGBoost, DistilBERT |
| Explainability | SHAP, LIME |
| GenAI Layer | Transformers (FLAN-T5 / GPT-2) |
| UI | Streamlit |
| Data | pandas, numpy |

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python src/train.py          # Train and save models
streamlit run app.py          # Launch the web interface
```

## Metrics (Example Results)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 94.1% | 93.8% | 94.5% | 94.1% | 0.978 |
| Random Forest | 96.3% | 96.1% | 96.6% | 96.3% | 0.991 |
| XGBoost | 97.2% | 97.0% | 97.4% | 97.2% | 0.994 |
| DistilBERT | 98.6% | 98.4% | 98.8% | 98.6% | 0.998 |

## Dataset Sources

- Kaggle Phishing Email Dataset
- SpamAssassin Public Corpus
- Enron Email Dataset (curated subset)
- CEAS 2008 Phishing Corpus
