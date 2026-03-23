"""
predict.py
----------
Inference engine for the Fraud Mail Intelligence System.
Loads trained models and runs prediction on new emails.
Supports all ML models + optional DistilBERT.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import preprocess_email
from features import FeatureEngineer, compute_severity_score

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "saved"

# Friendly model display names
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "distilbert": "DistilBERT",
}


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (cached)
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: dict = {}
_fe_cache: Optional[FeatureEngineer] = None


def load_feature_engineer() -> FeatureEngineer:
    """Load the saved FeatureEngineer (cached after first load)."""
    global _fe_cache
    if _fe_cache is None:
        fe_path = MODEL_DIR / "feature_engineer.pkl"
        if not fe_path.exists():
            raise FileNotFoundError(
                f"Feature engineer not found at {fe_path}. "
                "Run 'python src/train.py' first."
            )
        _fe_cache = FeatureEngineer.load(str(fe_path))
    return _fe_cache


def load_model(model_name: str):
    """Load a trained sklearn model by name (cached)."""
    global _model_cache
    if model_name not in _model_cache:
        model_path = MODEL_DIR / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model '{model_name}' not found at {model_path}. "
                "Run 'python src/train.py' first."
            )
        _model_cache[model_name] = joblib.load(str(model_path))
    return _model_cache[model_name]


def get_available_models() -> list:
    """Returns list of available trained model names."""
    available = []
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        if (MODEL_DIR / f"{name}.pkl").exists():
            available.append(name)
    if (MODEL_DIR / "distilbert").exists():
        available.append("distilbert")
    return available


# ─────────────────────────────────────────────────────────────────────────────
# DistilBERT inference
# ─────────────────────────────────────────────────────────────────────────────

def _predict_distilbert(text: str) -> tuple:
    """
    Runs DistilBERT inference on raw email text.
    Returns (label, fraud_probability).
    """
    try:
        import torch
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
        )
    except ImportError:
        raise ImportError("Install torch and transformers to use DistilBERT.")

    bert_dir = MODEL_DIR / "distilbert"
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(bert_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(bert_dir))
    model.eval()

    inputs = tokenizer(
        text, truncation=True, padding=True, max_length=256, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]

    fraud_prob = float(probs[1])
    label = 1 if fraud_prob >= 0.5 else 0
    return label, fraud_prob


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction function
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    subject: str,
    sender: str,
    body: str,
    model_name: str = "xgboost",
) -> dict:
    """
    Main prediction function. Takes raw email inputs and returns full analysis.

    Args:
        subject: Email subject line
        sender: Sender email address
        body: Email body text
        model_name: One of 'logistic_regression', 'random_forest', 'xgboost', 'distilbert'

    Returns dict with:
        label: 0 (safe) or 1 (fraud)
        fraud_probability: float 0-1
        confidence: "Low" | "Medium" | "High"
        risk_pct: 0-100
        severity: "Low" | "Medium" | "High" | "Critical"
        contributing_factors: list of strings
        preprocessed: raw preprocessed data (for explainability)
    """
    # ── Preprocess ──
    preprocessed = preprocess_email(subject=subject, sender=sender, body=body)

    # ── Predict ──
    if model_name == "distilbert":
        raw_text = f"{subject} {body}"
        label, fraud_prob = _predict_distilbert(raw_text)
    else:
        fe = load_feature_engineer()
        clf = load_model(model_name)

        # Build single-row dataframe
        row = pd.Series(preprocessed)
        df_single = pd.DataFrame([row])
        X = fe.transform(df_single)

        label = int(clf.predict(X)[0])
        fraud_prob = float(clf.predict_proba(X)[0][1])

    # ── Confidence tier ──
    if fraud_prob >= 0.85 or fraud_prob <= 0.15:
        confidence = "High"
    elif fraud_prob >= 0.65 or fraud_prob <= 0.35:
        confidence = "Medium"
    else:
        confidence = "Low"

    # ── Severity scoring ──
    severity_info = compute_severity_score(fraud_prob, preprocessed)

    return {
        "label": label,
        "prediction": "FRAUD" if label == 1 else "SAFE",
        "fraud_probability": round(fraud_prob, 4),
        "safe_probability": round(1.0 - fraud_prob, 4),
        "confidence": confidence,
        "model_used": MODEL_DISPLAY_NAMES.get(model_name, model_name),
        **severity_info,
        "preprocessed": preprocessed,
    }


def predict_batch(
    df: pd.DataFrame,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """
    Batch prediction on a DataFrame with columns: subject, sender, body.
    Returns the original DataFrame with added prediction columns.
    """
    import sys
    from preprocess import preprocess_dataframe
    from features import FeatureEngineer

    print(f"Batch predicting {len(df)} emails with {model_name}...")
    df_processed = preprocess_dataframe(df)

    fe = load_feature_engineer()
    clf = load_model(model_name)

    X = fe.transform(df_processed)
    df["prediction"] = clf.predict(X)
    df["fraud_probability"] = clf.predict_proba(X)[:, 1]
    df["prediction_label"] = df["prediction"].map({0: "SAFE", 1: "FRAUD"})

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble prediction (combines all available models)
# ─────────────────────────────────────────────────────────────────────────────

def predict_ensemble(
    subject: str,
    sender: str,
    body: str,
) -> dict:
    """
    Runs all available sklearn models and returns averaged probability.
    More robust than single model — good for high-stakes decisions.
    """
    available = [m for m in get_available_models() if m != "distilbert"]
    if not available:
        raise RuntimeError("No trained models found. Run 'python src/train.py' first.")

    fe = load_feature_engineer()
    preprocessed = preprocess_email(subject=subject, sender=sender, body=body)

    row = pd.Series(preprocessed)
    df_single = pd.DataFrame([row])
    X = fe.transform(df_single)

    all_probs = []
    individual_results = {}

    for name in available:
        clf = load_model(name)
        prob = float(clf.predict_proba(X)[0][1])
        all_probs.append(prob)
        individual_results[MODEL_DISPLAY_NAMES.get(name, name)] = round(prob, 4)

    avg_prob = float(np.mean(all_probs))
    label = 1 if avg_prob >= 0.5 else 0
    severity_info = compute_severity_score(avg_prob, preprocessed)

    return {
        "label": label,
        "prediction": "FRAUD" if label == 1 else "SAFE",
        "fraud_probability": round(avg_prob, 4),
        "confidence": "High" if (avg_prob >= 0.8 or avg_prob <= 0.2) else "Medium",
        "model_used": "Ensemble",
        "individual_models": individual_results,
        **severity_info,
        "preprocessed": preprocessed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with a known phishing email
    result = predict(
        subject="URGENT: Your account will be suspended!",
        sender="support@paypa1-security.com",
        body=(
            "Dear Customer, your PayPal account has been flagged for suspicious activity. "
            "Please verify your password immediately at http://paypa1-verify.xyz/secure. "
            "Failure to act within 24 hours will result in permanent account closure. "
            "Wire $500 to restore full access."
        ),
        model_name="xgboost",
    )

    print("\n=== PREDICTION RESULT ===")
    print(f"  Prediction    : {result['prediction']}")
    print(f"  Fraud Prob    : {result['fraud_probability']:.1%}")
    print(f"  Risk Score    : {result['risk_pct']}%")
    print(f"  Severity      : {result['severity']}")
    print(f"  Confidence    : {result['confidence']}")
    print(f"\n  Contributing Factors:")
    for f in result["contributing_factors"]:
        print(f"    • {f}")
