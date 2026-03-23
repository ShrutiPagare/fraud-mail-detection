"""
train.py
--------
Layer D: ML Engine
Multi-model training pipeline with comparison metrics.
Trains: Logistic Regression, Random Forest, XGBoost, DistilBERT (optional)
Saves all models + metrics report.
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

# Internal imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import preprocess_dataframe
from features import FeatureEngineer

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "saved"
METRICS_DIR = ROOT / "models" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load email dataset.
    Expected columns: subject, sender, body, label (0=safe, 1=fraud)
    Falls back to generating synthetic demo data if file not found.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} emails from {csv_path}")
        return df
    else:
        print(f"Dataset not found at {csv_path}. Generating synthetic demo data...")
        return _generate_demo_data()


def _generate_demo_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generates a small synthetic dataset for demo/testing purposes.
    In production, replace with real phishing datasets from Kaggle/SpamAssassin.
    """
    np.random.seed(42)

    fraud_subjects = [
        "URGENT: Your account will be suspended!",
        "Verify your PayPal account immediately",
        "FINAL WARNING: Wire transfer required",
        "Your bank account has been compromised",
        "Action Required: Confirm your identity",
        "You have won $1,000,000 lottery prize!",
        "ALERT: Unauthorized login detected",
        "Invoice #8821 - Payment overdue",
        "Confirm your password to avoid suspension",
        "FREE gift card - Claim NOW before it expires",
    ]

    fraud_bodies = [
        "Dear customer, your account requires immediate verification. Click http://paypa1-secure.xyz/verify to update your password and billing information. Failure to comply within 24 hours will result in permanent account termination.",
        "We have detected suspicious activity on your account. Please wire $500 to secure your funds immediately. Contact us at support@amaz0n-help.top for urgent assistance.",
        "Congratulations! You have been selected to receive a $1,000,000 inheritance from our late client. Please provide your bank account details and passport copy to claim your prize.",
        "URGENT: Your PayPal account balance of $2,340.00 is on hold. Verify your identity at http://secure-paypal.click/login to release funds. Act now!",
        "Your account password will expire in 24 hours. Login at http://192.168.1.1/gmail-secure to update your credentials immediately.",
    ]

    safe_subjects = [
        "Meeting notes from yesterday's standup",
        "Project update: Q3 milestone achieved",
        "Welcome to our newsletter",
        "Your order has been shipped",
        "Invoice attached for your records",
        "Team lunch this Friday at noon",
        "Feedback requested on the new design",
        "Re: Follow-up from our call",
        "Monthly report - please review",
        "Happy birthday! From the team",
    ]

    safe_bodies = [
        "Hi team, please find attached the meeting notes from yesterday. Key action items: review the Q3 roadmap, finalize the budget proposal, and schedule the next sprint planning session.",
        "Your order #12345 has been shipped and will arrive within 3-5 business days. You can track your package using the link in your account dashboard.",
        "Dear subscriber, thank you for signing up for our newsletter. This month we're featuring our new product line and upcoming events. Unsubscribe at any time.",
        "Please find attached the invoice for services rendered in October. Payment is due within 30 days. Contact accounting@company.com with any questions.",
        "Hi John, great connecting with you today. As discussed, I'll send over the proposal by end of week. Looking forward to working together.",
    ]

    records = []

    # Generate fraud emails
    for i in range(n_samples // 2):
        subj = fraud_subjects[i % len(fraud_subjects)]
        body = fraud_bodies[i % len(fraud_bodies)]
        # Add some noise
        senders = [
            "support@paypa1-security.com",
            "noreply@amaz0n-verify.xyz",
            "alert@secure-bank.top",
            f"admin{np.random.randint(100,999)}@temp-mail.click",
            "irs-refund@gov-official.loan",
        ]
        records.append({
            "subject": subj,
            "sender": senders[i % len(senders)],
            "body": body,
            "label": 1,
        })

    # Generate safe emails
    for i in range(n_samples // 2):
        subj = safe_subjects[i % len(safe_subjects)]
        body = safe_bodies[i % len(safe_bodies)]
        senders = [
            "alice@company.com",
            "newsletter@shopify.com",
            "orders@amazon.com",
            "hr@acme-corp.com",
            "john.doe@gmail.com",
        ]
        records.append({
            "subject": subj,
            "sender": senders[i % len(senders)],
            "body": body,
            "label": 0,
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Generated {len(df)} synthetic emails ({df['label'].sum()} fraud, {(df['label']==0).sum()} safe)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    """Returns dict of model_name -> sklearn estimator."""
    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",   # Handles imbalanced datasets
            solver="lbfgs",
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Computes all metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    print(f"\n{'='*50}")
    print(f"  Model: {model_name.upper()}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<15}: {v:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Fraud"]))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# DistilBERT fine-tuning (optional — requires GPU for good speed)
# ─────────────────────────────────────────────────────────────────────────────

def train_distilbert(
    X_train_texts: list,
    y_train: list,
    X_test_texts: list,
    y_test: list,
    epochs: int = 3,
    batch_size: int = 16,
) -> dict:
    """
    Fine-tunes DistilBERT on email classification.
    Only runs if transformers + torch are available.
    Returns metrics dict.
    """
    try:
        import torch
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
            Trainer,
            TrainingArguments,
        )
        from torch.utils.data import Dataset
    except ImportError:
        print("Transformers/torch not available. Skipping DistilBERT.")
        return {}

    print("\nFine-tuning DistilBERT...")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    class EmailDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=256,
            )
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = EmailDataset(X_train_texts, y_train)
    test_dataset = EmailDataset(X_test_texts, y_test)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "distilbert"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(MODEL_DIR / "distilbert" / "logs"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    # Evaluate
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_prob = predictions.predictions[:, 1]

    metrics = {
        "model": "distilbert",
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    # Save
    model.save_pretrained(str(MODEL_DIR / "distilbert"))
    tokenizer.save_pretrained(str(MODEL_DIR / "distilbert"))
    print("DistilBERT saved.")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_all(
    csv_path: str = None,
    test_size: float = 0.2,
    train_bert: bool = False,
):
    """
    Full training pipeline:
      1. Load + preprocess data
      2. Feature engineering
      3. Train all ML models
      4. Evaluate + save metrics
      5. Save models + feature engineer
      6. (Optional) Fine-tune DistilBERT
    """
    csv_path = csv_path or str(DATA_DIR / "emails.csv")

    # ── Step 1: Load data ──
    df = load_data(csv_path)
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")

    # ── Step 2: Preprocess ──
    print("\nPreprocessing emails...")
    df = preprocess_dataframe(df)

    # ── Step 3: Train/test split ──
    X_df = df.drop(columns=["label"], errors="ignore")
    y = df["label"].values

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(X_train_df)}, Test: {len(X_test_df)}")

    # ── Step 4: Feature engineering ──
    print("\nBuilding features...")
    fe = FeatureEngineer(max_tfidf_features=10000, ngram_range=(1, 2))
    X_train = fe.fit_transform(X_train_df)
    X_test = fe.transform(X_test_df)
    print(f"Feature matrix shape: {X_train.shape}")

    # Save feature engineer
    fe.save(str(MODEL_DIR / "feature_engineer.pkl"))

    # ── Step 5: Train ML models ──
    models = get_models()
    all_metrics = []

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        start = time.time()
        clf.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"Training time: {elapsed:.1f}s")

        metrics = evaluate_model(clf, X_test, y_test, name)
        metrics["train_time_sec"] = round(elapsed, 2)
        all_metrics.append(metrics)

        # Save model
        joblib.dump(clf, str(MODEL_DIR / f"{name}.pkl"))
        print(f"Saved: {MODEL_DIR / f'{name}.pkl'}")

    # ── Step 6: (Optional) DistilBERT ──
    if train_bert:
        train_texts = X_train_df["raw_text"].fillna("").tolist()
        test_texts = X_test_df["raw_text"].fillna("").tolist()
        bert_metrics = train_distilbert(
            train_texts, y_train.tolist(),
            test_texts, y_test.tolist(),
        )
        if bert_metrics:
            all_metrics.append(bert_metrics)

    # ── Step 7: Save metrics report ──
    metrics_path = str(METRICS_DIR / "model_comparison.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Print comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-"*70)
    for m in all_metrics:
        print(
            f"{m['model']:<25} "
            f"{m['accuracy']:>10.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1_score']:>10.4f} "
            f"{m['roc_auc']:>10.4f}"
        )

    return all_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Fraud Mail Detection models")
    parser.add_argument("--data", type=str, default=None, help="Path to email CSV dataset")
    parser.add_argument("--bert", action="store_true", help="Also fine-tune DistilBERT")
    args = parser.parse_args()

    train_all(csv_path=args.data, train_bert=args.bert)
