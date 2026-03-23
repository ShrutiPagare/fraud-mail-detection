"""
explain.py
----------
Layer E (partial): Explainability Engine
SHAP and LIME explanations for model predictions.
Shows WHICH words and features triggered the fraud detection.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Block TensorFlow from being imported by shap/transformers ──
# TF causes a DLL crash on Windows when its native runtime isn't compatible.
# SHAP only needs torch-based or sklearn paths — TF is never required here.
os.environ["TRANSFORMERS_NO_TF"] = "1"         # tells transformers: skip TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # silence TF C++ logs
sys.modules.setdefault("tensorflow", None)      # type: ignore  # poison the TF import slot

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works in Streamlit)
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from predict import load_model, load_feature_engineer
from preprocess import preprocess_email

ROOT = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explanation
# ─────────────────────────────────────────────────────────────────────────────

def explain_with_shap(
    subject: str,
    sender: str,
    body: str,
    model_name: str = "xgboost",
    top_n: int = 15,
) -> dict:
    """
    Generates SHAP explanation for a single email prediction.

    Returns:
        shap_values: array of SHAP values
        feature_names: list of feature names
        top_features: list of (feature_name, shap_value) for top contributors
        fig: matplotlib figure (bar chart)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap: pip install shap")

    # Prepare features
    preprocessed = preprocess_email(subject=subject, sender=sender, body=body)
    fe = load_feature_engineer()
    clf = load_model(model_name)

    row = pd.Series(preprocessed)
    df_single = pd.DataFrame([row])
    X = fe.transform(df_single)

    # SHAP TreeExplainer for tree-based models (XGBoost, RF)
    # Use LinearExplainer for Logistic Regression
    if model_name in ("xgboost", "random_forest"):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        # For binary: shap_values may be list of 2 arrays
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # Fraud class
        else:
            sv = shap_values[0]
    else:
        # Logistic Regression — use LinearExplainer with dense matrix
        X_dense = X.toarray()
        explainer = shap.LinearExplainer(clf, X_dense)
        shap_values_all = explainer.shap_values(X_dense)
        sv = shap_values_all[0] if isinstance(shap_values_all, list) else shap_values_all[0]

    feature_names = fe.get_feature_names()

    # Pair and sort by absolute SHAP value
    pairs = sorted(
        zip(feature_names, sv.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    # Filter to non-zero
    top_features = [(name, val) for name, val in pairs[:top_n] if val != 0.0]

    # ── Build SHAP bar chart ──
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    names = [f[0] for f in top_features][::-1]
    vals = [f[1] for f in top_features][::-1]
    colors = ["#FF4B4B" if v > 0 else "#00CC88" for v in vals]

    bars = ax.barh(names, vals, color=colors, alpha=0.85, height=0.6)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("SHAP Value (positive = increases fraud probability)", color="#CCCCCC", fontsize=9)
    ax.set_title("Feature Importance — SHAP Values", color="#FFFFFF", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#CCCCCC", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333")
    ax.spines["bottom"].set_color("#333")

    plt.tight_layout()

    return {
        "shap_values": sv,
        "feature_names": feature_names,
        "top_features": top_features,
        "fig": fig,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LIME Explanation
# ─────────────────────────────────────────────────────────────────────────────

def explain_with_lime(
    subject: str,
    sender: str,
    body: str,
    model_name: str = "xgboost",
    top_n: int = 10,
    num_samples: int = 500,
) -> dict:
    """
    Generates LIME text explanation.
    Shows which words most influenced the fraud prediction.

    Returns:
        word_weights: list of (word, weight) sorted by importance
        fig: matplotlib figure
        lime_exp: raw LIME explanation object
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        raise ImportError("Install lime: pip install lime")

    fe = load_feature_engineer()
    clf = load_model(model_name)

    def predict_proba_text(texts: list) -> np.ndarray:
        """Wrapper so LIME can call the full pipeline."""
        from preprocess import clean_text
        rows = [{"cleaned_text": clean_text(t), "raw_text": t,
                 "subject": "", "sender": "", "body": t,
                 "sender_has_brand_spoof": 0, "sender_suspicious_tld": 0,
                 "sender_numeric_substitution": 0, "sender_domain_length": 10,
                 "sender_subdomain_count": 1, "org_count": 0, "money_count": 0,
                 "person_count": 0, "gpe_count": 0}
                for t in texts]
        df_batch = pd.DataFrame(rows)
        X = fe.transform(df_batch)
        return clf.predict_proba(X)

    explainer = LimeTextExplainer(
        class_names=["Safe", "Fraud"],
        split_expression=r"\s+",
        bow=True,
    )

    raw_text = f"{subject} {body}"
    lime_exp = explainer.explain_instance(
        raw_text,
        predict_proba_text,
        num_features=top_n,
        num_samples=num_samples,
    )

    # Word weights for fraud class (index 1)
    word_weights = lime_exp.as_list(label=1)
    word_weights = sorted(word_weights, key=lambda x: abs(x[1]), reverse=True)

    # ── Build LIME bar chart ──
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    words = [w[0] for w in word_weights][::-1]
    weights = [w[1] for w in word_weights][::-1]
    colors = ["#FF4B4B" if w > 0 else "#00CC88" for w in weights]

    ax.barh(words, weights, color=colors, alpha=0.85, height=0.6)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("LIME Weight (positive = fraud signal, negative = safe signal)", color="#CCCCCC", fontsize=9)
    ax.set_title("Word-Level Explanation — LIME", color="#FFFFFF", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#CCCCCC", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333")
    ax.spines["bottom"].set_color("#333")

    plt.tight_layout()

    return {
        "word_weights": word_weights,
        "fig": fig,
        "lime_exp": lime_exp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Word highlighting for Streamlit
# ─────────────────────────────────────────────────────────────────────────────

def highlight_fraud_words(text: str, word_weights: list) -> str:
    """
    Returns HTML with fraud words highlighted in red/orange,
    and safe words in green. Used in Streamlit UI.
    
    Args:
        text: Raw email text
        word_weights: List of (word, weight) from LIME
    """
    import html

    # Build lookup: word -> weight
    word_map = {w.lower(): score for w, score in word_weights}

    words = text.split()
    highlighted = []
    for word in words:
        clean_w = word.strip(".,!?;:\"'()").lower()
        score = word_map.get(clean_w, 0.0)

        if score > 0.1:
            intensity = min(int(score * 300), 255)
            color = f"rgb(255, {max(50, 150 - intensity)}, {max(50, 100 - intensity)})"
            highlighted.append(
                f'<mark style="background-color:{color};color:#fff;padding:1px 3px;'
                f'border-radius:3px;font-weight:600">{html.escape(word)}</mark>'
            )
        elif score < -0.05:
            highlighted.append(
                f'<mark style="background-color:#004D33;color:#00FF88;padding:1px 3px;'
                f'border-radius:3px">{html.escape(word)}</mark>'
            )
        else:
            highlighted.append(html.escape(word))

    return " ".join(highlighted)


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> plt.Figure:
    """Returns a styled confusion matrix figure."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    im = ax.imshow(cm, cmap="Reds")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Safe", "Fraud"], color="#CCCCCC")
    ax.set_yticklabels(["Safe", "Fraud"], color="#CCCCCC")
    ax.set_xlabel("Predicted", color="#CCCCCC")
    ax.set_ylabel("Actual", color="#CCCCCC")
    ax.set_title(title, color="#FFFFFF", fontsize=11, fontweight="bold")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "#CCCCCC",
                fontsize=14, fontweight="bold",
            )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ROC curve plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_proba, model_name: str = "Model") -> plt.Figure:
    """Returns a styled ROC curve figure."""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    ax.plot(fpr, tpr, color="#FF4B4B", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#444", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate", color="#CCCCCC")
    ax.set_ylabel("True Positive Rate", color="#CCCCCC")
    ax.set_title(f"ROC Curve — {model_name}", color="#FFFFFF", fontsize=11, fontweight="bold")
    ax.legend(facecolor="#1A1A2E", labelcolor="#CCCCCC", fontsize=9)
    ax.tick_params(colors="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333")
    ax.spines["bottom"].set_color("#333")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("explain.py — run via app.py or predict.py")
    print("SHAP and LIME explainers require trained models.")
    print("Run 'python src/train.py' first, then use via Streamlit.")
