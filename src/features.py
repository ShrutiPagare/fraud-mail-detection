"""
features.py
-----------
Layer C: Feature Engineering Engine
Creates rich feature sets from preprocessed email data:
  1. TF-IDF text features
  2. Fraud signal word counts (urgency, money, credential keywords)
  3. Structural features (link count, uppercase ratio, symbol density)
  4. Domain/sender features (from preprocess.py)
  5. Statistical text features
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix
import joblib
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Fraud signal word lists (curated for phishing/fraud email patterns)
# ─────────────────────────────────────────────────────────────────────────────

URGENCY_WORDS = [
    "urgent", "immediately", "asap", "now", "today", "expires",
    "deadline", "limited", "act", "quickly", "hurry", "last chance",
    "warning", "alert", "suspended", "terminated", "blocked",
    "critical", "important", "attention", "final", "notice",
]

MONEY_WORDS = [
    "payment", "wire", "transfer", "bank", "account", "invoice",
    "money", "fund", "dollar", "usd", "cash", "credit", "debit",
    "deposit", "withdraw", "refund", "prize", "won", "lottery",
    "inheritance", "million", "billion", "free", "earn",
]

CREDENTIAL_WORDS = [
    "password", "verify", "confirm", "login", "username",
    "credential", "authenticate", "sign in", "click here",
    "update", "validate", "secure", "access", "unlock",
    "pin", "otp", "code", "account",
]

THREAT_WORDS = [
    "suspend", "terminate", "close", "delete", "banned",
    "hack", "breach", "compromised", "unauthorized", "illegal",
    "legal", "action", "report", "police", "irs", "tax",
]

ALL_FRAUD_SIGNALS = set(
    URGENCY_WORDS + MONEY_WORDS + CREDENTIAL_WORDS + THREAT_WORDS
)


# ─────────────────────────────────────────────────────────────────────────────
# Structural feature extraction (raw text input)
# ─────────────────────────────────────────────────────────────────────────────

def extract_structural_features(raw_text: str, subject: str = "") -> dict:
    """
    Extracts structural signals that don't rely on NLP cleaning.
    These are powerful fraud indicators on their own.
    """
    text = raw_text if isinstance(raw_text, str) else ""
    subject = subject if isinstance(subject, str) else ""
    full_text = f"{subject} {text}"

    # URL / link count
    url_count = len(re.findall(r"http\S+|www\.\S+", full_text))

    # Suspicious URL patterns (IP-based, very long, encoded)
    suspicious_url_count = len(re.findall(
        r"http\S*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|\.xyz|\.top|\.click|%[0-9a-f]{2})",
        full_text, re.IGNORECASE
    ))

    # Uppercase ratio (SHOUTING is a fraud signal)
    alpha_chars = [c for c in full_text if c.isalpha()]
    uppercase_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars else 0.0
    )

    # Special symbol density ($$, !!!, ???)
    dollar_count = full_text.count("$")
    exclamation_count = full_text.count("!")
    question_count = full_text.count("?")

    # ALL CAPS words count
    all_caps_words = len(re.findall(r"\b[A-Z]{3,}\b", full_text))

    # Email address count (multiple emails = bulk/spam signal)
    email_count = len(re.findall(r"\b[\w._%+\-]+@[\w.\-]+\.[a-z]{2,}\b", full_text))

    # Character count and word count
    char_count = len(text)
    word_count = len(text.split())

    # Average word length
    words = text.split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0

    # Digit ratio
    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / char_count if char_count > 0 else 0.0

    # HTML indicators in plaintext (could mean stripped HTML)
    html_indicator = int(bool(re.search(r"&[a-z]+;|<br|<p|&nbsp", text, re.IGNORECASE)))

    return {
        "url_count": url_count,
        "suspicious_url_count": suspicious_url_count,
        "uppercase_ratio": round(uppercase_ratio, 4),
        "dollar_count": dollar_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "all_caps_words": all_caps_words,
        "email_count": email_count,
        "char_count": char_count,
        "word_count": word_count,
        "avg_word_len": round(avg_word_len, 4),
        "digit_ratio": round(digit_ratio, 4),
        "html_indicator": html_indicator,
    }


def count_fraud_signals(text: str) -> dict:
    """
    Counts occurrences of fraud signal words across all categories.
    Works on raw (uncleaned) text for maximum signal capture.
    """
    text_lower = text.lower() if isinstance(text, str) else ""

    urgency_score = sum(text_lower.count(w) for w in URGENCY_WORDS)
    money_score = sum(text_lower.count(w) for w in MONEY_WORDS)
    credential_score = sum(text_lower.count(w) for w in CREDENTIAL_WORDS)
    threat_score = sum(text_lower.count(w) for w in THREAT_WORDS)
    total_fraud_signal = urgency_score + money_score + credential_score + threat_score

    return {
        "urgency_score": urgency_score,
        "money_score": money_score,
        "credential_score": credential_score,
        "threat_score": threat_score,
        "total_fraud_signal": total_fraud_signal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF vectorizer configuration
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_vectorizer(
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
) -> TfidfVectorizer:
    """
    Builds a TF-IDF vectorizer with n-grams.
    Bigrams capture patterns like 'click here', 'verify account', 'urgent payment'.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,          # Apply log normalization (reduces impact of high freq terms)
        min_df=2,                   # Ignore very rare terms
        max_df=0.95,                # Ignore overly common terms
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-z][a-z]+\b",  # Only alphabetic tokens
    )


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngineer class — main interface
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Full feature engineering pipeline.
    Combines TF-IDF text features with structural + fraud signal features.
    
    Usage:
        fe = FeatureEngineer()
        X_train = fe.fit_transform(df_train)
        X_test  = fe.transform(df_test)
    """

    def __init__(self, max_tfidf_features: int = 10000, ngram_range: tuple = (1, 2)):
        self.tfidf = build_tfidf_vectorizer(max_tfidf_features, ngram_range)
        self.is_fitted = False
        self._feature_names_hand = None  # Handcrafted feature names

    def _get_handcrafted(self, df: pd.DataFrame) -> np.ndarray:
        """
        Assembles all non-TF-IDF features into a numpy matrix.
        """
        rows = []
        for _, row in df.iterrows():
            raw = row.get("raw_text", "")
            subj = row.get("subject", "")

            structural = extract_structural_features(raw, subj)
            fraud_signals = count_fraud_signals(raw)

            # Sender features (already in df after preprocess_dataframe)
            sender_feats = {
                "sender_has_brand_spoof": row.get("sender_has_brand_spoof", 0),
                "sender_suspicious_tld": row.get("sender_suspicious_tld", 0),
                "sender_numeric_substitution": row.get("sender_numeric_substitution", 0),
                "sender_domain_length": row.get("sender_domain_length", 0),
                "sender_subdomain_count": row.get("sender_subdomain_count", 0),
            }

            # Entity features
            entity_feats = {
                "org_count": row.get("org_count", 0),
                "money_count": row.get("money_count", 0),
                "person_count": row.get("person_count", 0),
                "gpe_count": row.get("gpe_count", 0),
            }

            combined = {**structural, **fraud_signals, **sender_feats, **entity_feats}
            rows.append(combined)

        hand_df = pd.DataFrame(rows)
        if self._feature_names_hand is None:
            self._feature_names_hand = list(hand_df.columns)

        return hand_df.values.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        """Fit on training data and return combined feature matrix."""
        tfidf_matrix = self.tfidf.fit_transform(df["cleaned_text"].fillna(""))
        hand_matrix = self._get_handcrafted(df)
        self.is_fitted = True
        return hstack([tfidf_matrix, csr_matrix(hand_matrix)])

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """Transform new data using fitted TF-IDF."""
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer not fitted. Call fit_transform first.")
        tfidf_matrix = self.tfidf.transform(df["cleaned_text"].fillna(""))
        hand_matrix = self._get_handcrafted(df)
        return hstack([tfidf_matrix, csr_matrix(hand_matrix)])

    def transform_single(self, preprocessed_email: dict) -> csr_matrix:
        """
        Transform a single preprocessed email dict (from preprocess.preprocess_email).
        Used during inference.
        """
        row = pd.Series(preprocessed_email)
        df = pd.DataFrame([row])
        return self.transform(df)

    def get_feature_names(self) -> list:
        """Returns all feature names (TF-IDF + handcrafted)."""
        tfidf_names = self.tfidf.get_feature_names_out().tolist()
        hand_names = self._feature_names_hand or []
        return tfidf_names + hand_names

    def save(self, path: str):
        """Persist the fitted FeatureEngineer."""
        joblib.dump(self, path)
        print(f"FeatureEngineer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureEngineer":
        """Load a persisted FeatureEngineer."""
        fe = joblib.load(path)
        print(f"FeatureEngineer loaded from {path}")
        return fe


# ─────────────────────────────────────────────────────────────────────────────
# Compute severity score for single email (used by predict.py / UI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_severity_score(
    fraud_prob: float,
    preprocessed: dict,
) -> dict:
    """
    Converts raw fraud probability + signal counts into a human-readable
    severity score with risk tier labeling.
    
    Returns:
        risk_pct: 0-100 integer
        severity: "Low" | "Medium" | "High" | "Critical"
        contributing_factors: list of strings
    """
    # Base score from model probability
    risk_pct = int(fraud_prob * 100)

    # Boost score based on strong structural signals
    structural = extract_structural_features(
        preprocessed.get("raw_text", ""), preprocessed.get("subject", "")
    )
    signals = count_fraud_signals(preprocessed.get("raw_text", ""))

    # Boost for high-confidence signals
    if preprocessed.get("sender_has_brand_spoof", 0):
        risk_pct = min(100, risk_pct + 8)
    if structural["suspicious_url_count"] > 0:
        risk_pct = min(100, risk_pct + 6)
    if signals["credential_score"] > 3:
        risk_pct = min(100, risk_pct + 5)

    # Severity tier
    if risk_pct >= 85:
        severity = "Critical"
    elif risk_pct >= 65:
        severity = "High"
    elif risk_pct >= 40:
        severity = "Medium"
    else:
        severity = "Low"

    # Contributing factors for UI display
    factors = []
    if signals["urgency_score"] > 2:
        factors.append("Urgency manipulation language")
    if signals["credential_score"] > 1:
        factors.append("Credential/password request")
    if signals["money_score"] > 2:
        factors.append("Financial transaction request")
    if signals["threat_score"] > 0:
        factors.append("Threat/suspension warning")
    if structural["suspicious_url_count"] > 0:
        factors.append("Suspicious URL patterns detected")
    if structural["uppercase_ratio"] > 0.3:
        factors.append("Excessive capitalization (SHOUTING)")
    if preprocessed.get("sender_has_brand_spoof", 0):
        factors.append("Sender domain impersonates known brand")
    if structural["url_count"] > 3:
        factors.append(f"Multiple links ({structural['url_count']})")

    if not factors:
        factors = ["General suspicious content pattern"]

    return {
        "risk_pct": risk_pct,
        "severity": severity,
        "contributing_factors": factors,
    }


if __name__ == "__main__":
    # Quick test of structural + fraud signal extraction
    sample_text = (
        "URGENT!! Your PayPal account has been SUSPENDED. "
        "Click http://paypa1-verify.xyz/secure to verify your password immediately. "
        "Wire $500 to restore access. This is your FINAL WARNING!"
    )

    structural = extract_structural_features(sample_text, "URGENT: Account Suspended")
    signals = count_fraud_signals(sample_text)

    print("=== Structural Features ===")
    for k, v in structural.items():
        print(f"  {k}: {v}")

    print("\n=== Fraud Signal Scores ===")
    for k, v in signals.items():
        print(f"  {k}: {v}")
