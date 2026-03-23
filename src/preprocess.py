"""
preprocess.py
-------------
Layer B: NLP Processing Engine
Handles email text cleaning, tokenization, lemmatization, and entity extraction.
"""

import re
import string
import nltk
import spacy
import pandas as pd
import numpy as np
from typing import Optional

# Download required NLTK data (run once)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load spaCy model (run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ─────────────────────────────────────────────────────────────────────────────
# Core text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Full NLP pipeline:
      1. Lowercase
      2. Remove HTML tags
      3. Remove URLs (preserve count via features.py)
      4. Remove email addresses
      5. Remove punctuation / special characters
      6. Tokenize
      7. Remove stopwords
      8. Lemmatize
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " url_token ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " email_token ", text)

    # Remove punctuation except useful signals (keep $ % for fraud signals)
    text = re.sub(r"[^a-z0-9\s\$\%]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize + stopword removal + lemmatization
    tokens = text.split()
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 1
    ]

    return " ".join(tokens)


def extract_entities(text: str) -> dict:
    """
    Use spaCy to extract named entities from raw email text.
    Returns counts of ORG, MONEY, PERSON, GPE entities.
    Useful for detecting impersonation (fake org names).
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"org_count": 0, "money_count": 0, "person_count": 0, "gpe_count": 0}

    # Limit length to avoid spaCy timeout on very long emails
    doc = nlp(text[:5000])

    entity_counts = {"org_count": 0, "money_count": 0, "person_count": 0, "gpe_count": 0}
    label_map = {
        "ORG": "org_count",
        "MONEY": "money_count",
        "PERSON": "person_count",
        "GPE": "gpe_count",
    }

    for ent in doc.ents:
        if ent.label_ in label_map:
            entity_counts[label_map[ent.label_]] += 1

    return entity_counts


# ─────────────────────────────────────────────────────────────────────────────
# Sender / domain analysis
# ─────────────────────────────────────────────────────────────────────────────

# Common spoofed brand names — attackers substitute letters (paypa1, amaz0n)
BRAND_KEYWORDS = [
    "paypal", "amazon", "google", "microsoft", "apple", "netflix",
    "facebook", "instagram", "twitter", "bank", "chase", "wellsfargo",
    "citibank", "irs", "fedex", "dhl", "ups",
]

SUSPICIOUS_TLD = [".xyz", ".top", ".click", ".loan", ".work", ".club", ".info"]


def analyze_sender(sender: str) -> dict:
    """
    Extracts fraud signals from the sender email address.
    Detects:
      - Domain/display name mismatch
      - Spoofed brand domains (paypa1.com, amaz0n-security.com)
      - Suspicious TLDs
      - Numeric substitutions in domain
    """
    features = {
        "sender_has_brand_spoof": 0,
        "sender_suspicious_tld": 0,
        "sender_numeric_substitution": 0,
        "sender_domain_length": 0,
        "sender_subdomain_count": 0,
    }

    if not isinstance(sender, str):
        return features

    sender = sender.lower().strip()

    # Extract domain part
    domain_match = re.search(r"@([\w\.\-]+)", sender)
    if not domain_match:
        return features

    domain = domain_match.group(1)
    features["sender_domain_length"] = len(domain)
    features["sender_subdomain_count"] = domain.count(".")

    # Check for suspicious TLD
    for tld in SUSPICIOUS_TLD:
        if domain.endswith(tld):
            features["sender_suspicious_tld"] = 1
            break

    # Detect numeric substitutions (0→o, 1→l/i, 3→e, etc.)
    # Classic spoof: paypa1.com, amaz0n.com
    normalized = (
        domain.replace("0", "o")
              .replace("1", "i")
              .replace("3", "e")
              .replace("4", "a")
              .replace("5", "s")
              .replace("@", "a")
    )
    for brand in BRAND_KEYWORDS:
        if brand in normalized and brand not in domain:
            features["sender_has_brand_spoof"] = 1
            features["sender_numeric_substitution"] = 1
            break

    # Direct brand name in suspicious context (e.g. paypal-security.com)
    for brand in BRAND_KEYWORDS:
        if brand in domain:
            # If brand appears with extra text after hyphen/dot it's likely spoof
            pattern = rf"{brand}[\-\.]"
            if re.search(pattern, domain):
                features["sender_has_brand_spoof"] = 1

    return features


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame-level preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies full preprocessing to a DataFrame with columns:
      - subject, sender, body, label (0=safe, 1=fraud)
    
    Returns enriched DataFrame with:
      - cleaned_text: combined & cleaned email text
      - Sender analysis features
      - Entity counts
    """
    df = df.copy()

    # Fill missing values
    df["subject"] = df["subject"].fillna("")
    df["sender"] = df["sender"].fillna("")
    df["body"] = df["body"].fillna("")

    # Combine subject + body for NLP (subject is high signal)
    df["raw_text"] = df["subject"] + " " + df["body"]
    df["cleaned_text"] = df["raw_text"].apply(clean_text)

    # Sender analysis
    sender_features = df["sender"].apply(analyze_sender).apply(pd.Series)
    df = pd.concat([df, sender_features], axis=1)

    # Entity extraction (slower — uses spaCy)
    print("Extracting named entities (this may take a moment)...")
    entity_features = df["raw_text"].apply(extract_entities).apply(pd.Series)
    df = pd.concat([df, entity_features], axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Single email preprocessing (for inference)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_email(
    subject: str,
    sender: str,
    body: str,
) -> dict:
    """
    Preprocesses a single email for inference.
    Returns a dict of all raw signals needed by features.py.
    """
    raw_text = f"{subject} {body}"
    cleaned = clean_text(raw_text)
    sender_feats = analyze_sender(sender)
    entity_feats = extract_entities(raw_text)

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "subject": subject,
        "sender": sender,
        "body": body,
        **sender_feats,
        **entity_feats,
    }


if __name__ == "__main__":
    # Quick sanity check
    test_email = {
        "subject": "URGENT: Your account will be suspended!",
        "sender": "support@paypa1-security.com",
        "body": (
            "Dear Customer, please verify your password immediately. "
            "Click here: http://fake-link.xyz/verify to avoid suspension. "
            "Your bank account requires urgent attention. Wire $500 NOW."
        ),
    }

    result = preprocess_email(**test_email)
    print("=== Preprocessed Email ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
