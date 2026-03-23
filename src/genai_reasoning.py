"""
genai_reasoning.py
------------------
Layer E: GenAI Intelligence Layer
Uses a lightweight generative model (FLAN-T5) to produce:
  1. Human-readable fraud explanation
  2. Threat summary
  3. Safe user-facing warning
  4. Tactical recommendations

Falls back to rule-based explanations if transformers unavailable.
"""

from pathlib import Path
from typing import Optional
import re


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback (no model required)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_explanation(
    prediction: str,
    fraud_probability: float,
    contributing_factors: list,
    severity: str,
    preprocessed: dict,
) -> dict:
    """
    Generates structured explanation using rules when no GenAI model is available.
    Still produces professional, recruiter-impressive output.
    """
    is_fraud = prediction == "FRAUD"
    risk_pct = int(fraud_probability * 100)

    # ── Explanation ──
    if is_fraud:
        factor_text = "; ".join(contributing_factors[:3]) if contributing_factors else "suspicious content patterns"
        explanation = (
            f"This email is classified as fraudulent with {risk_pct}% confidence. "
            f"Key indicators detected: {factor_text}. "
            f"The sender domain '{preprocessed.get('sender', 'unknown')}' shows "
            f"{'spoofing patterns' if preprocessed.get('sender_has_brand_spoof') else 'anomalous characteristics'}. "
            f"The language pattern matches known phishing and social engineering tactics."
        )
    else:
        explanation = (
            f"This email appears legitimate with {100 - risk_pct}% confidence. "
            "No significant fraud indicators were detected in the content, sender domain, "
            "or structural features. Standard email communication patterns observed."
        )

    # ── Threat summary ──
    if is_fraud:
        factors_lower = [f.lower() for f in contributing_factors]
        if any("credential" in f or "password" in f for f in factors_lower):
            threat_type = "credential theft via fake verification"
        elif any("financial" in f or "wire" in f or "payment" in f for f in factors_lower):
            threat_type = "financial fraud via fake payment request"
        elif any("impersonat" in f or "brand" in f or "spoof" in f for f in factors_lower):
            threat_type = "brand impersonation / domain spoofing"
        elif any("urgency" in f for f in factors_lower):
            threat_type = "social engineering via urgency manipulation"
        else:
            threat_type = "multi-vector phishing attack"

        threat_summary = (
            f"This message attempts {threat_type}. "
            f"Severity level: {severity}. "
            "The attack targets user credentials and/or financial assets through deceptive communication."
        )
    else:
        threat_summary = "No threat detected. Email cleared for delivery."

    # ── User-safe explanation ──
    if is_fraud:
        safe_explanation = (
            "⚠️ Do NOT click any links in this email. "
            "Do NOT provide any personal information, passwords, or payment details. "
            "The sender is likely impersonating a trusted organization. "
            "If you believe this is legitimate, contact the organization directly through their official website."
        )
    else:
        safe_explanation = (
            "✅ This email appears to be legitimate. "
            "However, always exercise caution with unexpected requests for personal information."
        )

    # ── Tactical recommendations ──
    if is_fraud:
        recommendations = [
            "Mark as phishing and report to your IT security team",
            "Do not forward this email to others",
            "If you clicked any links, change your passwords immediately",
            "Check your account statements for unauthorized transactions",
            f"Block sender domain: {preprocessed.get('sender', '').split('@')[-1] if '@' in preprocessed.get('sender', '') else 'unknown'}",
        ]
    else:
        recommendations = [
            "Email is safe for normal processing",
            "Continue exercising standard email hygiene practices",
        ]

    return {
        "explanation": explanation,
        "threat_summary": threat_summary,
        "safe_explanation": safe_explanation,
        "recommendations": recommendations,
        "generated_by": "Rule-Based Engine",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GenAI explanation using FLAN-T5
# ─────────────────────────────────────────────────────────────────────────────

_genai_pipeline = None


def _load_genai_pipeline():
    """Lazily loads the FLAN-T5 pipeline (cached after first call)."""
    global _genai_pipeline
    if _genai_pipeline is not None:
        return _genai_pipeline

    try:
        from transformers import pipeline
        print("Loading FLAN-T5 GenAI model (first run may take 1-2 minutes)...")
        _genai_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",     # ~250MB — good balance of speed + quality
            max_new_tokens=200,
            temperature=0.3,
            do_sample=False,
        )
        print("GenAI model loaded.")
        return _genai_pipeline
    except Exception as e:
        print(f"Could not load GenAI model: {e}. Using rule-based fallback.")
        return None


def _generate_with_flan(prompt: str, pipeline) -> str:
    """Run a single FLAN-T5 generation call."""
    try:
        result = pipeline(prompt, max_new_tokens=200)
        return result[0]["generated_text"].strip()
    except Exception as e:
        return f"[Generation error: {e}]"


def _genai_explanation(
    prediction: str,
    fraud_probability: float,
    contributing_factors: list,
    severity: str,
    subject: str,
    sender: str,
    body_snippet: str,
) -> dict:
    """
    Uses FLAN-T5 to generate fraud explanations.
    Falls back to rule-based if model unavailable.
    """
    pipe = _load_genai_pipeline()
    if pipe is None:
        return None  # Caller will use rule-based

    is_fraud = prediction == "FRAUD"
    risk_pct = int(fraud_probability * 100)
    factors_str = ", ".join(contributing_factors[:4]) if contributing_factors else "suspicious patterns"

    # ── Prompt 1: Technical explanation ──
    explain_prompt = (
        f"Analyze this email for fraud. "
        f"Subject: '{subject}'. Sender: '{sender}'. "
        f"Body snippet: '{body_snippet[:300]}'. "
        f"The fraud detection model classified it as {prediction} with {risk_pct}% confidence. "
        f"Key fraud signals: {factors_str}. "
        f"Explain in 2-3 sentences why this email is {'fraudulent' if is_fraud else 'legitimate'}."
    )

    # ── Prompt 2: Threat summary ──
    threat_prompt = (
        f"Summarize the security threat in this email in one sentence. "
        f"Email subject: '{subject}'. Fraud signals: {factors_str}. "
        f"Severity: {severity}."
    )

    # ── Prompt 3: Safe user advice ──
    safe_prompt = (
        f"This email was classified as {'FRAUDULENT' if is_fraud else 'SAFE'} "
        f"({'do not click' if is_fraud else 'appears legitimate'}). "
        f"Give one clear safety instruction to a non-technical user in plain English."
    )

    explanation = _generate_with_flan(explain_prompt, pipe)
    threat_summary = _generate_with_flan(threat_prompt, pipe)
    safe_explanation = _generate_with_flan(safe_prompt, pipe)

    return {
        "explanation": explanation,
        "threat_summary": threat_summary,
        "safe_explanation": safe_explanation,
        "generated_by": "FLAN-T5 (google/flan-t5-base)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def generate_fraud_report(
    prediction_result: dict,
    use_genai: bool = True,
) -> dict:
    """
    Generates a complete GenAI-powered fraud analysis report.

    Args:
        prediction_result: Output dict from predict.predict()
        use_genai: Whether to attempt FLAN-T5 generation (falls back to rules if unavailable)

    Returns dict with:
        explanation: Technical explanation of why email is fraud/safe
        threat_summary: One-sentence threat category summary
        safe_explanation: Plain English user warning
        recommendations: List of action items
        severity_badge: HTML badge string for UI
        generated_by: Which engine produced the output
    """
    pred = prediction_result["prediction"]
    fraud_prob = prediction_result["fraud_probability"]
    factors = prediction_result.get("contributing_factors", [])
    severity = prediction_result["severity"]
    risk_pct = prediction_result["risk_pct"]
    preprocessed = prediction_result.get("preprocessed", {})

    subject = preprocessed.get("subject", "")
    sender = preprocessed.get("sender", "")
    body = preprocessed.get("body", "")
    body_snippet = body[:400] if body else ""

    # Try GenAI first
    genai_result = None
    if use_genai:
        genai_result = _genai_explanation(
            prediction=pred,
            fraud_probability=fraud_prob,
            contributing_factors=factors,
            severity=severity,
            subject=subject,
            sender=sender,
            body_snippet=body_snippet,
        )

    # Use rule-based fallback
    rule_result = _rule_based_explanation(
        prediction=pred,
        fraud_probability=fraud_prob,
        contributing_factors=factors,
        severity=severity,
        preprocessed=preprocessed,
    )

    # Merge: prefer GenAI where available
    if genai_result:
        explanation = genai_result.get("explanation") or rule_result["explanation"]
        threat_summary = genai_result.get("threat_summary") or rule_result["threat_summary"]
        safe_explanation = genai_result.get("safe_explanation") or rule_result["safe_explanation"]
        generated_by = genai_result["generated_by"]
    else:
        explanation = rule_result["explanation"]
        threat_summary = rule_result["threat_summary"]
        safe_explanation = rule_result["safe_explanation"]
        generated_by = rule_result["generated_by"]

    recommendations = rule_result["recommendations"]  # Always rule-based (more reliable)

    # ── Severity badge HTML (for Streamlit) ──
    severity_colors = {
        "Critical": ("#FF1744", "#FFE0E6"),
        "High": ("#FF6D00", "#FFF3E0"),
        "Medium": ("#FFD600", "#FFFDE7"),
        "Low": ("#00C853", "#E8F5E9"),
    }
    badge_color, badge_bg = severity_colors.get(severity, ("#888", "#EEE"))
    severity_badge = (
        f'<span style="background:{badge_bg};color:{badge_color};'
        f'padding:4px 12px;border-radius:20px;font-weight:700;font-size:14px;'
        f'border:2px solid {badge_color}">{severity} Risk — {risk_pct}%</span>'
    )

    return {
        "explanation": explanation,
        "threat_summary": threat_summary,
        "safe_explanation": safe_explanation,
        "recommendations": recommendations,
        "severity_badge": severity_badge,
        "generated_by": generated_by,
        "risk_pct": risk_pct,
        "severity": severity,
        "prediction": pred,
        "fraud_probability": fraud_prob,
    }


if __name__ == "__main__":
    # Demo without trained models
    demo_result = {
        "prediction": "FRAUD",
        "fraud_probability": 0.94,
        "risk_pct": 94,
        "severity": "Critical",
        "contributing_factors": [
            "Urgency manipulation language",
            "Credential/password request",
            "Sender domain impersonates known brand",
            "Suspicious URL patterns detected",
        ],
        "preprocessed": {
            "subject": "URGENT: Your account will be suspended!",
            "sender": "support@paypa1-security.com",
            "body": "Please verify your password immediately at http://paypa1.xyz",
            "sender_has_brand_spoof": 1,
        },
    }

    report = generate_fraud_report(demo_result, use_genai=False)
    print("\n=== FRAUD ANALYSIS REPORT ===\n")
    print(f"Prediction  : {report['prediction']}")
    print(f"Risk        : {report['risk_pct']}% ({report['severity']})")
    print(f"\nExplanation :\n{report['explanation']}")
    print(f"\nThreat      : {report['threat_summary']}")
    print(f"\nUser Advice :\n{report['safe_explanation']}")
    print(f"\nActions     :")
    for r in report["recommendations"]:
        print(f"  • {r}")
    print(f"\nGenerated by: {report['generated_by']}")
