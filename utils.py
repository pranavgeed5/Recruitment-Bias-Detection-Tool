"""
utils.py — Bias detection logic, word highlighting, LLM/rule-based explanation
"""

import re
import pickle
import numpy as np
from preprocessing import preprocess_text, load_vectorizer

# ── Bias keyword lexicon ──────────────────────────────────────────────────────
BIAS_KEYWORDS = {
    "gender": {
        "male"   : ["he ","him ","his ","himself","male","man ","men ","gentleman",
                     "brotherhood","paternal","fatherhood","manpower"],
        "female" : ["she ","her ","hers ","herself","female","woman ","women ",
                     "lady","ladies","maternal","motherhood","manpower"],
    },
    "age": {
        "old"    : ["seasoned","veteran","over 40","mature","decades of experience",
                     "long career","established career","retired","experienced professional"],
        "young"  : ["recent graduate","fresh graduate","young ","entry-level",
                     "new to the field","just graduated","millennial","gen z",
                     "young professional"],
    },
    "name": {
        "male_names"  : ["james","john","robert","michael","david","william",
                          "richard","joseph","thomas","charles","daniel","matthew"],
        "female_names": ["mary","patricia","jennifer","linda","barbara","elizabeth",
                          "susan","jessica","sarah","karen","lisa","nancy","betty"],
    },
}

# Flat list for quick lookup
ALL_BIAS_WORDS: dict[str, str] = {}  # word → bias_category
for category, sub in BIAS_KEYWORDS.items():
    for _, words in sub.items():
        for w in words:
            ALL_BIAS_WORDS[w.strip().lower()] = category


def find_bias_words(text: str) -> dict[str, list[str]]:
    """Return {category: [matched_words]} found in text."""
    lower = text.lower()
    found: dict[str, list[str]] = {"gender": [], "age": [], "name": []}

    for category, sub in BIAS_KEYWORDS.items():
        for _, words in sub.items():
            for w in words:
                w_clean = w.strip()
                if w_clean in lower:
                    if w_clean not in found[category]:
                        found[category].append(w_clean)
    return found


def highlight_bias_words(text: str, found: dict[str, list[str]]) -> str:
    """Return text with bias words wrapped in **markers** for the UI."""
    highlighted = text
    all_words = []
    for words in found.values():
        all_words.extend(words)
    # Sort longest first to avoid partial replacements
    all_words.sort(key=len, reverse=True)
    for w in all_words:
        pattern = re.compile(re.escape(w), re.IGNORECASE)
        highlighted = pattern.sub(f"**{w.upper()}**", highlighted)
    return highlighted


# ── Model-based prediction ────────────────────────────────────────────────────
def load_model(path: str = "model_best.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_bias(text: str, model, vectorizer) -> tuple[float, int]:
    """Returns (probability_biased 0-1, label 0/1)."""
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    prob  = model.predict_proba(vec)[0][1]
    label = int(model.predict(vec)[0])
    return prob, label


# ── Combined bias score ───────────────────────────────────────────────────────
def compute_bias_score(text: str, model=None, vectorizer=None) -> dict:
    """
    Returns a dict with:
      bias_score (0-100), found_words, highlighted_text,
      gender_bias, age_bias, name_bias, model_prob
    """
    found = find_bias_words(text)
    highlighted = highlight_bias_words(text, found)

    # Keyword score component (0–60)
    total_hits = sum(len(v) for v in found.values())
    keyword_score = min(total_hits * 12, 60)

    # Model score component (0–40)
    model_prob = 0.0
    if model and vectorizer:
        try:
            model_prob, _ = predict_bias(text, model, vectorizer)
        except Exception:
            model_prob = 0.0
    model_score = model_prob * 40

    bias_score = round(min(keyword_score + model_score, 100))

    return {
        "bias_score"   : bias_score,
        "found_words"  : found,
        "highlighted"  : highlighted,
        "gender_bias"  : len(found["gender"]) > 0,
        "age_bias"     : len(found["age"]) > 0,
        "name_bias"    : len(found["name"]) > 0,
        "model_prob"   : round(model_prob * 100, 1),
    }


# ── Rule-based explanation (LLM fallback) ────────────────────────────────────
SUGGESTIONS = {
    "gender": (
        "Replace gender-specific pronouns (he/she/him/her) with "
        "neutral alternatives like 'they/them' or rewrite sentences "
        "to avoid pronouns entirely."
    ),
    "age": (
        "Remove age indicators like 'recent graduate', 'seasoned veteran', or "
        "'decades of experience'. Focus on specific skills and measurable achievements instead."
    ),
    "name": (
        "Anonymize or use initials instead of full names during screening. "
        "Name-based bias (gender/ethnicity inference from names) is a common unconscious bias."
    ),
}

def generate_explanation(result: dict, jd_text: str = "") -> dict:
    """
    Returns {explanation: str, suggestions: list[str]}
    Rule-based — no external API needed.
    """
    score  = result["bias_score"]
    found  = result["found_words"]
    active = [k for k, v in found.items() if v]

    if score == 0:
        explanation = (
            "No significant bias detected in this resume. "
            "The text appears to use neutral, professional language."
        )
        suggestions = ["Continue using inclusive, skills-focused language."]
    elif score < 30:
        explanation = (
            f"Low bias detected (score: {score}/100). "
            f"Minor indicators found in: {', '.join(active) or 'none'}. "
            "These may not significantly impact hiring decisions but are worth reviewing."
        )
        suggestions = [SUGGESTIONS[k] for k in active if k in SUGGESTIONS]
    elif score < 60:
        explanation = (
            f"Moderate bias detected (score: {score}/100). "
            f"Bias indicators found: {', '.join(active)}. "
            "These could unconsciously influence a recruiter's decision."
        )
        suggestions = [SUGGESTIONS[k] for k in active if k in SUGGESTIONS]
    else:
        explanation = (
            f"High bias detected (score: {score}/100). "
            f"Multiple bias signals found across: {', '.join(active)}. "
            "This resume is at significant risk of triggering unconscious bias in screening."
        )
        suggestions = [SUGGESTIONS[k] for k in active if k in SUGGESTIONS]
        suggestions.append(
            "Consider a blind review process — remove name, pronouns, "
            "and age indicators before evaluation."
        )

    return {"explanation": explanation, "suggestions": suggestions or ["No specific suggestions."]}
