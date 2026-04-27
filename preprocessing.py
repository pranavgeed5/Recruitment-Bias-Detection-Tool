"""
preprocessing.py — Text cleaning, tokenization, TF-IDF vectorization
No NLTK required — uses regex tokenization with built-in stopword list.
"""

import re
import string
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

# Built-in English stopwords (no NLTK needed)
STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","up","about","into","through","during","before","after",
    "above","below","between","out","off","over","under","again","then",
    "once","here","there","when","where","why","how","all","both","each",
    "few","more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","s","t","can","will","just","don","should",
    "now","d","ll","m","o","re","ve","y","ain","aren","couldn","didn","doesn",
    "hadn","hasn","haven","isn","ma","mightn","mustn","needn","shan","shouldn",
    "wasn","weren","won","wouldn","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","shall","should","may",
    "might","must","can","could","it","its","this","that","these","those","i",
    "me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","they","them","their","theirs","themselves","what","which","who",
    "whom","am","if","as","while","because","although","though","since",
}

# ── Core cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/digits, strip extra spaces."""
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-z]{3,}\b", text)
    return [t for t in tokens if t not in STOP_WORDS]


def preprocess_text(text: str) -> str:
    """Full pipeline: clean → tokenize → rejoin."""
    cleaned = clean_text(text)
    tokens  = tokenize(cleaned)
    return " ".join(tokens)


# ── Vectorizer helpers ────────────────────────────────────────────────────────

def build_vectorizer(texts: list[str], max_features: int = 5000) -> TfidfVectorizer:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    vec.fit(texts)
    return vec


def save_vectorizer(vec: TfidfVectorizer, path: str = "vectorizer.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump(vec, f)


def load_vectorizer(path: str = "vectorizer.pkl") -> TfidfVectorizer:
    with open(path, "rb") as f:
        return pickle.load(f)
