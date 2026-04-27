"""
model.py — Train Logistic Regression + Random Forest, save models
Run once: python model.py
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from preprocessing import preprocess_text, build_vectorizer, save_vectorizer

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv("dataset.csv")
df["processed"] = df["resume_text"].apply(preprocess_text)

X_raw = df["processed"].values
y     = df["biased"].values

# ── Vectorize ─────────────────────────────────────────────────────────────────
vectorizer = build_vectorizer(X_raw.tolist())
X = vectorizer.transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model 1 : Logistic Regression ────────────────────────────────────────────
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
print("=== Logistic Regression ===")
print(classification_report(y_test, lr_preds))

# ── Model 2 : Random Forest ───────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("=== Random Forest ===")
print(classification_report(y_test, rf_preds))

# ── Pick best model ───────────────────────────────────────────────────────────
lr_acc = accuracy_score(y_test, lr_preds)
rf_acc = accuracy_score(y_test, rf_preds)
best   = rf if rf_acc >= lr_acc else lr
print(f"Best model: {'RandomForest' if rf_acc >= lr_acc else 'LogisticRegression'} "
      f"(acc={max(lr_acc,rf_acc):.3f})")

# ── Save ──────────────────────────────────────────────────────────────────────
with open("model_lr.pkl",   "wb") as f: pickle.dump(lr, f)
with open("model_rf.pkl",   "wb") as f: pickle.dump(rf, f)
with open("model_best.pkl", "wb") as f: pickle.dump(best, f)
save_vectorizer(vectorizer, "vectorizer.pkl")
print("✅ Models and vectorizer saved.")
