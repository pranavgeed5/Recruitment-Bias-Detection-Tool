# Recruitment Bias Detection Tool
# Build Model + Evaluate + Save Model Only

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("Dataset.csv")

# Encode categorical columns
le_gender = LabelEncoder()
le_education = LabelEncoder()

df["gender"] = le_gender.fit_transform(df["gender"])
df["education_level"] = le_education.fit_transform(df["education_level"])

# Features and Target
X = df.drop("shortlisted", axis=1)
y = df["shortlisted"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save Model
joblib.dump(model, "recruitment_bias_model.pkl")
joblib.dump(le_gender, "gender_encoder.pkl")
joblib.dump(le_education, "education_encoder.pkl")

print("\nModel and encoders saved successfully!")