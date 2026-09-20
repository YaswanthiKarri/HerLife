import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "pcod_pcos_dataset.csv"
)


# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Rows:", len(df))


# -----------------------------
# Rename dataset columns
# -----------------------------

df = df.rename(columns={
    "Age": "age",
    "BMI": "bmi",
    "Cycle length": "cycle_length_days",
    "Period duration": "period_duration_days",
    "Irregular periods": "irregular_periods",
    "Acne": "acne",
    "Excess hair growth": "excess_hair_growth",
    "Hair thinning": "hair_thinning",
    "Weight gain": "weight_gain",
    "Dark skin patches": "dark_skin_patches",
    "Pelvic pain": "pelvic_pain",
    "Fatigue": "fatigue",
    "Family history": "family_history_pcos",
    "Physical activity": "physical_activity_days_per_week",
    "Sleep": "sleep_hours",
    "Stress": "stress_level_1_to_5",
    "Water intake": "water_glasses_per_day",
    "PCOS risk label": "pcos_risk_label",
    "PCOD risk label": "pcod_risk_label"
})


# -----------------------------
# Features
# -----------------------------

features = [
    "age",
    "bmi",
    "cycle_length_days",
    "period_duration_days",
    "irregular_periods",
    "acne",
    "excess_hair_growth",
    "hair_thinning",
    "weight_gain",
    "dark_skin_patches",
    "pelvic_pain",
    "fatigue",
    "family_history_pcos",
    "physical_activity_days_per_week",
    "sleep_hours",
    "stress_level_1_to_5",
    "water_glasses_per_day"
]


X = df[features]


# =========================================================
# PCOD MODEL
# =========================================================

print("\nTraining PCOD model...")

pcod_encoder = LabelEncoder()

y_pcod = pcod_encoder.fit_transform(
    df["pcod_risk_label"]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_pcod,
    test_size=0.20,
    random_state=42,
    stratify=y_pcod
)

pcod_model = RandomForestClassifier(
    n_estimators=60,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

pcod_model.fit(X_train, y_train)

pcod_prediction = pcod_model.predict(X_test)

pcod_accuracy = accuracy_score(
    y_test,
    pcod_prediction
)

print(
    "PCOD Accuracy:",
    round(pcod_accuracy * 100, 2),
    "%"
)


# Save compressed PCOD model
joblib.dump(
    pcod_model,
    os.path.join(BASE_DIR, "pcod_model.pkl"),
    compress=3
)

joblib.dump(
    pcod_encoder,
    os.path.join(BASE_DIR, "pcod_label_encoder.pkl"),
    compress=3
)

print("PCOD model saved successfully.")


# =========================================================
# PCOS MODEL
# =========================================================

print("\nTraining PCOS model...")

pcos_encoder = LabelEncoder()

y_pcos = pcos_encoder.fit_transform(
    df["pcos_risk_label"]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_pcos,
    test_size=0.20,
    random_state=42,
    stratify=y_pcos
)

pcos_model = RandomForestClassifier(
    n_estimators=60,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

pcos_model.fit(X_train, y_train)

pcos_prediction = pcos_model.predict(X_test)

pcos_accuracy = accuracy_score(
    y_test,
    pcos_prediction
)

print(
    "PCOS Accuracy:",
    round(pcos_accuracy * 100, 2),
    "%"
)


# Save compressed PCOS model
joblib.dump(
    pcos_model,
    os.path.join(BASE_DIR, "pcos_model.pkl"),
    compress=3
)

joblib.dump(
    pcos_encoder,
    os.path.join(BASE_DIR, "pcos_label_encoder.pkl"),
    compress=3
)

print("PCOS model saved successfully.")


# -----------------------------
# Final message
# -----------------------------

print("\n===================================")
print("MODEL TRAINING COMPLETED")
print("===================================")
print("PCOD model  : pcod_model.pkl")
print("PCOS model  : pcos_model.pkl")
print("Encoders    : saved successfully")
print("===================================")