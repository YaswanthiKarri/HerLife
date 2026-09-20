import os
import pandas as pd
import numpy as np

# Number of records
np.random.seed(42)
n = 12000

# Create datasets folder automatically
os.makedirs("datasets", exist_ok=True)

# Generate data
data = {
    "Age": np.random.randint(18, 41, n),
    "BMI": np.round(np.random.uniform(18, 35, n), 1),
    "Cycle length": np.random.randint(21, 46, n),
    "Period duration": np.random.randint(2, 9, n),
    "Irregular periods": np.random.randint(0, 2, n),
    "Acne": np.random.randint(0, 2, n),
    "Excess hair growth": np.random.randint(0, 2, n),
    "Hair thinning": np.random.randint(0, 2, n),
    "Weight gain": np.random.randint(0, 2, n),
    "Dark skin patches": np.random.randint(0, 2, n),
    "Pelvic pain": np.random.randint(0, 2, n),
    "Fatigue": np.random.randint(0, 2, n),
    "Family history": np.random.randint(0, 2, n),
    "Physical activity": np.random.randint(1, 6, n),
    "Sleep": np.random.randint(4, 10, n),
    "Stress": np.random.randint(1, 6, n),
    "Water intake": np.round(np.random.uniform(1, 4, n), 1)
}

df = pd.DataFrame(data)

# -----------------------------
# PCOS RISK SCORE
# -----------------------------

pcos_score = (
    (df["BMI"] >= 25).astype(int)
    + (df["Cycle length"] >= 35).astype(int)
    + df["Irregular periods"]
    + df["Acne"]
    + df["Excess hair growth"]
    + df["Hair thinning"]
    + df["Weight gain"]
    + df["Dark skin patches"]
    + df["Family history"]
    + (df["Stress"] >= 4).astype(int)
)

df["PCOS risk label"] = np.where(
    pcos_score <= 2,
    "Low",
    np.where(pcos_score <= 5, "Moderate", "Higher")
)

# -----------------------------
# PCOD RISK SCORE
# -----------------------------

pcod_score = (
    (df["BMI"] >= 25).astype(int)
    + (df["Cycle length"] >= 32).astype(int)
    + df["Irregular periods"]
    + df["Acne"]
    + df["Weight gain"]
    + df["Fatigue"]
    + df["Pelvic pain"]
    + df["Family history"]
    + (df["Sleep"] <= 5).astype(int)
    + (df["Stress"] >= 4).astype(int)
)

df["PCOD risk label"] = np.where(
    pcod_score <= 2,
    "Low",
    np.where(pcod_score <= 5, "Moderate", "Higher")
)

# -----------------------------
# SAVE CSV
# -----------------------------

file_path = "datasets/pcod_pcos_dataset.csv"

df.to_csv(
    file_path,
    index=False,
    encoding="utf-8"
)

print()
print("======================================")
print(" HERLIFE DATASET CREATED SUCCESSFULLY")
print("======================================")
print()
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()
print("File created:")
print(file_path)
print()
print("Dataset is ready for ML training.")