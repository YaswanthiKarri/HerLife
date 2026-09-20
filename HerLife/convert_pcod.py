import pandas as pd

# Read the PCOD dataset
df = pd.read_csv(
    "PCOD_dataset_10000_processed.csv",
    encoding="utf-8"
)

# Save as Excel
df.to_excel(
    "PCOD_dataset_clean.xlsx",
    index=False,
    engine="openpyxl"
)

print("PCOD dataset converted successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("File: PCOD_dataset_clean.xlsx")