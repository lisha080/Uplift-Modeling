import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("data/processed/synthetic_data.csv")
OUTPUT_PATH = Path("data/processed/cleaned_data.csv")

def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Handle missing or inconsistent values
    before = len(df)
    df = df.drop_duplicates()
    print(f"\nDropped {before - len(df)} duplicate rows")

    # Standardize text 
    for col in ["Geography", "Gender"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Fill missing categoricals 
    for col in ["Geography", "Gender"]:
        if col in df.columns and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    # Clip known columns to valid ranges
    clip_ranges = {
        "Age": (18, 100), "CreditScore": (300, 900), "Tenure": (0, 50),
        "Balance": (0, None), "NumOfProducts": (1, 10),
        "EstimatedSalary": (0, None), "engagement_score": (0, 100),
        "login_frequency": (0, 50), "purchase_count": (0, None),
        "support_tickets": (0, 20), "treatment": (0, 1), "outcome": (0, 1),
        "previous_campaign_response": (0, 1), "HasCrCard": (0, 1),
        "IsActiveMember": (0, 1),
    }
    for col, (lo, hi) in clip_ranges.items():
        if col in df.columns:
            if lo is not None:
                df[col] = df[col].clip(lower=lo)
            if hi is not None:
                df[col] = df[col].clip(upper=hi)

    # Remove non-modeling columns
    leakage_cols = [
        "RowNumber", "Surname", "Exited", "CustomerId",
        "p_churn_base", "ite", "segment",
        "retention_prob", "risk_score",
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    print(f"Dropped leakage/ID columns")

    # Encode categorical variables 
    if "Geography" in df.columns:
        df = pd.get_dummies(df, columns=["Geography"], drop_first=True, dtype=int)
        print("Encoded Geography → one-hot (drop_first=True)")

    if "Gender" in df.columns:
        df["Gender"] = (df["Gender"] == "Male").astype(int)
        print("Encoded Gender → binary (Female=0, Male=1)")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Shape: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
