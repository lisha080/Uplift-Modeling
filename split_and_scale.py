import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

INPUT_PATH = Path("data/processed/cleaned_data.csv")
OUTPUT_DIR = Path("data/processed")

RANDOM_SEED = 42
TEST_SIZE = 0.2

def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Split dataset preserving treatment/control balance 
    target_col = "outcome"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=X["treatment"]
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    print(f"\nTrain: {len(X_train)} rows, Test: {len(X_test)} rows")

    # Verify distributions across splits
    print("\n--- Distribution Check ---")
    print(f"  Treatment rate — Train: {X_train['treatment'].mean():.3f}, Test: {X_test['treatment'].mean():.3f}")
    print(f"  Retention rate — Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

    key_cols = [col for col in [
        "Age", "CreditScore", "Balance", "Tenure",
        "engagement_score", "login_frequency",
        "purchase_count", "support_tickets"
    ] if col in X_train.columns]
    for col in key_cols:
        print(f"  {col} mean — Train: {X_train[col].mean():.3f}, Test: {X_test[col].mean():.3f}")

    # Identify binary columns to skip scaling
    binary_cols = [
        col for col in X_train.columns
        if set(X_train[col].dropna().unique()).issubset({0, 1})
    ]
    scale_cols = [
        col for col in X_train.select_dtypes(include=[np.number]).columns
        if col not in binary_cols
    ]

    # Fit on train only, apply to both
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    print(f"\nScaled {len(scale_cols)} columns: {scale_cols}")
    print(f"Skipped {len(binary_cols)} binary columns")

    # Save 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    pd.DataFrame({target_col: y_train.values}).to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    pd.DataFrame({target_col: y_test.values}).to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    # Save
    summary = {
        "train": {"rows": len(X_train), "treatment_rate": round(float(X_train["treatment"].mean()), 4), "retention_rate": round(float(y_train.mean()), 4)},
        "test":  {"rows": len(X_test),  "treatment_rate": round(float(X_test["treatment"].mean()), 4),  "retention_rate": round(float(y_test.mean()), 4)},
        "scaled_columns": scale_cols,
        "binary_columns_skipped": binary_cols,
        "key_feature_means": {
            col: {
                "train_mean": round(float(X_train[col].mean()), 4),
                "test_mean": round(float(X_test[col].mean()), 4)
            }
            for col in key_cols
        },
    }
    with open(OUTPUT_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: X_train.csv, X_test.csv, y_train.csv, y_test.csv, split_summary.json")


if __name__ == "__main__":
    main()
