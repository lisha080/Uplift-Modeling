import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path("data/processed")

def main():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["outcome"]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["outcome"]

    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # T-Learner: train one model on treated, one on control
    # Uplift = P(retain | treated) - P(retain | control)
    # Using logistic regression as a simple baseline
    print("\nApproach: T-Learner with Logistic Regression")

    # Split train data by treatment group
    treated_mask = X_train["treatment"] == 1
    control_mask = X_train["treatment"] == 0

    feature_cols = [col for col in X_train.columns if col != "treatment"]

    X_train_treated = X_train.loc[treated_mask, feature_cols]
    y_train_treated = y_train.loc[treated_mask]
    X_train_control = X_train.loc[control_mask, feature_cols]
    y_train_control = y_train.loc[control_mask]

    print(f"  Treated training samples: {len(X_train_treated)}")
    print(f"  Control training samples: {len(X_train_control)}")

    model_treated = LogisticRegression(max_iter=1000, random_state=42)
    model_treated.fit(X_train_treated, y_train_treated)
    print("\nTrained treatment model completed")

    model_control = LogisticRegression(max_iter=1000, random_state=42)
    model_control.fit(X_train_control, y_train_control)
    print("Trained control model completed")

    X_test_features = X_test[feature_cols]

    # Predict P(retain) under each model
    p_treated = model_treated.predict_proba(X_test_features)[:, 1]
    p_control = model_control.predict_proba(X_test_features)[:, 1]
    uplift_scores = p_treated - p_control

    print("\n--- Basic Output Checks ---")

    # Check 1: Are predictions varying across customers?
    print(f"\n  Uplift score range: [{uplift_scores.min():.4f}, {uplift_scores.max():.4f}]")
    print(f"  Uplift score mean:  {uplift_scores.mean():.4f}")
    print(f"  Uplift score std:   {uplift_scores.std():.4f}")

    # Check 2: Are they all the same value? (red flag)
    n_unique = len(np.unique(np.round(uplift_scores, 4)))
    print(f"  Unique score values: {n_unique} (should be >> 1)")

    # Check 3: Do scores vary by key features?
    print("\n--- Uplift by Subgroup (sanity check) ---")

    results = pd.DataFrame({
        "uplift_score": uplift_scores,
        "treatment": X_test["treatment"].values,
        "outcome": y_test.values,
    })

    # Attach a few unscaled-friendly checks using binary columns
    for col in ["Gender", "IsActiveMember", "HasCrCard", "Geography_Germany"]:
        if col in X_test.columns:
            results[col] = X_test[col].values

    if "IsActiveMember" in results.columns:
        active = results.groupby("IsActiveMember")["uplift_score"].mean()
        print(f"  Active members:   {active.get(1, 'N/A'):.4f}")
        print(f"  Inactive members: {active.get(0, 'N/A'):.4f}")

    if "Geography_Germany" in results.columns:
        geo = results.groupby("Geography_Germany")["uplift_score"].mean()
        print(f"  Germany:     {geo.get(1, 'N/A'):.4f}")
        print(f"  Non-Germany: {geo.get(0, 'N/A'):.4f}")

    # Check 4: Uplift by decile
    results["decile"] = pd.qcut(results["uplift_score"], 5, labels=False, duplicates="drop")
    decile_summary = results.groupby("decile").agg(
        mean_uplift=("uplift_score", "mean"),
        n_customers=("outcome", "count"),
    ).reset_index()
    print(f"\n--- Uplift by Quintile ---")
    print(decile_summary.to_string(index=False))

    print("\n--- Status ---")
    print("Pipeline runs end-to-end. Outputs vary across customers.")
    print("Not optimized yet — this is a baseline for validation only.")


if __name__ == "__main__":
    main()
