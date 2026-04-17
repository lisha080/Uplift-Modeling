import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs")

# Columns preserved in predictions for segment analysis
SEGMENT_READY_COLS = [
    "Age", "Tenure", "NumOfProducts", "Balance", "EstimatedSalary",
    "CreditScore", "IsActiveMember", "HasCrCard", "Gender",
    "Geography_Germany", "Geography_Spain",
    "engagement_score", "login_frequency",
    "support_tickets", "previous_campaign_response",
]

def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["outcome"]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["outcome"]
    return X_train, X_test, y_train, y_test

def get_feature_columns(X_train):
    return [col for col in X_train.columns if col != "treatment"]

def train_t_learner(X_train, y_train, feature_cols, model_type="logistic"):
    """Train two models: one on treated customers, one on control."""
    treated_mask = X_train["treatment"] == 1
    control_mask = X_train["treatment"] == 0

    X_treated = X_train.loc[treated_mask, feature_cols]
    y_treated = y_train.loc[treated_mask]
    X_control = X_train.loc[control_mask, feature_cols]
    y_control = y_train.loc[control_mask]

    print(f"  Treated training samples: {len(X_treated)}")
    print(f"  Control training samples: {len(X_control)}")

    if model_type == "logistic":
        model_treated = LogisticRegression(max_iter=1000, random_state=42)
        model_control = LogisticRegression(max_iter=1000, random_state=42)
        model_name = "logistic_t_learner"
    elif model_type == "random_forest":
        model_treated = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        )
        model_control = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        )
        model_name = "random_forest_t_learner"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model_treated.fit(X_treated, y_treated)
    model_control.fit(X_control, y_control)
    print(f"  Trained both models")

    return model_treated, model_control, model_name

def build_predictions(X_test, y_test, model_treated, model_control, feature_cols):
    """Generate uplift scores: P(retain|treated) - P(retain|control)."""
    X_features = X_test[feature_cols]

    p_treated = model_treated.predict_proba(X_features)[:, 1]
    p_control = model_control.predict_proba(X_features)[:, 1]

    predictions = pd.DataFrame({
        "customer_index": X_test.index,
        "treatment": X_test["treatment"].values,
        "outcome": y_test.values,
        "p_treated": p_treated,
        "p_control": p_control,
        "uplift_score": p_treated - p_control,
    })

    for col in SEGMENT_READY_COLS:
        if col in X_test.columns:
            predictions[col] = X_test[col].values

    return predictions

def print_checks(predictions, model_name):
    """Print sanity checks and return metrics dict."""
    scores = predictions["uplift_score"]
    n_unique = len(np.unique(np.round(scores, 4)))

    predictions = predictions.copy()
    predictions["quintile"] = pd.qcut(scores, 5, labels=False, duplicates="drop")
    quintile_summary = (
        predictions.groupby("quintile")
        .agg(mean_uplift=("uplift_score", "mean"), n_customers=("outcome", "count"))
        .reset_index()
    )

    top_q = float(quintile_summary["mean_uplift"].max())
    bottom_q = float(quintile_summary["mean_uplift"].min())

    print(f"\n  Uplift score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Uplift score mean:  {scores.mean():.4f}")
    print(f"  Uplift score std:   {scores.std():.4f}")
    print(f"  Unique score values: {n_unique}")
    print(f"  Top quintile mean:    {top_q:.4f}")
    print(f"  Bottom quintile mean: {bottom_q:.4f}")
    print(f"  Quintile spread:      {top_q - bottom_q:.4f}")

    print("\n  --- Subgroup Check ---")
    for col, labels in [
        ("IsActiveMember", {1: "Active", 0: "Inactive"}),
        ("Geography_Germany", {1: "Germany", 0: "Non-Germany"}),
        ("HasCrCard", {1: "Has card", 0: "No card"}),
    ]:
        if col in predictions.columns:
            means = predictions.groupby(col)["uplift_score"].mean()
            for val, label in labels.items():
                print(f"  {label:15s} {means.get(val, float('nan')):.4f}")

    print(f"\n  --- Uplift by Quintile ---")
    print(quintile_summary.to_string(index=False))

    return {
        "model_name": model_name,
        "mean_uplift": round(float(scores.mean()), 4),
        "std_uplift": round(float(scores.std()), 4),
        "min_uplift": round(float(scores.min()), 4),
        "max_uplift": round(float(scores.max()), 4),
        "n_unique_scores": n_unique,
        "top_quintile_mean_uplift": round(top_q, 4),
        "bottom_quintile_mean_uplift": round(bottom_q, 4),
        "quintile_spread": round(top_q - bottom_q, 4),
    }

def run_model(X_train, X_test, y_train, y_test, model_type):
    """Full pipeline for one model: train, predict, check, save."""
    print(f"\n{'='*60}")
    print(f"Training: {model_type}")
    print(f"{'='*60}")

    feature_cols = get_feature_columns(X_train)
    model_treated, model_control, model_name = train_t_learner(
        X_train, y_train, feature_cols, model_type=model_type
    )
    predictions = build_predictions(
        X_test, y_test, model_treated, model_control, feature_cols
    )
    metrics = print_checks(predictions, model_name)

    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(model_dir / "predictions.csv", index=False)
    print(f"\n  Saved to {model_dir}/predictions.csv")

    return metrics

def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    lr_metrics = run_model(X_train, X_test, y_train, y_test, "logistic")
    rf_metrics = run_model(X_train, X_test, y_train, y_test, "random_forest")

    print(f"\n{'='*60}")
    print("Quick Comparison")
    print(f"{'='*60}")
    print(f"  {'Metric':<28} {'Logistic':>10} {'RandomForest':>14}")
    print(f"  {'-'*52}")
    for key in ["mean_uplift", "std_uplift", "quintile_spread",
                 "top_quintile_mean_uplift", "bottom_quintile_mean_uplift"]:
        print(f"  {key:<28} {lr_metrics[key]:>10} {rf_metrics[key]:>14}")

if __name__ == "__main__":
    main()
