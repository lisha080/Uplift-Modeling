import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs")
MODEL_NAMES = ["logistic_t_learner", "random_forest_t_learner"]

# Best model = highest quintile spread, top quintile mean as tiebreaker
SELECTION_RULE = "Highest quintile_spread, with top_quintile_mean_uplift as tiebreaker"

# Model comparison
def load_predictions(model_name):
    path = OUTPUT_DIR / model_name / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run train_models.py first.")
    return pd.read_csv(path)

def compute_metrics(predictions, model_name):
    scores = predictions["uplift_score"]
    predictions = predictions.copy()
    predictions["quintile"] = pd.qcut(scores, 5, labels=False, duplicates="drop")
    quintile_summary = predictions.groupby("quintile")["uplift_score"].mean()

    top_q = float(quintile_summary.max())
    bottom_q = float(quintile_summary.min())

    return {
        "model_name": model_name,
        "mean_uplift": round(float(scores.mean()), 4),
        "std_uplift": round(float(scores.std()), 4),
        "min_uplift": round(float(scores.min()), 4),
        "max_uplift": round(float(scores.max()), 4),
        "n_unique_scores": int(len(np.unique(np.round(scores, 4)))),
        "top_quintile_mean_uplift": round(top_q, 4),
        "bottom_quintile_mean_uplift": round(bottom_q, 4),
        "quintile_spread": round(top_q - bottom_q, 4),
    }

def choose_best_model(metrics_df):
    ranked = metrics_df.sort_values(
        by=["quintile_spread", "top_quintile_mean_uplift"], ascending=False
    ).reset_index(drop=True)
    return ranked.iloc[0]["model_name"], ranked

def print_comparison(ranked):
    names = list(ranked["model_name"])
    compare_keys = [
        "mean_uplift", "std_uplift", "min_uplift", "max_uplift",
        "n_unique_scores", "top_quintile_mean_uplift",
        "bottom_quintile_mean_uplift", "quintile_spread",
    ]

    print(f"\n{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}")
    header = f"  {'Metric':<28}" + "".join(f"{n:>22}" for n in names)
    print(header)
    print(f"  {'-'*28}" + "-" * 22 * len(names))
    for key in compare_keys:
        vals = [ranked.loc[ranked["model_name"] == n, key].values[0] for n in names]
        formatted = "".join(
            f"{float(v):>22.4f}" if isinstance(v, (int, float)) else f"{str(v):>22}"
            for v in vals
        )
        print(f"  {key:<28}{formatted}")

# Segment analysis
def safe_qcut(series, n, labels):
    """qcut with fallback to median split if bins collapse."""
    try:
        return pd.qcut(series, n, labels=labels, duplicates="drop")
    except ValueError:
        median = series.median()
        return np.where(series <= median, labels[0], labels[-1])

def add_segments(df):
    """Create business-relevant customer segments from available columns."""
    segmented = df.copy()

    if "Age" in segmented.columns:
        segmented["age_group"] = safe_qcut(segmented["Age"], 3, ["Young", "Mid", "Older"])

    if "Tenure" in segmented.columns:
        segmented["tenure_group"] = safe_qcut(segmented["Tenure"], 3, ["Short", "Medium", "Long"])

    if "NumOfProducts" in segmented.columns:
        median = segmented["NumOfProducts"].median()
        segmented["product_group"] = pd.cut(
            segmented["NumOfProducts"],
            bins=[-np.inf, median, np.inf],
            labels=["1-2 Products", "3+ Products"],
        )

    if "IsActiveMember" in segmented.columns:
        segmented["activity_segment"] = segmented["IsActiveMember"].map(
            {0: "Inactive", 1: "Active"}
        )

    if "Geography_Germany" in segmented.columns and "Geography_Spain" in segmented.columns:
        segmented["geography"] = np.select(
            [segmented["Geography_Germany"] == 1, segmented["Geography_Spain"] == 1],
            ["Germany", "Spain"],
            default="France",
        )
    elif "Geography_Germany" in segmented.columns:
        segmented["geography"] = np.where(
            segmented["Geography_Germany"] == 1, "Germany", "Other"
        )

    if "engagement_score" in segmented.columns:
        segmented["engagement_level"] = safe_qcut(
            segmented["engagement_score"], 3, ["Low", "Medium", "High"]
        )

    return segmented

def summarize_segment(df, segment_col):
    """Calculate predicted and observed uplift for each group in a segment."""
    rows = []
    for group, subset in df.dropna(subset=[segment_col]).groupby(segment_col, observed=False):
        treated = subset[subset["treatment"] == 1]
        control = subset[subset["treatment"] == 0]

        treated_ret = treated["outcome"].mean() if len(treated) > 0 else np.nan
        control_ret = control["outcome"].mean() if len(control) > 0 else np.nan
        observed = (treated_ret - control_ret) if pd.notna(treated_ret) and pd.notna(control_ret) else np.nan

        rows.append({
            "segment_type": segment_col,
            "segment_value": str(group),
            "n_customers": len(subset),
            "mean_predicted_uplift": round(float(subset["uplift_score"].mean()), 4),
            "treated_retention": round(float(treated_ret), 4) if pd.notna(treated_ret) else None,
            "control_retention": round(float(control_ret), 4) if pd.notna(control_ret) else None,
            "observed_uplift": round(float(observed), 4) if pd.notna(observed) else None,
        })
    return rows

def build_segment_analysis(df):
    segment_cols = [
        "activity_segment", "geography", "age_group",
        "tenure_group", "product_group", "engagement_level",
    ]
    all_rows = []
    for col in segment_cols:
        if col in df.columns:
            all_rows.extend(summarize_segment(df, col))

    if not all_rows:
        raise ValueError("No segment columns available for analysis.")

    segment_df = pd.DataFrame(all_rows)
    return segment_df.sort_values("mean_predicted_uplift", ascending=False).reset_index(drop=True)

def build_recommendations(segment_df, best_model):
    """Identify top targeting opportunities and segments to avoid."""
    positive = segment_df[segment_df["mean_predicted_uplift"] > 0].head(5)
    negative = segment_df[segment_df["mean_predicted_uplift"] < 0].sort_values("mean_predicted_uplift").head(5)

    def to_records(df):
        return df[["segment_type", "segment_value", "mean_predicted_uplift",
                    "observed_uplift", "n_customers"]].to_dict(orient="records")

    return {
        "best_model_used": best_model,
        "recommendation_rule": "Prioritize segments with highest positive predicted uplift. "
                               "Avoid segments with negative predicted uplift.",
        "top_target_segments": to_records(positive),
        "avoid_segments": to_records(negative),
    }

def main():
    # Load predictions and compare models
    all_predictions = {}
    metrics_rows = []
    for name in MODEL_NAMES:
        preds = load_predictions(name)
        all_predictions[name] = preds
        metrics_rows.append(compute_metrics(preds, name))
        print(f"Loaded: {name} ({len(preds)} customers)")

    metrics_df = pd.DataFrame(metrics_rows)
    best_model, ranked = choose_best_model(metrics_df)

    print_comparison(ranked)

    print(f"\n{'='*60}")
    print("Best Model Selection")
    print(f"{'='*60}")
    print(f"  Rule: {SELECTION_RULE}")
    print(f"  Best model: {best_model}")
    print(f"  Quintile spread: {ranked.iloc[0]['quintile_spread']}")

    ranked.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Segment analysis on best model
    predictions = all_predictions[best_model]
    segmented = add_segments(predictions)
    segment_df = build_segment_analysis(segmented)
    recommendations = build_recommendations(segment_df, best_model)

    print(f"\n{'='*60}")
    print(f"Segment Analysis (using {best_model})")
    print(f"{'='*60}")
    print(segment_df.to_string(index=False))

    print(f"\n{'='*60}")
    print("Targeting Recommendations")
    print(f"{'='*60}")
    print("\n  Top targets:")
    for t in recommendations["top_target_segments"]:
        print(f"    {t['segment_type']} = {t['segment_value']}: "
              f"uplift = {t['mean_predicted_uplift']}, n = {t['n_customers']}")

    print("\n  Avoid targeting:")
    if recommendations["avoid_segments"]:
        for t in recommendations["avoid_segments"]:
            print(f"    {t['segment_type']} = {t['segment_value']}: "
                  f"uplift = {t['mean_predicted_uplift']}, n = {t['n_customers']}")
    else:
        print("    None — all segments show positive uplift")

    segment_df.to_csv(OUTPUT_DIR / "segment_analysis.csv", index=False)
    with open(OUTPUT_DIR / "targeting_recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)

    print(f"\nSaved: model_comparison.csv, segment_analysis.csv, targeting_recommendations.json")

if __name__ == "__main__":
    main()
