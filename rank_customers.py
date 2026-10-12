import json
from pathlib import Path

import numpy as np
import pandas as pd

import uplift_metrics as um
from train_models import OUTPUT_DIR

SYNTHETIC_PATH = Path("data/processed/synthetic_data.csv")
RANK_DIR = OUTPUT_DIR / "ranking"
FINAL_PATH = OUTPUT_DIR / "final_model" / "oof_predictions.csv"
REASONS_PATH = OUTPUT_DIR / "shap" / "customer_reasons.csv"

# quintile 4 = highest uplift -> Priority 1
SEGMENTS = {
    4: ("Priority 1", "Contact first"),
    3: ("Priority 2", "Contact"),
    2: ("Priority 3", "Contact if budget allows"),
    1: ("Priority 4", "Low priority"),
    0: ("Priority 5", "Do not prioritize"),
}
PROFILE_COLS = ["Geography", "Age", "Tenure", "Balance", "NumOfProducts", "IsActiveMember",
                "engagement_score", "support_tickets", "previous_campaign_response"]


def load_scores():
    selected = json.loads((OUTPUT_DIR / "selection_summary.json").read_text())["selected_model"]
    if FINAL_PATH.exists():
        pred, selected = pd.read_csv(FINAL_PATH), f"{selected} (tuned, Sprint 8)"
    else:
        pred = pd.read_csv(OUTPUT_DIR / selected / "oof_predictions.csv")

    # customer_index is the row number in synthetic_data.csv
    syn = pd.read_csv(SYNTHETIC_PATH)
    ids = pred["customer_index"].values
    if not (syn.loc[ids, "treatment"].values == pred["treatment"].values).all():
        raise ValueError("Predictions do not line up with synthetic_data.csv. Re-run the pipeline.")
    pred["CustomerId"] = syn.loc[ids, "CustomerId"].values
    pred["Geography"] = syn.loc[ids, "Geography"].values
    return pred, selected


def build_ranked_list(pred):
    df = pred.copy()
    quintile = um.assign_quintiles(df["uplift_score"].values)
    df["priority_segment"] = [SEGMENTS[q][0] for q in quintile]
    df["recommended_action"] = [SEGMENTS[q][1] for q in quintile]
    df["retention_with_outreach"] = df["p_treated"]
    df["retention_without_outreach"] = df["p_control"]
    if REASONS_PATH.exists():
        df = df.merge(pd.read_csv(REASONS_PATH), on="customer_index", how="left")

    df = df.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    df["priority_rank"] = np.arange(1, len(df) + 1)

    cols = ["priority_rank", "CustomerId", "priority_segment", "recommended_action", "uplift_score",
            "retention_with_outreach", "retention_without_outreach"]
    if "main_reasons" in df.columns:
        cols.append("main_reasons")
    cols += PROFILE_COLS + ["treatment", "outcome"]
    return df[cols]


def summarize_segments(ranked):
    rows = []
    for name, action in [SEGMENTS[q] for q in [4, 3, 2, 1, 0]]:
        g = ranked[ranked["priority_segment"] == name]
        rows.append({
            "priority_segment": name,
            "recommended_action": action,
            "n_customers": len(g),
            "min_uplift_score": g["uplift_score"].min(),
            "max_uplift_score": g["uplift_score"].max(),
            "mean_predicted_uplift": g["uplift_score"].mean(),
            "observed_uplift": um.observed_uplift(g["treatment"], g["outcome"]),
            "n_negative_uplift": int((g["uplift_score"] < 0).sum()),
        })
    summary = pd.DataFrame(rows)
    summary["cumulative_customers"] = summary["n_customers"].cumsum()
    return summary


def verify(ranked, summary, n_scored):
    assert summary["n_customers"].sum() == n_scored, "Segment totals do not match the number of scored customers"
    assert ranked["CustomerId"].is_unique, "A customer appears more than once"
    assert (np.diff(ranked["uplift_score"].values) <= 0).all(), "List is not sorted by uplift"


def main():
    pred, selected = load_scores()
    ranked = build_ranked_list(pred)
    summary = summarize_segments(ranked)
    verify(ranked, summary, len(pred))

    RANK_DIR.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(RANK_DIR / "ranked_customers.csv", index=False)
    summary.to_csv(RANK_DIR / "segment_summary.csv", index=False)

    print(f"Ranked {len(ranked):,} customers using {selected}\n")
    show = summary.copy()
    for c in ["min_uplift_score", "max_uplift_score", "mean_predicted_uplift", "observed_uplift"]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print(f"\nCheck passed: segment totals add up to {summary['n_customers'].sum():,} customers")
    print(f"Saved {RANK_DIR}/ranked_customers.csv and segment_summary.csv")


if __name__ == "__main__":
    main()