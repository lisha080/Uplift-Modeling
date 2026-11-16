"""
Loads the saved outputs shown in dashboard.py, plus the filter and explanation logic it uses.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from train_models import OUTPUT_DIR

PATHS = {
    "ranked": OUTPUT_DIR / "ranking" / "ranked_customers.csv",
    "segments": OUTPUT_DIR / "ranking" / "segment_summary.csv",
    "importance": OUTPUT_DIR / "shap" / "feature_importance.csv",
    "shap_values": OUTPUT_DIR / "shap" / "shap_values.csv",
    "strategy_scores": OUTPUT_DIR / "strategy_scores.csv",
    "roi_table": OUTPUT_DIR / "roi" / "roi_by_outreach_size.csv",
    "final_config": OUTPUT_DIR / "final_model" / "final_model_config.json",
    "customers": Path("data/processed/synthetic_data.csv"),
}
SHAP_IMAGES = {
    "summary": OUTPUT_DIR / "shap" / "shap_summary.png",
    "top_factors": OUTPUT_DIR / "shap" / "shap_top_factors.png",
}
MAKES = {
    "ranked": "rank_customers.py", "segments": "rank_customers.py", "importance": "shap_analysis.py",
    "shap_values": "shap_analysis.py", "strategy_scores": "baseline_comparison.py",
    "roi_table": "roi_simulation.py", "final_config": "check_final_model.py", "customers": "generate_dataset.py",
}
PROFILE_COLS = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
                "IsActiveMember", "EstimatedSalary", "engagement_score", "login_frequency", "purchase_count",
                "support_tickets", "previous_campaign_response"]
YES_NO = {"HasCrCard", "IsActiveMember", "previous_campaign_response"}


@dataclass
class DashboardData:
    ranked: pd.DataFrame
    segments: pd.DataFrame
    importance: pd.DataFrame
    shap: pd.DataFrame
    customers: pd.DataFrame
    strategy_scores: pd.DataFrame
    roi_table: pd.DataFrame
    final_config: dict


def missing_files():
    return [(str(p), MAKES[k]) for k, p in PATHS.items() if not Path(p).exists()]


def load_all():
    missing = missing_files()
    if missing:
        lines = "\n".join(f"  {f}  (run {s})" for f, s in missing)
        raise FileNotFoundError(f"Dashboard data missing:\n{lines}")
    raw = pd.read_csv(PATHS["customers"])
    shap = pd.read_csv(PATHS["shap_values"])
    shap.insert(0, "CustomerId", raw.loc[shap["customer_index"], "CustomerId"].values)
    return DashboardData(
        ranked=pd.read_csv(PATHS["ranked"]),
        segments=pd.read_csv(PATHS["segments"]),
        importance=pd.read_csv(PATHS["importance"]),
        shap=shap.set_index("CustomerId"),
        customers=raw.set_index("CustomerId")[PROFILE_COLS],
        strategy_scores=pd.read_csv(PATHS["strategy_scores"]),
        roi_table=pd.read_csv(PATHS["roi_table"]),
        final_config=json.loads(Path(PATHS["final_config"]).read_text()),
    )


def filter_customers(ranked, groups=None, countries=None, activity="All", response="All",
                     products=None, age_range=None):
    d = ranked
    if groups:
        d = d[d["priority_segment"].isin(groups)]
    if countries:
        d = d[d["Geography"].isin(countries)]
    if activity != "All":
        d = d[d["IsActiveMember"] == (1 if activity == "Active" else 0)]
    if response != "All":
        d = d[d["previous_campaign_response"] == (1 if response == "Responded" else 0)]
    if products:
        d = d[d["NumOfProducts"].isin(products)]
    if age_range:
        d = d[d["Age"].between(*age_range)]
    d = d.sort_values("priority_rank").copy()
    d.insert(0, "rank_in_selection", range(1, len(d) + 1))
    return d


def display_value(feature, row):
    if feature.startswith("Geography_"):
        return "Yes" if row["Geography"] == feature.split("_", 1)[1] else "No"
    value = row[feature]
    if feature in YES_NO:
        return "Yes" if value == 1 else "No"
    if feature in ("Balance", "EstimatedSalary"):
        return f"${value:,.0f}"
    return str(value)


def explain_customer(data, customer_id, top_n=8):
    labels = dict(zip(data.importance["feature"], data.importance["label"]))
    row = data.shap.loc[customer_id]
    raw = data.customers.loc[customer_id]
    contrib = row.drop(["customer_index", "uplift_score", "base_value"]).astype(float)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)
    table = pd.DataFrame({
        "feature": top.index,
        "label": [f"{labels.get(f, f)}: {display_value(f, raw)}" for f in top.index],
        "effect_pts": top.values * 100,
    })
    return {"base_pts": float(row["base_value"]) * 100, "uplift_pts": float(row["uplift_score"]) * 100,
            "other_pts": float(contrib.drop(top.index).sum()) * 100, "factors": table}