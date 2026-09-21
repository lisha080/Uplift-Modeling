"""
Model comparison + segment analysis.

Usage:
    python evaluate_and_segment.py                 # uses cross-validated (OOF) predictions if present, else holdout
    python evaluate_and_segment.py --source oof    # 10,000 customers, each scored by a model that never saw them
    python evaluate_and_segment.py --source test   # 2,000-customer holdout from train_models.py

All quality metrics are computed on OBSERVED outcomes (see uplift_metrics.py).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import uplift_metrics as um

OUTPUT_DIR = Path("outputs")
MODEL_NAMES = ["logistic_t_learner", "random_forest_t_learner"]
SYNTHETIC_PATH = Path("data/processed/synthetic_data.csv")
N_BOOT = 1000

# Team-proposed rule, now applied to OBSERVED (not predicted) uplift.
SELECTION_RULE = (
    "Highest observed quintile spread (observed uplift of top predicted-uplift quintile "
    "minus observed uplift of bottom quintile), with top-quintile observed uplift as "
    "tiebreaker. A paired bootstrap reports whether the winning margin is distinguishable from zero."
)

# Tie rule: what to do when the models cannot be told apart statistically.
TIE_DEFAULT_MODEL = "logistic_t_learner"
TIE_RULE = (
    "If the top model's margin over another model is not statistically distinguishable from zero "
    "(the 95% paired-bootstrap interval includes 0), the models are treated as TIED and the simpler, "
    "more interpretable model (Logistic Regression) is selected. A model that is decisively worse than "
    "the best is never part of the tie."
)

PREDICTION_FILES = {"test": "predictions.csv", "oof": "oof_predictions.csv"}


# ---------------------------------------------------------------- loading
def resolve_source(requested):
    if requested != "auto":
        return requested
    if all((OUTPUT_DIR / m / PREDICTION_FILES["oof"]).exists() for m in MODEL_NAMES):
        return "oof"
    print("WARNING: no cross-validated predictions found (run cross_validate.py). "
          "Falling back to the 2,000-row holdout, which is too small to separate similar models.")
    return "test"


def load_predictions(model_name, source):
    path = OUTPUT_DIR / model_name / PREDICTION_FILES[source]
    if not path.exists():
        hint = "cross_validate.py" if source == "oof" else "train_models.py"
        raise FileNotFoundError(f"Missing: {path}. Run {hint} first.")
    return pd.read_csv(path)


def check_aligned(all_predictions):
    """Paired comparison requires identical customers in identical order."""
    names = list(all_predictions)
    base = all_predictions[names[0]]
    for other in names[1:]:
        p = all_predictions[other]
        same = (
            len(p) == len(base)
            and (p["customer_index"].values == base["customer_index"].values).all()
            and (p["treatment"].values == base["treatment"].values).all()
            and (p["outcome"].values == base["outcome"].values).all()
        )
        if not same:
            raise ValueError(f"{other} was not scored on the same customers as {names[0]}.")


# ---------------------------------------------------------------- model comparison
def synthetic_truth_correlation(pred):
    """DIAGNOSTIC ONLY (synthetic data): how well do scores track the true effect?

    Never used for model selection: real data has no ground-truth effect.
    """
    if not SYNTHETIC_PATH.exists():
        return None
    syn = pd.read_csv(SYNTHETIC_PATH)
    ids = pred["customer_index"].values
    if ids.max() >= len(syn) or not (syn.loc[ids, "treatment"].values == pred["treatment"].values).all():
        return None  # ids do not map back to the synthetic file
    return float(np.corrcoef(pred["uplift_score"].values, syn.loc[ids, "ite"].values)[0, 1])


def compute_metrics(pred, model_name):
    s, t, o = pred["uplift_score"].values, pred["treatment"].values, pred["outcome"].values
    row = {
        "model_name": model_name,
        "n_customers": len(pred),
        "mean_predicted_uplift": float(s.mean()),
        "std_predicted_uplift": float(s.std()),
        "predicted_quintile_spread": um.predicted_quintile_spread(s),
    }
    row.update(um.core_metrics(s, t, o))
    row["corr_with_true_ite_DIAGNOSTIC"] = synthetic_truth_correlation(pred)
    return row


def choose_best_model(metrics_df):
    ranked = metrics_df.sort_values(
        by=["observed_quintile_spread", "observed_top_quintile_uplift"], ascending=False
    ).reset_index(drop=True)
    return ranked.iloc[0]["model_name"], ranked


def add_bootstrap(ranked, all_predictions):
    """Attach 95% CIs and test whether the best model's margin is distinguishable from zero."""
    first = all_predictions[MODEL_NAMES[0]]
    t, o = first["treatment"].values, first["outcome"].values
    scores = {m: all_predictions[m]["uplift_score"].values for m in ranked["model_name"]}
    draws = um.paired_bootstrap(scores, t, o, n_boot=N_BOOT)

    for metric in ["observed_quintile_spread", "observed_top_quintile_uplift", "auuc_per_1000"]:
        lo, hi = zip(*[um.ci95(draws[m][metric]) for m in ranked["model_name"]])
        ranked[f"{metric}_ci_low"], ranked[f"{metric}_ci_high"] = lo, hi

    best = ranked.iloc[0]["model_name"]
    margins = []
    for other in ranked["model_name"].iloc[1:]:
        for metric, is_rule in [("observed_quintile_spread", True), ("auuc_per_1000", False)]:
            diff = draws[best][metric] - draws[other][metric]
            lo, hi = um.ci95(diff)
            margins.append({"vs": other, "metric": metric, "used_by_rule": is_rule,
                            "diff_mean": float(np.nanmean(diff)), "ci_low": lo, "ci_high": hi,
                            "decisive": bool(lo > 0)})
    return ranked, margins


def apply_tie_rule(best_model, margins):
    """Return (selected_model, reason).

    tied set = the best model plus every model it does NOT decisively beat on the
    primary metric. If the tied set has more than one model, prefer TIE_DEFAULT_MODEL
    (when it is in the set); otherwise the best model by the selection rule wins.
    """
    rule_margins = [m for m in margins if m["used_by_rule"]]
    tied = [best_model] + [m["vs"] for m in rule_margins if not m["decisive"]]
    if len(tied) == 1:
        return best_model, f"{best_model} is decisively better than every other model on the primary metric."
    if TIE_DEFAULT_MODEL in tied:
        return TIE_DEFAULT_MODEL, (f"Statistical tie among {sorted(tied)}; tie rule selects the simpler, more "
                                   f"interpretable model ({TIE_DEFAULT_MODEL}).")
    return best_model, (f"Statistical tie among {sorted(tied)}, but {TIE_DEFAULT_MODEL} is not in the tie; "
                        f"{best_model} wins by the selection rule.")


def print_comparison(ranked):
    names = list(ranked["model_name"])
    keys = [
        ("n_customers", "n_customers"),
        ("predicted_quintile_spread", "predicted_quintile_spread (old metric)"),
        ("observed_top_quintile_uplift", "observed_top_quintile_uplift"),
        ("observed_bottom_quintile_uplift", "observed_bottom_quintile_uplift"),
        ("observed_quintile_spread", "observed_quintile_spread  <-- rule"),
        ("auuc_per_1000", "auuc_per_1000"),
        ("incremental_per_1000_at_20pct", "incremental_per_1000_at_20pct"),
        ("incremental_per_1000_at_30pct", "incremental_per_1000_at_30pct"),
        ("corr_with_true_ite_DIAGNOSTIC", "corr_with_true_ite (synthetic)"),
    ]
    print(f"\n{'=' * 78}\nModel Comparison\n{'=' * 78}")
    print(f"  {'Metric':<42}" + "".join(f"{n:>26}" for n in names))
    print(f"  {'-' * 42}" + "-" * 26 * len(names))
    for col, label in keys:
        vals = ranked.set_index("model_name").loc[names, col]
        cells = "".join("%26s" % ("n/a" if pd.isna(v) else (f"{int(v)}" if col == "n_customers" else f"{v:.4f}"))
                        for v in vals)
        print(f"  {label:<42}{cells}")
    for col, label in [("observed_quintile_spread", "observed_quintile_spread 95% CI"),
                       ("auuc_per_1000", "auuc_per_1000 95% CI")]:
        cells = "".join("%26s" % f"[{r[col + '_ci_low']:.3f}, {r[col + '_ci_high']:.3f}]"
                        for _, r in ranked.iterrows())
        print(f"  {label:<42}{cells}")


# ---------------------------------------------------------------- segment analysis
def assert_unscaled(df):
    """Segments need real feature values. Fail loudly if standardized ones sneak in."""
    if "NumOfProducts" in df.columns and df["NumOfProducts"].min() < 1:
        raise ValueError("NumOfProducts looks standardized. Re-run split_and_scale.py, train_models.py "
                         "and cross_validate.py so predictions carry unscaled segment columns.")
    if "Age" in df.columns and df["Age"].min() < 18:
        raise ValueError("Age looks standardized. Re-run split_and_scale.py, train_models.py and cross_validate.py.")


def safe_qcut(series, n, labels):
    """qcut with fallback to median split if bins collapse."""
    try:
        return pd.qcut(series, n, labels=labels, duplicates="drop")
    except ValueError:
        median = series.median()
        return np.where(series <= median, labels[0], labels[-1])


def add_segments(df):
    """Create business-relevant customer segments from (unscaled) columns."""
    assert_unscaled(df)
    segmented = df.copy()

    if "Age" in segmented.columns:
        segmented["age_group"] = safe_qcut(segmented["Age"], 3, ["Young", "Mid", "Older"])

    if "Tenure" in segmented.columns:
        segmented["tenure_group"] = safe_qcut(segmented["Tenure"], 3, ["Short", "Medium", "Long"])

    if "NumOfProducts" in segmented.columns:
        # Fixed, explicit bins. (The old median split labelled 2+ products as "3+".)
        segmented["product_group"] = pd.cut(
            segmented["NumOfProducts"],
            bins=[0, 1, 2, np.inf],
            labels=["1 Product", "2 Products", "3+ Products"],
        )

    if "IsActiveMember" in segmented.columns:
        segmented["activity_segment"] = segmented["IsActiveMember"].map({0: "Inactive", 1: "Active"})

    if "Geography_Germany" in segmented.columns and "Geography_Spain" in segmented.columns:
        segmented["geography"] = np.select(
            [segmented["Geography_Germany"] == 1, segmented["Geography_Spain"] == 1],
            ["Germany", "Spain"],
            default="France",
        )
    elif "Geography_Germany" in segmented.columns:
        segmented["geography"] = np.where(segmented["Geography_Germany"] == 1, "Germany", "Other")

    if "engagement_score" in segmented.columns:
        segmented["engagement_level"] = safe_qcut(segmented["engagement_score"], 3, ["Low", "Medium", "High"])

    return segmented


def summarize_segment(df, segment_col):
    """Predicted and observed uplift for each group in a segment."""
    rows = []
    for group, subset in df.dropna(subset=[segment_col]).groupby(segment_col, observed=False):
        if len(subset) == 0:
            continue
        treated = subset[subset["treatment"] == 1]
        control = subset[subset["treatment"] == 0]
        p_t = treated["outcome"].mean() if len(treated) else np.nan
        p_c = control["outcome"].mean() if len(control) else np.nan
        observed = p_t - p_c if pd.notna(p_t) and pd.notna(p_c) else np.nan
        se = (np.sqrt(p_t * (1 - p_t) / len(treated) + p_c * (1 - p_c) / len(control))
              if pd.notna(observed) else np.nan)

        rows.append({
            "segment_type": segment_col,
            "segment_value": str(group),
            "n_customers": len(subset),
            "mean_predicted_uplift": round(float(subset["uplift_score"].mean()), 4),
            "treated_retention": round(float(p_t), 4) if pd.notna(p_t) else None,
            "control_retention": round(float(p_c), 4) if pd.notna(p_c) else None,
            "observed_uplift": round(float(observed), 4) if pd.notna(observed) else None,
            "observed_uplift_se": round(float(se), 4) if pd.notna(se) else None,
        })
    return rows


def build_segment_analysis(df):
    segment_cols = ["activity_segment", "geography", "age_group",
                    "tenure_group", "product_group", "engagement_level"]
    all_rows = []
    for col in segment_cols:
        if col in df.columns:
            all_rows.extend(summarize_segment(df, col))
    if not all_rows:
        raise ValueError("No segment columns available for analysis.")
    return pd.DataFrame(all_rows).sort_values("mean_predicted_uplift", ascending=False).reset_index(drop=True)


def build_recommendations(segment_df, best_model):
    """Top targeting opportunities and segments to avoid."""
    positive = segment_df[segment_df["mean_predicted_uplift"] > 0].head(5)
    negative = segment_df[segment_df["mean_predicted_uplift"] < 0].sort_values("mean_predicted_uplift").head(5)

    def to_records(d):
        return d[["segment_type", "segment_value", "mean_predicted_uplift",
                  "observed_uplift", "observed_uplift_se", "n_customers"]].to_dict(orient="records")

    return {
        "best_model_used": best_model,
        "recommendation_rule": "Prioritize segments with highest positive predicted uplift. "
                               "Avoid segments with negative predicted uplift. "
                               "observed_uplift_se shows how noisy each observed estimate is.",
        "top_target_segments": to_records(positive),
        "avoid_segments": to_records(negative),
    }


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["auto", "oof", "test"], default="auto")
    source = resolve_source(parser.parse_args().source)
    label = {"oof": "5-fold cross-validated (all customers)", "test": "holdout test set"}[source]
    print(f"Evaluating on: {label}")

    all_predictions = {}
    for name in MODEL_NAMES:
        all_predictions[name] = load_predictions(name, source)
        print(f"Loaded: {name} ({len(all_predictions[name])} customers)")
    check_aligned(all_predictions)

    metrics_df = pd.DataFrame([compute_metrics(all_predictions[n], n) for n in MODEL_NAMES])
    best_model, ranked = choose_best_model(metrics_df)
    ranked, margins = add_bootstrap(ranked, all_predictions)
    print_comparison(ranked)

    selected_model, selection_reason = apply_tie_rule(best_model, margins)

    print(f"\n{'=' * 78}\nModel Selection\n{'=' * 78}")
    print(f"  Rule:      {SELECTION_RULE}")
    print(f"  Tie rule:  {TIE_RULE}")
    print(f"  Best model by rule (highest observed spread): {best_model}")
    for m in margins:
        verdict = ("DISTINGUISHABLE from zero" if m["decisive"]
                   else "NOT distinguishable from zero -> statistically tied")
        tag = "  <-- rule" if m["used_by_rule"] else "  (informational)"
        print(f"  {best_model} minus {m['vs']} on {m['metric']}: {m['diff_mean']:+.4f}  "
              f"95% CI [{m['ci_low']:+.4f}, {m['ci_high']:+.4f}]  {verdict}{tag}")
    print(f"\n  >>> SELECTED MODEL: {selected_model}")
    print(f"      Why: {selection_reason}")

    # Per-quintile predicted-vs-observed table for every model (feeds the Sprint 7 dashboard)
    quintile_frames = []
    for name in MODEL_NAMES:
        p = all_predictions[name]
        qt = um.quintile_table(p["uplift_score"], p["treatment"], p["outcome"])
        qt.insert(0, "model_name", name)
        quintile_frames.append(qt)
        print(f"\n  --- {name}: predicted vs observed uplift by quintile (4 = highest predicted) ---")
        print(qt.drop(columns="model_name").round(4).to_string(index=False))
    quintile_df = pd.concat(quintile_frames, ignore_index=True)

    ranked.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    quintile_df.to_csv(OUTPUT_DIR / "quintile_table.csv", index=False)
    with open(OUTPUT_DIR / "selection_summary.json", "w") as f:
        json.dump({"evaluated_on": label, "rule": SELECTION_RULE, "tie_rule": TIE_RULE,
                   "best_model_by_rule": best_model, "selected_model": selected_model,
                   "selection_reason": selection_reason, "margins_vs_others": margins}, f, indent=2)

    # Segment analysis on the best model
    segmented = add_segments(all_predictions[selected_model])
    segment_df = build_segment_analysis(segmented)
    recommendations = build_recommendations(segment_df, selected_model)

    print(f"\n{'=' * 78}\nSegment Analysis (using {selected_model})\n{'=' * 78}")
    print(segment_df.to_string(index=False))

    print(f"\n{'=' * 78}\nTargeting Recommendations\n{'=' * 78}")
    print("\n  Top targets:")
    for t in recommendations["top_target_segments"]:
        print(f"    {t['segment_type']} = {t['segment_value']}: predicted uplift = {t['mean_predicted_uplift']}, "
              f"observed = {t['observed_uplift']} (+/-{t['observed_uplift_se']}), n = {t['n_customers']}")
    print("\n  Avoid targeting:")
    if recommendations["avoid_segments"]:
        for t in recommendations["avoid_segments"]:
            print(f"    {t['segment_type']} = {t['segment_value']}: predicted uplift = {t['mean_predicted_uplift']}, "
                  f"observed = {t['observed_uplift']} (+/-{t['observed_uplift_se']}), n = {t['n_customers']}")
    else:
        print("    None: all segments show positive predicted uplift")

    segment_df.to_csv(OUTPUT_DIR / "segment_analysis.csv", index=False)
    with open(OUTPUT_DIR / "targeting_recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)
    print("\nSaved: model_comparison.csv, quintile_table.csv, selection_summary.json, "
          "segment_analysis.csv, targeting_recommendations.json")


if __name__ == "__main__":
    main()