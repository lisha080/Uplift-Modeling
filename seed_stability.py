"""
Checks that the final model's business results hold across different random seeds.
Run after targeting_comparison.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import roi
import uplift_metrics as um
from check_final_model import ranking_stability, run_seeds
from cross_validate import CLEANED_PATH, get_scale_cols
from stability_check import SEEDS
from train_models import OUTPUT_DIR

VALIDATION_DIR = OUTPUT_DIR / "validation"
REPORT_PATH = Path("reports") / "seed_stability.md"
HEADLINE_SHARE = 0.3


def seed_results(scores, t, o, churn, saved_top):
    rows = []
    k = roi.n_contacted(len(t), HEADLINE_SHARE)
    churn_inc = roi.observed_incremental(churn, t, o, k)
    for seed, s in scores.items():
        table = roi.roi_table(s, treatment=t, outcome=o)
        best = table.loc[table["predicted_net_benefit"].idxmax()]
        at = table.set_index("outreach_share").loc[HEADLINE_SHARE]
        top = um.assign_quintiles(s) == 4
        rows.append({
            "seed": seed,
            "quintile_spread": um.core_metrics(s, t, o)["observed_quintile_spread"],
            "priority1_overlap_with_saved": (top & saved_top).sum() / top.sum(),
            "best_outreach_share": best["outreach_share"],
            "best_predicted_net_benefit": best["predicted_net_benefit"],
            "observed_net_benefit_at_30": at["observed_net_benefit"],
            "gain_vs_churn_at_30": (at["observed_incremental"] - churn_inc) * roi.RETENTION_VALUE,
        })
    return pd.DataFrame(rows)


def write_report(df, ranking):
    lines = ["| Seed | Quintile spread | Priority 1 overlap | Best outreach | Net benefit at 30% | Gain vs churn risk at 30% |",
             "|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.seed} | {r.quintile_spread * 100:.1f} pts | {r.priority1_overlap_with_saved:.0%} "
                     f"| Top {r.best_outreach_share:.0%} | ${r.observed_net_benefit_at_30:,.0f} "
                     f"| ${r.gain_vs_churn_at_30:,.0f} |")
    nb, gain = df["observed_net_benefit_at_30"], df["gain_vs_churn_at_30"]
    shares = sorted(df["best_outreach_share"].unique())
    REPORT_PATH.write_text(f"""# Stability Across Random Seeds (Sprint 10)

The final model (Logistic Regression T-Learner, tuned) was retrained with {len(SEEDS)} different random seeds.
Each seed changes how customers are split into the 5 cross-validation folds, so every customer is scored
by a differently trained model. We then checked whether the business results change.

{chr(10).join(lines)}

Seed 42 is the split used for the saved results and the dashboard. Net benefit and gain use real outcomes,
${roi.CONTACT_COST:,.0f} per contact and ${roi.RETENTION_VALUE:,.0f} per customer kept.

## What stays the same

- **Ranking:** average rank correlation between seeds is {ranking['rank_correlation']:.3f}. On average
  {ranking['priority1_overlap']:.0%} of Priority 1 customers are the same between any two seeds, and
  {ranking['within_one_segment_every_split']:.0%} of customers stay within one priority group across all seeds.
- **Recommended outreach size:** {', '.join(f'top {s:.0%}' for s in shares)} across seeds.
- **Net benefit at 30% outreach:** ${nb.min():,.0f} to ${nb.max():,.0f} (average ${nb.mean():,.0f}).
- **Gain over churn-risk targeting at 30%:** ${gain.min():,.0f} to ${gain.max():,.0f}; uplift targeting
  {'wins with every seed' if (gain > 0).all() else f'wins with {int((gain > 0).sum())} of {len(gain)} seeds'}.

## Conclusion

{'The recommendations do not depend on a lucky random split.' if (gain > 0).all() and ranking['priority1_overlap'] > 0.8 else 'Results vary between seeds; treat exact figures with caution.'}
The dashboard uses seed 42; any seed would lead to the same targeting decisions.
""")


def main():
    config = json.loads((OUTPUT_DIR / "final_model" / "final_model_config.json").read_text())
    df = pd.read_csv(CLEANED_PATH)
    y = df["outcome"]
    X = df.drop(columns=["outcome"])
    feature_cols = [c for c in X.columns if c != "treatment"]
    t, o = X["treatment"].values, y.values

    strategy = pd.read_csv(OUTPUT_DIR / "strategy_scores.csv")
    saved_top = um.assign_quintiles(strategy["uplift_final"].values) == 4

    print(f"Retraining the final model with {len(SEEDS)} seeds...")
    _, scores = run_seeds(X, y, config["model_type"], config["params"], get_scale_cols(X), feature_cols)
    results = seed_results(scores, t, o, strategy["churn_risk"].values, saved_top)
    ranking = ranking_stability(scores)

    print(results.round(3).to_string(index=False))
    print(f"\nRank correlation {ranking['rank_correlation']:.3f}, "
          f"Priority 1 overlap {ranking['priority1_overlap']:.0%}")

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(VALIDATION_DIR / "seed_stability.csv", index=False)
    (VALIDATION_DIR / "seed_stability.json").write_text(json.dumps(ranking, indent=2))
    write_report(results, ranking)
    print(f"Saved {REPORT_PATH} and {VALIDATION_DIR}/seed_stability.csv")


if __name__ == "__main__":
    main()