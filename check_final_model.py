import contextlib
import io
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import uplift_metrics as um
from cross_validate import CLEANED_PATH, get_scale_cols, make_folds, run_cv
from stability_check import SEEDS
from train_models import OUTPUT_DIR

FINAL_DIR = OUTPUT_DIR / "final_model"
REPORT_PATH = Path("reports") / "final_model.md"


def run_seeds(X, y, model_type, params, scale_cols, feature_cols):
    t, o = X["treatment"].values, y.values
    metrics, scores = [], {}
    for seed in SEEDS:
        folds = make_folds(X, y, seed=seed)
        with contextlib.redirect_stdout(io.StringIO()):
            oof, _ = run_cv(X, y, folds, scale_cols, feature_cols, model_type, verbose=False, params=params)
        s = oof["uplift_score"].values
        scores[seed] = s
        metrics.append({"seed": seed, **um.core_metrics(s, t, o)})
    return pd.DataFrame(metrics), scores


def ranking_stability(scores):
    # Does a customer land in roughly the same place no matter which split trained the model?
    corrs, overlaps = [], []
    for a, b in itertools.combinations(SEEDS, 2):
        corrs.append(pd.Series(scores[a]).corr(pd.Series(scores[b]), method="spearman"))
        top_a = um.assign_quintiles(scores[a]) == 4
        top_b = um.assign_quintiles(scores[b]) == 4
        overlaps.append((top_a & top_b).sum() / top_a.sum())
    segments = np.vstack([um.assign_quintiles(scores[s]) for s in SEEDS])
    moved = segments.max(axis=0) - segments.min(axis=0)
    return {
        "rank_correlation": float(np.mean(corrs)),
        "priority1_overlap": float(np.mean(overlaps)),
        "same_segment_every_split": float((moved == 0).mean()),
        "within_one_segment_every_split": float((moved <= 1).mean()),
    }


def write_report(config, final_df, default_df, ranking):
    p = config["params"]
    setting = f"C = {p['C']}, {p['penalty'].upper()} penalty" if config["model_type"] == "logistic" else str(p)
    wins = int((final_df["observed_quintile_spread"].values > default_df["observed_quintile_spread"].values).sum())
    s7, fin = config["all_customers"]["sprint7"], config["all_customers"]["final"]
    if wins == len(SEEDS):
        conclusion = "The improvement from tuning is small but consistent."
    elif wins > len(SEEDS) / 2:
        conclusion = "Tuning helped in most splits, but the improvement is small."
    else:
        conclusion = "Tuning did not reliably improve the model across splits."

    text = f"""# Final Model Configuration (Sprint 8)

- **Model:** Logistic Regression T-Learner (one model for contacted customers, one for non-contacted)
- **Setting:** {setting}{' (tuned)' if config['tuned'] else ' (default)'}
- **Features:** all customer columns except the treatment flag, numeric columns standardized
- **Uplift score:** chance of staying if contacted minus chance of staying if not contacted

## Tuning

We tried {len(pd.read_csv(FINAL_DIR / 'tuning_results.csv'))} settings on the 8,000 training customers and checked the best
one on the 2,000 holdout customers. {config['reason']}

| Metric (all 10,000 customers) | Sprint 7 | Final |
|---|---|---|
| Quintile spread | {s7['observed_quintile_spread'] * 100:.1f} pts | {fin['observed_quintile_spread'] * 100:.1f} pts |
| Top quintile observed uplift | {s7['observed_top_quintile_uplift'] * 100:.1f} pts | {fin['observed_top_quintile_uplift'] * 100:.1f} pts |
| Extra retained per 1,000 (top 20% contacted) | {s7['incremental_per_1000_at_20pct']:.1f} | {fin['incremental_per_1000_at_20pct']:.1f} |
| AUUC per 1,000 | {s7['auuc_per_1000']:.1f} | {fin['auuc_per_1000']:.1f} |

## Stability across data splits

We reran the final model and the Sprint 7 model on {len(SEEDS)} different data splits.

- The final model had a higher quintile spread than the Sprint 7 model in {wins} of {len(SEEDS)} splits
  (average {final_df['observed_quintile_spread'].mean() * 100:.1f} pts vs {default_df['observed_quintile_spread'].mean() * 100:.1f} pts).
- Its quintile spread ranged from {final_df['observed_quintile_spread'].min() * 100:.1f} to {final_df['observed_quintile_spread'].max() * 100:.1f} pts across splits.
- Customer rankings were very similar across splits (rank correlation {ranking['rank_correlation']:.2f}).
  {ranking['priority1_overlap']:.0%} of Priority 1 customers stayed in Priority 1, and {ranking['within_one_segment_every_split']:.0%}
  of customers never moved more than one segment.

{conclusion} The rankings the retention team would use do not depend much on how the data was split.
"""
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(text)


def main():
    config = json.loads((FINAL_DIR / "final_model_config.json").read_text())
    df = pd.read_csv(CLEANED_PATH)
    y = df["outcome"]
    X = df.drop(columns=["outcome"])
    feature_cols = [c for c in X.columns if c != "treatment"]
    scale_cols = get_scale_cols(X)

    print(f"Checking the final model on {len(SEEDS)} data splits...")
    final_df, final_scores = run_seeds(X, y, config["model_type"], config["params"], scale_cols, feature_cols)
    default_df, _ = run_seeds(X, y, config["model_type"], config["default_params"], scale_cols, feature_cols)
    ranking = ranking_stability(final_scores)

    print(f"\n  {'seed':>6}{'Sprint 7 spread':>18}{'final spread':>15}")
    for (_, d), (_, f) in zip(default_df.iterrows(), final_df.iterrows()):
        print(f"  {int(f['seed']):>6}{d['observed_quintile_spread']:>18.4f}{f['observed_quintile_spread']:>15.4f}")
    print(f"\n  Rank correlation between splits:     {ranking['rank_correlation']:.3f}")
    print(f"  Priority 1 customers that stay put:  {ranking['priority1_overlap']:.0%}")
    print(f"  Customers within one segment:        {ranking['within_one_segment_every_split']:.1%}")

    final_df.to_csv(FINAL_DIR / "stability_by_seed.csv", index=False)
    config["stability"] = ranking
    (FINAL_DIR / "final_model_config.json").write_text(json.dumps(config, indent=2))
    write_report(config, final_df, default_df, ranking)
    print(f"\nSaved {REPORT_PATH} and {FINAL_DIR}/stability_by_seed.csv")


if __name__ == "__main__":
    main()