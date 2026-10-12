import contextlib
import io
import json

import pandas as pd

import uplift_metrics as um
from cross_validate import CLEANED_PATH, MODEL_TYPES, get_scale_cols, make_folds, run_cv
from evaluate_and_segment import choose_best_model
from train_models import OUTPUT_DIR

SEEDS = [42, 7, 123, 2024, 31337]
STABILITY_DIR = OUTPUT_DIR / "stability"
METRICS = [
    "observed_quintile_spread",
    "observed_top_quintile_uplift",
    "auuc_per_1000",
    "incremental_per_1000_at_20pct",
    "incremental_per_1000_at_30pct",
]
LR, RF = "logistic_t_learner", "random_forest_t_learner"


def run_all_seeds(X, y, feature_cols, scale_cols):
    t, o = X["treatment"].values, y.values
    rows = []
    for seed in SEEDS:
        folds = make_folds(X, y, seed=seed)
        for model_type in MODEL_TYPES:
            with contextlib.redirect_stdout(io.StringIO()):
                oof, model_name = run_cv(X, y, folds, scale_cols, feature_cols, model_type, verbose=False)
            row = {"seed": seed, "model_name": model_name}
            row.update({k: float(v) for k, v in um.core_metrics(oof["uplift_score"].values, t, o).items()})
            rows.append(row)
        print(f"  seed {seed:>5}: both models scored")
    return pd.DataFrame(rows)


def summarize(seed_df):
    # Winner per seed
    winners = {}
    for seed, g in seed_df.groupby("seed"):
        best, _ = choose_best_model(g.reset_index(drop=True))
        winners[int(seed)] = best

    # Gap between models (LR minus RF)
    wide = seed_df.pivot(index="seed", columns="model_name", values=METRICS)
    gaps = {}
    for m in METRICS:
        diff = wide[m][LR] - wide[m][RF]
        gaps[m] = {"mean": float(diff.mean()), "min": float(diff.min()), "max": float(diff.max()),
                   "seeds_logistic_ahead": int((diff > 0).sum()), "per_seed": {int(k): float(v) for k, v in diff.items()}}

    # Spread of each model's metrics across seeds
    per_model = {}
    for name, g in seed_df.groupby("model_name"):
        per_model[name] = {m: {"mean": float(g[m].mean()), "std": float(g[m].std(ddof=1)),
                               "min": float(g[m].min()), "max": float(g[m].max())} for m in METRICS}

    win_counts = pd.Series(winners).value_counts().to_dict()
    consistent = len(win_counts) == 1
    return {
        "seeds": SEEDS,
        "winner_by_selection_rule_per_seed": winners,
        "wins": {k: int(v) for k, v in win_counts.items()},
        "same_winner_every_seed": consistent,
        "logistic_minus_rf_by_metric": gaps,
        "per_model_metric_spread": per_model,
        "interpretation": (
            "The same model wins under every split, so the selection is robust to how the data is split."
            if consistent else
            "The winner changes with the split, so the models are not reliably different; this supports "
            "treating them as tied (see the tie rule in evaluate_and_segment.py)."
        ),
    }


def print_summary(summary):
    print(f"\n{'=' * 78}\nModel comparison across {len(SEEDS)} different data splits\n{'=' * 78}")
    for seed, w in summary["winner_by_selection_rule_per_seed"].items():
        print(f"  seed {seed:>5}: winner by selection rule = {w}")
    print(f"  Wins: {summary['wins']}")

    print(f"\n  Logistic minus Random Forest:")
    for m, g in summary["logistic_minus_rf_by_metric"].items():
        print(f"    {m:<32} mean {g['mean']:+8.4f}   range [{g['min']:+.4f}, {g['max']:+.4f}]   "
              f"LR ahead in {g['seeds_logistic_ahead']}/{len(SEEDS)}")

    print(f"\n  How much each model moves between splits (quintile spread):")
    for name, d in summary["per_model_metric_spread"].items():
        s = d["observed_quintile_spread"]
        print(f"    {name:<26} mean {s['mean']:.4f}   std {s['std']:.4f}   range [{s['min']:.4f}, {s['max']:.4f}]")

    print(f"\n  >>> {summary['interpretation']}")


def main():
    df = pd.read_csv(CLEANED_PATH)
    y = df["outcome"]
    X = df.drop(columns=["outcome"])
    feature_cols = [c for c in X.columns if c != "treatment"]
    scale_cols = get_scale_cols(X)
    print(f"Repeating the model comparison on {len(SEEDS)} different 5-fold splits ({len(X):,} customers)...")

    seed_df = run_all_seeds(X, y, feature_cols, scale_cols)
    summary = summarize(seed_df)
    print_summary(summary)

    STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(STABILITY_DIR / "seed_metrics.csv", index=False)
    with open(STABILITY_DIR / "stability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {STABILITY_DIR}/seed_metrics.csv, stability_summary.json")


if __name__ == "__main__":
    main()