"""
Tunes the selected uplift model and re-evaluates it with the same uplift metrics.

1. Try a small set of settings using 5-fold CV on the training split only.
2. Check the best setting against the default on the 2,000 holdout customers.
3. Keep the tuned setting only if it is not worse on the holdout.
4. Re-score all customers with the final setting so it can be compared with Sprint 7.

Run after evaluate_and_segment.py.
"""

import contextlib
import io
import itertools
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import uplift_metrics as um
from cross_validate import CLEANED_PATH, get_scale_cols, make_folds, run_cv
from split_and_scale import RANDOM_SEED, TEST_SIZE
from train_models import OUTPUT_DIR, train_t_learner

FINAL_DIR = OUTPUT_DIR / "final_model"
GRIDS = {
    "logistic": {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0], "penalty": ["l2", "l1"]},
    "random_forest": {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [10, 25, 50, 100]},
}
DEFAULTS = {
    "logistic": {"C": 1.0, "penalty": "l2"},
    "random_forest": {"max_depth": 8, "min_samples_leaf": 10},
}
METRICS = ["observed_quintile_spread", "observed_top_quintile_uplift", "auuc_per_1000",
           "incremental_per_1000_at_20pct", "incremental_per_1000_at_30pct"]


def all_settings(model_type):
    grid = GRIDS[model_type]
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def score(scores, t, o):
    m = um.core_metrics(scores, t, o)
    return {k: float(m[k]) for k in METRICS}


def search(X_train, y_train, model_type, scale_cols, feature_cols):
    # Rank settings by AUUC: it uses the whole ranking, so it is less noisy than quintile spread
    X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
    folds = make_folds(X_train, y_train)
    t, o = X_train["treatment"].values, y_train.values
    rows = []
    for params in all_settings(model_type):
        oof, _ = run_cv(X_train, y_train, folds, scale_cols, feature_cols, model_type, verbose=False, params=params)
        rows.append({**params, **score(oof["uplift_score"].values, t, o)})
        print(f"  {params}  AUUC {rows[-1]['auuc_per_1000']:.2f}  spread {rows[-1]['observed_quintile_spread']:.4f}")
    return pd.DataFrame(rows).sort_values("auuc_per_1000", ascending=False).reset_index(drop=True)


def holdout_scores(X_train, y_train, X_test, model_type, params, scale_cols, feature_cols):
    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = StandardScaler().fit(X_train[scale_cols])
    X_train[scale_cols] = scaler.transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])
    with contextlib.redirect_stdout(io.StringIO()):
        m_t, m_c, _ = train_t_learner(X_train, y_train, feature_cols, model_type, params)
    return m_t.predict_proba(X_test[feature_cols])[:, 1] - m_c.predict_proba(X_test[feature_cols])[:, 1]


def main():
    selected = json.loads((OUTPUT_DIR / "selection_summary.json").read_text())["selected_model"]
    model_type = selected.replace("_t_learner", "")
    default = DEFAULTS[model_type]

    df = pd.read_csv(CLEANED_PATH)
    y = df["outcome"]
    X = df.drop(columns=["outcome"])
    feature_cols = [c for c in X.columns if c != "treatment"]
    scale_cols = get_scale_cols(X)
    # Same split as split_and_scale.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=X["treatment"])

    print(f"Tuning {selected} on {len(X_train):,} training customers...")
    results = search(X_train, y_train, model_type, scale_cols, feature_cols)
    best = {k: results.iloc[0][k] for k in default}
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    print(f"\n  Best setting: {best}   Default: {default}")

    # Holdout check on customers the search never saw
    t, o = X_test["treatment"].values, y_test.values
    hold_default = score(holdout_scores(X_train, y_train, X_test, model_type, default, scale_cols, feature_cols), t, o)
    hold_best = score(holdout_scores(X_train, y_train, X_test, model_type, best, scale_cols, feature_cols), t, o)
    print(f"\n  Holdout ({len(X_test):,} customers)      default     tuned")
    for k in METRICS:
        print(f"    {k:<32}{hold_default[k]:>9.4f}{hold_best[k]:>10.4f}")

    keep_tuned = (best != default
                  and hold_best["observed_quintile_spread"] >= hold_default["observed_quintile_spread"]
                  and hold_best["auuc_per_1000"] >= hold_default["auuc_per_1000"])
    final = best if keep_tuned else default
    reason = ("Tuned setting was not worse than the default on the holdout customers."
              if keep_tuned else "Tuned setting did not hold up on the holdout customers, so the default is kept.")
    print(f"\n  >>> Final setting: {final}  ({'tuned' if keep_tuned else 'default'})")
    print(f"      {reason}")

    # Re-score all customers with the final setting (same folds as Sprint 7)
    folds = make_folds(X, y)
    oof, _ = run_cv(X, y, folds, scale_cols, feature_cols, model_type, verbose=False, params=final)
    sprint7 = pd.read_csv(OUTPUT_DIR / selected / "oof_predictions.csv")
    t_all, o_all = oof["treatment"].values, oof["outcome"].values
    before = score(sprint7["uplift_score"].values, t_all, o_all)
    after = score(oof["uplift_score"].values, t_all, o_all)
    print(f"\n  All {len(X):,} customers              Sprint 7     final")
    for k in METRICS:
        print(f"    {k:<32}{before[k]:>9.4f}{after[k]:>10.4f}")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(FINAL_DIR / "oof_predictions.csv", index=False)
    results.to_csv(FINAL_DIR / "tuning_results.csv", index=False)
    config = {
        "model_name": selected,
        "model_type": model_type,
        "params": final,
        "default_params": default,
        "tuned": keep_tuned,
        "reason": reason,
        "holdout": {"default": hold_default, "tuned": hold_best},
        "all_customers": {"sprint7": before, "final": after},
    }
    (FINAL_DIR / "final_model_config.json").write_text(json.dumps(config, indent=2))
    print(f"\nSaved {FINAL_DIR}/final_model_config.json, oof_predictions.csv, tuning_results.csv")


if __name__ == "__main__":
    main()