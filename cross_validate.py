"""
Out-of-fold (OOF) uplift predictions for ALL customers.

How each fold works:
  1. Split customers 80/20, stratified on treatment x outcome.
  2. Fit on the 80% only (never on the held-out 20%).
  3. Train the T-Learner (same models / hyperparameters as train_models.py).
  4. Score the held-out 20%.

Output per model: outputs/<model_name>/oof_predictions.csv
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from train_models import OUTPUT_DIR, SEGMENT_READY_COLS, train_t_learner

CLEANED_PATH = Path("data/processed/cleaned_data.csv")
N_FOLDS = 5
RANDOM_SEED = 42
MODEL_TYPES = ["logistic", "random_forest"]


def get_scale_cols(X):
    """Same rule as split_and_scale.py: scale numeric, non-binary columns."""
    binary_cols = [c for c in X.columns if set(X[c].dropna().unique()).issubset({0, 1})]
    return [c for c in X.select_dtypes(include=[np.number]).columns if c not in binary_cols]


def make_folds(X, y):
    """The single definition of the CV folds. Anything compared against the uplift
    models (e.g. the traditional baseline) must use these exact folds.

    Stratified on treatment x outcome so every fold has the same arm sizes AND the
    same retention mix in each arm.
    """
    strata = X["treatment"].astype(str) + "_" + y.astype(str)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return list(skf.split(X, strata))


def run_cv(X, y, folds, scale_cols, feature_cols, model_type):
    n = len(X)
    p_treated = np.full(n, np.nan)
    p_control = np.full(n, np.nan)
    fold_id = np.full(n, -1)

    for fold, (train_pos, test_pos) in enumerate(folds):
        X_tr, X_te = X.iloc[train_pos].copy(), X.iloc[test_pos].copy()
        y_tr = y.iloc[train_pos]

        scaler = StandardScaler().fit(X_tr[scale_cols])  # train fold only
        X_tr[scale_cols] = scaler.transform(X_tr[scale_cols])
        X_te[scale_cols] = scaler.transform(X_te[scale_cols])

        with contextlib.redirect_stdout(io.StringIO()):  # silence per-fold prints
            m_treated, m_control, model_name = train_t_learner(
                X_tr, y_tr, feature_cols, model_type=model_type
            )

        p_treated[test_pos] = m_treated.predict_proba(X_te[feature_cols])[:, 1]
        p_control[test_pos] = m_control.predict_proba(X_te[feature_cols])[:, 1]
        fold_id[test_pos] = fold
        print(f"    fold {fold + 1}/{len(folds)} done ({len(test_pos)} customers scored)")

    assert not np.isnan(p_treated).any(), "Some customers were never scored"

    out = pd.DataFrame({
        "customer_index": X.index,  # original customer row id
        "treatment": X["treatment"].values,
        "outcome": y.values,
        "p_treated": p_treated,
        "p_control": p_control,
        "uplift_score": p_treated - p_control,
        "fold": fold_id,
    })
    for col in SEGMENT_READY_COLS:  # unscaled originals
        if col in X.columns:
            out[col] = X[col].values
    return out, model_name


def main():
    df = pd.read_csv(CLEANED_PATH)
    y = df["outcome"]
    X = df.drop(columns=["outcome"])
    feature_cols = [c for c in X.columns if c != "treatment"]
    scale_cols = get_scale_cols(X)
    print(f"Loaded {len(X)} customers | {N_FOLDS}-fold CV | scaling {len(scale_cols)} columns per fold")

    folds = make_folds(X, y)  # identical folds for every model

    for model_type in MODEL_TYPES:
        print(f"\n{'=' * 60}\nCross-validating: {model_type}\n{'=' * 60}")
        oof, model_name = run_cv(X, y, folds, scale_cols, feature_cols, model_type)
        model_dir = OUTPUT_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        oof.to_csv(model_dir / "oof_predictions.csv", index=False)
        print(f"  Saved {model_dir}/oof_predictions.csv  (mean uplift {oof['uplift_score'].mean():.4f})")


if __name__ == "__main__":
    main()