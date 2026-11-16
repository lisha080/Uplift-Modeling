# Reproducibility Check (Sprint 10)

We saved the key results, reran the full pipeline from raw data (`run_pipeline.py`, 17 steps),
and compared the new results with the saved ones. Numbers must match to within 1e-09.

| Output | Reproduced | Detail |
|---|---|---|
| Cleaned dataset | Yes | sha256 ebda5b77895d vs ebda5b77895d |
| Final model settings | Yes | {"C": 0.1, "penalty": "l1"} |
| Uplift scores (final model) | Yes | max difference 0.00e+00 |
| Customer ranking | Yes | max difference 0.00e+00 |
| Priority group summary | Yes | max difference 0.00e+00 |
| SHAP factor importance | Yes | max difference 0.00e+00 |
| ROI by outreach size | Yes | max difference 0.00e+00 |
| Targeting comparison | Yes | max difference 0.00e+00 |

**Result:** Every output was reproduced exactly.

All random steps use fixed seeds (data generation, train/test split, cross-validation folds,
Random Forest), so the same code on the same machine gives the same results. Different library
versions can change Random Forest results slightly; the selected Logistic Regression model is not affected.
Package versions are listed in `requirements.txt`.
