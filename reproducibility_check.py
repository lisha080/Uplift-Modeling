"""
Reruns the full pipeline and checks that it reproduces the saved results.
Run after the pipeline has been run once (python run_pipeline.py).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_pipeline
from train_models import OUTPUT_DIR

VALIDATION_DIR = OUTPUT_DIR / "validation"
REPORT_PATH = Path("reports") / "reproducibility.md"
TOL = 1e-9

TABLES = {
    "Uplift scores (final model)": (OUTPUT_DIR / "final_model" / "oof_predictions.csv", ["uplift_score"]),
    "Customer ranking": (OUTPUT_DIR / "ranking" / "ranked_customers.csv", ["uplift_score"]),
    "Priority group summary": (OUTPUT_DIR / "ranking" / "segment_summary.csv", ["observed_uplift", "mean_predicted_uplift"]),
    "SHAP factor importance": (OUTPUT_DIR / "shap" / "feature_importance.csv", ["avg_impact_pts"]),
    "ROI by outreach size": (OUTPUT_DIR / "roi" / "roi_by_outreach_size.csv", ["predicted_net_benefit", "observed_net_benefit"]),
    "Targeting comparison": (OUTPUT_DIR / "roi" / "targeting_comparison.csv", ["net_benefit"]),
}
CONFIG_PATH = OUTPUT_DIR / "final_model" / "final_model_config.json"
DATA_PATH = Path("data/processed/cleaned_data.csv")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def snapshot():
    missing = [str(p) for p, _ in TABLES.values() if not p.exists()] + \
              [str(p) for p in (CONFIG_PATH, DATA_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError("Run python run_pipeline.py first. Missing:\n  " + "\n  ".join(missing))
    snap = {name: pd.read_csv(path) for name, (path, _) in TABLES.items()}
    snap["config"] = json.loads(CONFIG_PATH.read_text())
    snap["data_hash"] = file_hash(DATA_PATH)
    return snap


def compare(before, after):
    rows = [{"check": "Cleaned dataset", "match": before["data_hash"] == after["data_hash"],
             "detail": f"sha256 {before['data_hash']} vs {after['data_hash']}"}]
    rows.append({"check": "Final model settings", "match": before["config"] == after["config"],
                 "detail": json.dumps(after["config"].get("params", {}))})
    for name, (_, cols) in TABLES.items():
        b, a = before[name], after[name]
        if b.shape != a.shape:
            rows.append({"check": name, "match": False, "detail": f"shape {b.shape} vs {a.shape}"})
            continue
        diff = max(float(np.nanmax(np.abs(b[c].values - a[c].values))) for c in cols)
        same_order = True
        if "CustomerId" in b.columns:
            same_order = (b["CustomerId"].values == a["CustomerId"].values).all()
        rows.append({"check": name, "match": diff <= TOL and same_order,
                     "detail": f"max difference {diff:.2e}" + ("" if same_order else ", customer order changed")})
    return pd.DataFrame(rows)


def write_report(result):
    lines = ["| Output | Reproduced | Detail |", "|---|---|---|"]
    lines += [f"| {r.check} | {'Yes' if r.match else 'No'} | {r.detail} |" for r in result.itertuples()]
    verdict = ("Every output was reproduced exactly." if result["match"].all()
               else "Some outputs changed; see the rows marked No.")
    REPORT_PATH.write_text(f"""# Reproducibility Check (Sprint 10)

We saved the key results, reran the full pipeline from raw data (`run_pipeline.py`, {len(run_pipeline.STEPS)} steps),
and compared the new results with the saved ones. Numbers must match to within {TOL:g}.

{chr(10).join(lines)}

**Result:** {verdict}

All random steps use fixed seeds (data generation, train/test split, cross-validation folds,
Random Forest), so the same code on the same machine gives the same results. Different library
versions can change Random Forest results slightly; the selected Logistic Regression model is not affected.
Package versions are listed in `requirements.txt`.
""")


def main():
    print("Saving current results...")
    before = snapshot()
    print("Rerunning the full pipeline...\n")
    run_pipeline.run(run_pipeline.STEPS)
    after = snapshot()

    result = compare(before, after)
    print("\n" + "\n".join(f"{'PASS' if r.match else 'FAIL'}  {r.check}  ({r.detail})" for r in result.itertuples()))
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(VALIDATION_DIR / "reproducibility.csv", index=False)
    write_report(result)
    print(f"\n{int(result['match'].sum())}/{len(result)} outputs reproduced. Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()