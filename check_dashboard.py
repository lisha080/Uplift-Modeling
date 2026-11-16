"""
Checks that the dashboard matches the saved model outputs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import roi
import uplift_metrics as um
from dashboard_data import load_all
from targeting_comparison import compare
from train_models import OUTPUT_DIR

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def main():
    d = load_all()
    final = pd.read_csv(OUTPUT_DIR / "final_model" / "oof_predictions.csv")
    seg = d.segments.set_index("priority_segment")

    check("same number of customers as the final model", len(d.ranked) == len(final) == len(d.strategy_scores),
          f"{len(d.ranked):,}")
    check("ranked list uses the final model's uplift scores",
          np.allclose(np.sort(d.ranked["uplift_score"].values), np.sort(final["uplift_score"].values)))
    check("strategy scores use the final model's uplift scores",
          np.allclose(d.strategy_scores["uplift_final"].values, final["uplift_score"].values))
    check("ranked list is sorted from highest uplift down",
          (np.diff(d.ranked["uplift_score"].values) <= 1e-12).all()
          and (d.ranked["priority_rank"].values == np.arange(1, len(d.ranked) + 1)).all())

    syn = pd.read_csv(Path("data/processed/synthetic_data.csv"))
    top10_idx = final.sort_values("uplift_score", ascending=False, kind="stable").head(10)["customer_index"]
    top10_model = syn.loc[top10_idx, "CustomerId"].tolist()
    check("top 10 customers shown match the final model's top 10", d.ranked.head(10)["CustomerId"].tolist() == top10_model)

    counts = d.ranked["priority_segment"].value_counts()
    check("priority group sizes match segment_summary.csv",
          all(counts[g] == seg.loc[g, "n_customers"] for g in seg.index), counts.sort_index().to_dict())

    q = um.quintile_table(final["uplift_score"], final["treatment"], final["outcome"]).set_index("quintile")
    check("Priority 1 observed uplift matches a fresh calculation",
          np.isclose(seg.loc["Priority 1", "observed_uplift"], q.loc[4, "observed_uplift"]),
          f"{seg.loc['Priority 1', 'observed_uplift']:.4f}")

    shap_vals = pd.read_csv(OUTPUT_DIR / "shap" / "shap_values.csv")
    factor_cols = [c for c in shap_vals.columns if c not in ("customer_index", "uplift_score", "base_value")]
    recomputed = shap_vals[factor_cols].abs().mean() * 100
    top_feature = recomputed.idxmax()
    check("top SHAP factor matches shap_values.csv", d.importance.iloc[0]["feature"] == top_feature, top_feature)

    s = d.strategy_scores
    saved = d.roi_table.set_index("outreach_share")
    ok = True
    for share in roi.OUTREACH_SHARES:
        r = roi.simulate(s["uplift_final"].values, share, treatment=s["treatment"].values, outcome=s["outcome"].values)
        for col in ["predicted_incremental", "predicted_net_benefit", "predicted_roi", "observed_net_benefit"]:
            ok &= np.isclose(r[col], saved.loc[share, col])
    check("ROI simulator matches roi_by_outreach_size.csv at every outreach size", ok)

    comp = compare(s).set_index(["strategy", "outreach_share"])
    saved_comp = pd.read_csv(OUTPUT_DIR / "roi" / "targeting_comparison.csv").set_index(["strategy", "outreach_share"])
    check("targeting comparison matches targeting_comparison.csv",
          np.allclose(comp["net_benefit"].values, saved_comp.loc[comp.index, "net_benefit"].values))

    summary = json.loads((OUTPUT_DIR / "roi" / "roi_summary.json").read_text())
    check("ROI assumptions match roi_summary.json",
          summary["contact_cost"] == roi.CONTACT_COST and summary["retention_value"] == roi.RETENTION_VALUE)

    try:
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).with_name("dashboard.py")), default_timeout=120).run()
        check("dashboard loads with no errors", len(app.exception) == 0,
              "; ".join(str(e.value) for e in app.exception))
        shown = {m.label: m.value for m in app.metric}
        default = saved.loc[0.3]
        check("dashboard default (30%) net benefit matches saved ROI table",
              shown.get("Net benefit") == f"${default['predicted_net_benefit']:,.0f}", shown.get("Net benefit"))
        check("dashboard Priority 1 uplift matches segment_summary.csv",
              shown.get("Priority 1 uplift") == f"+{seg.loc['Priority 1', 'observed_uplift'] * 100:.1f} pts",
              shown.get("Priority 1 uplift"))
        best = saved["predicted_net_benefit"].idxmax()
        check("dashboard recommended outreach matches saved ROI table",
              shown.get("Recommended outreach") == f"Top {best:.0%}", shown.get("Recommended outreach"))
    except ImportError:
        print("SKIP  dashboard run (streamlit not installed)")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()