"""
Simulates net benefit and ROI by outreach size for the final model.
Run after check_final_model.py.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import roi
from train_models import OUTPUT_DIR

FINAL_PATH = OUTPUT_DIR / "final_model" / "oof_predictions.csv"
ROI_DIR = OUTPUT_DIR / "roi"
REPORT_PATH = Path("reports") / "roi_assumptions.md"


def load_final_scores():
    if not FINAL_PATH.exists():
        raise FileNotFoundError(f"Missing {FINAL_PATH}. Run check_final_model.py first.")
    return pd.read_csv(FINAL_PATH)


def make_chart(table, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = table["outreach_share"] * 100
    ax.plot(x, table["predicted_net_benefit"] / 1000, color="#2a78d6", lw=2.2, label="Predicted (model)")
    ax.plot(x, table["observed_net_benefit"] / 1000, color="#d1603d", lw=1.8, ls="--", label="Observed (real outcomes)")
    ax.axhline(0, color="#b9bfc6", lw=1)
    ax.set_xlabel("Share of customers contacted (%), highest uplift first")
    ax.set_ylabel("Net benefit ($ thousands)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(table, summary):
    best = summary["best_outreach_size"]
    rows = table[table["outreach_share"].isin([0.1, 0.2, 0.3, 0.5, 0.7, 1.0])]
    lines = [
        "| Customers contacted | Extra customers kept (predicted / observed) | Cost | Net benefit (predicted / observed) | ROI (predicted / observed) |",
        "|---|---|---|---|---|",
    ]
    for _, r in rows.iterrows():
        lines.append(
            f"| {r.outreach_share:.0%} ({int(r.n_contacted):,}) | {r.predicted_incremental:,.0f} / {r.observed_incremental:,.0f} "
            f"| ${r.predicted_cost:,.0f} | ${r.predicted_net_benefit:,.0f} / ${r.observed_net_benefit:,.0f} "
            f"| {r.predicted_roi:.0%} / {r.observed_roi:.0%} |"
        )
    table_md = "\n".join(lines)

    text = f"""# ROI Simulation: Assumptions and Results (Sprint 9)

## What we calculate

For a campaign that contacts the top N customers ranked by uplift score:

- **Cost** = N x contact cost
- **Extra customers kept** = customers who stay *because of* the outreach
- **Benefit** = extra customers kept x value of keeping one customer
- **Net benefit** = benefit - cost
- **ROI** = net benefit / cost (100% means every $1 spent returns $2)

## Assumptions

| Assumption | Value used | Notes |
|---|---|---|
| Cost to contact one customer | ${roi.CONTACT_COST:,.0f} | Staff time, offer and channel cost. Illustrative; replace with the sponsor's real figure. |
| Value of keeping one customer | ${roi.RETENTION_VALUE:,.0f} | Profit from a customer who would otherwise have left. Illustrative. |
| Customers in the simulation | {summary['n_customers']:,} | Every customer is scored by a model that never saw them (5-fold cross-validation). |
| Uplift score meaning | Probability | An uplift of 0.20 means outreach raises the chance of staying by 20 points. |
| Who gets contacted | Highest uplift first | One contact per customer, no capacity limits. |
| Break-even uplift | {summary['break_even_uplift']:.3f} | Contact cost / retention value. Customers below this cost more to contact than they return. |

Limitations:
- The data is synthetic (Kaggle churn data plus generated treatment and behavior columns), so dollar
  amounts show how the method works, not what a real campaign would earn.
- Costs and values are the same for every customer. In practice, high-balance customers are worth more.
- The simulation assumes outreach works the same way it did in the experiment.

## Checking the math (#88)

Every number is calculated two ways:

- **Predicted**: adds up the model's uplift scores for the customers contacted.
- **Observed**: uses what actually happened. Among the customers contacted, we compare those who
  received outreach in the experiment with those who did not.

If the model is well calibrated the two should be close. Largest gap in extra customers kept across
all outreach sizes: {summary['max_abs_gap_incremental']:.0f} customers
({summary['max_rel_gap_incremental']:.0%} of the observed value). The formulas are also checked
against hand-worked cases in `test_roi.py`.

## Results

{table_md}

- **Best outreach size:** contact the top **{best['outreach_share']:.0%}** ({int(best['n_contacted']):,} customers).
  Predicted net benefit **${best['predicted_net_benefit']:,.0f}** (observed ${best['observed_net_benefit']:,.0f}),
  ROI {best['predicted_roi']:.0%}.
- Contacting everyone gives a net benefit of ${summary['contact_everyone']['predicted_net_benefit']:,.0f},
  so targeting saves money on customers who would not benefit.

Chart: `outputs/roi/roi_curve.png`. Full table: `outputs/roi/roi_by_outreach_size.csv`.
"""
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(text)


def main():
    pred = load_final_scores()
    s, t, o = pred["uplift_score"].values, pred["treatment"].values, pred["outcome"].values
    print(f"Customers: {len(pred):,}   Contact cost: ${roi.CONTACT_COST:,.0f}   "
          f"Retention value: ${roi.RETENTION_VALUE:,.0f}")

    table = roi.roi_table(s, treatment=t, outcome=o)
    best = table.loc[table["predicted_net_benefit"].idxmax()]

    gap = (table["predicted_incremental"] - table["observed_incremental"]).abs()
    summary = {
        "contact_cost": roi.CONTACT_COST,
        "retention_value": roi.RETENTION_VALUE,
        "n_customers": int(len(pred)),
        "break_even_uplift": roi.break_even_uplift(),
        "best_outreach_size": {k: (int(v) if k == "n_contacted" else float(v)) for k, v in best.items()},
        "contact_everyone": {k: float(v) for k, v in table.iloc[-1].items()},
        "max_abs_gap_incremental": float(gap.max()),
        "max_rel_gap_incremental": float((gap / table["observed_incremental"].abs()).max()),
    }

    show = table[["outreach_share", "n_contacted", "predicted_incremental", "observed_incremental",
                  "predicted_cost", "predicted_net_benefit", "observed_net_benefit", "predicted_roi"]]
    print(show.round(2).to_string(index=False))
    print(f"\nBest outreach size: top {best.outreach_share:.0%} -> predicted net benefit "
          f"${best.predicted_net_benefit:,.0f} (observed ${best.observed_net_benefit:,.0f})")

    ROI_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(ROI_DIR / "roi_by_outreach_size.csv", index=False)
    (ROI_DIR / "roi_summary.json").write_text(json.dumps(summary, indent=2))
    make_chart(table, ROI_DIR / "roi_curve.png")
    write_report(table, summary)
    print(f"\nSaved: {ROI_DIR}/roi_by_outreach_size.csv, roi_summary.json, roi_curve.png, {REPORT_PATH}")


if __name__ == "__main__":
    main()