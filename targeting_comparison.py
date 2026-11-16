"""
Compares uplift, churn-risk and random targeting on net benefit and ROI.
Run after roi_simulation.py and baseline_comparison.py.
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

SCORES_PATH = OUTPUT_DIR / "strategy_scores.csv"
ROI_DIR = OUTPUT_DIR / "roi"
REPORT_PATH = Path("reports") / "targeting_comparison.md"

UPLIFT = "Uplift targeting"
CHURN = "Churn-risk targeting"
RANDOM = "Random targeting"
COLORS = {UPLIFT: "#2a78d6", CHURN: "#d1603d", RANDOM: "#7f7f7f"}
CI_SHARES = [0.2, 0.3, 0.5]
N_BOOT = 500


def load_strategy_scores():
    if not SCORES_PATH.exists():
        raise FileNotFoundError(f"Missing {SCORES_PATH}. Run baseline_comparison.py first.")
    df = pd.read_csv(SCORES_PATH)
    if "uplift_final" not in df.columns:
        raise ValueError("strategy_scores.csv has no uplift_final column. Run check_final_model.py, "
                         "then baseline_comparison.py.")
    return df


def compare(df, contact_cost=roi.CONTACT_COST, retention_value=roi.RETENTION_VALUE, shares=roi.OUTREACH_SHARES):
    t, o = df["treatment"].values, df["outcome"].values
    rows = []
    for share in shares:
        for name, col in [(UPLIFT, "uplift_final"), (CHURN, "churn_risk")]:
            k = roi.n_contacted(len(df), share)
            inc = roi.observed_incremental(df[col].values, t, o, k)
            rows.append({"strategy": name, "outreach_share": float(share), "n_contacted": k,
                         "observed_incremental": inc, **roi.money(inc, k, contact_cost, retention_value)})
        r = roi.random_targeting(t, o, share, contact_cost, retention_value)
        rows.append({"strategy": RANDOM, **{k.replace("observed_", ""): v for k, v in r.items()
                                            if k != "observed_incremental"},
                     "observed_incremental": r["observed_incremental"]})
    out = pd.DataFrame(rows)
    return out[["strategy", "outreach_share", "n_contacted", "observed_incremental",
                "cost", "benefit", "net_benefit", "roi"]]


def bootstrap_uplift_vs_churn(df, retention_value=roi.RETENTION_VALUE):
    rng = np.random.default_rng(42)
    s_u, s_c = df["uplift_final"].values, df["churn_risk"].values
    t, o = df["treatment"].values, df["outcome"].values
    n = len(df)
    out = {}
    for share in CI_SHARES:
        k = roi.n_contacted(n, share)
        diffs = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            inc_u = roi.observed_incremental(s_u[idx], t[idx], o[idx], k)
            inc_c = roi.observed_incremental(s_c[idx], t[idx], o[idx], k)
            diffs.append((inc_u - inc_c) * retention_value)
        diffs = np.array(diffs)
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        out[f"{int(share * 100)}%"] = {"mean_diff": float(np.nanmean(diffs)), "ci_low": float(lo),
                                       "ci_high": float(hi), "uplift_better": bool(lo > 0)}
    return out


def make_chart(table, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in [UPLIFT, CHURN, RANDOM]:
        d = table[table["strategy"] == name]
        ax.plot(d["outreach_share"] * 100, d["net_benefit"] / 1000, color=COLORS[name], lw=2,
                ls="-" if name == UPLIFT else "--", label=name)
    ax.axhline(0, color="#b9bfc6", lw=1)
    ax.set_xlabel("Share of customers contacted (%)")
    ax.set_ylabel("Net benefit ($ thousands, real outcomes)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def dollars(x):
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def write_report(table, ci):
    piv = table.pivot(index="outreach_share", columns="strategy", values="net_benefit")
    inc = table.pivot(index="outreach_share", columns="strategy", values="observed_incremental")
    best = {name: piv[name].idxmax() for name in [UPLIFT, CHURN, RANDOM]}

    lines = ["| Customers contacted | Uplift: extra kept / net benefit | Churn risk: extra kept / net benefit "
             "| Random: extra kept / net benefit |", "|---|---|---|---|"]
    for share in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        cells = [f"{inc.loc[share, n]:,.0f} / ${piv.loc[share, n]:,.0f}" for n in [UPLIFT, CHURN, RANDOM]]
        lines.append(f"| {share:.0%} | " + " | ".join(cells) + " |")

    ci_lines = ["| Customers contacted | Uplift minus churn risk (net benefit) | 95% interval | Clear winner? |",
                "|---|---|---|---|"]
    for share, r in ci.items():
        ci_lines.append(f"| {share} | {dollars(r['mean_diff'])} | {dollars(r['ci_low'])} to {dollars(r['ci_high'])} "
                        f"| {'Yes, uplift' if r['uplift_better'] else 'Not clear'} |")

    s30 = 0.3
    text = f"""# Uplift-Based vs Traditional Targeting (Sprint 9)

All three strategies contact the same number of customers at the same cost
(${roi.CONTACT_COST:,.0f} per contact, ${roi.RETENTION_VALUE:,.0f} per customer kept). Only *who* is contacted changes:

- **Uplift targeting**: customers whose chance of staying rises most with outreach (final tuned model).
- **Churn-risk targeting**: customers most likely to leave, the traditional approach.
- **Random targeting**: a random sample of customers.

Extra customers kept are measured from real outcomes in the experiment, not from model predictions,
so each strategy is judged fairly.

## Results

{chr(10).join(lines)}

## Is the difference real?

Bootstrap with {N_BOOT} resamples of the 10,000 customers:

{chr(10).join(ci_lines)}

## Takeaways

- With 30% of customers contacted, uplift targeting keeps {inc.loc[s30, UPLIFT]:,.0f} extra customers, compared with
  {inc.loc[s30, CHURN]:,.0f} for churn-risk targeting and {inc.loc[s30, RANDOM]:,.0f} for random targeting. That is
  ${piv.loc[s30, UPLIFT] - piv.loc[s30, CHURN]:,.0f} more net benefit than churn-risk targeting for the same budget.
- Best outreach size for each strategy (real outcomes): uplift {best[UPLIFT]:.0%} (${piv[UPLIFT].max():,.0f}),
  churn risk {best[CHURN]:.0%} (${piv[CHURN].max():,.0f}), random {best[RANDOM]:.0%} (${piv[RANDOM].max():,.0f}).
- Customers at highest risk of leaving are not always the ones outreach helps most. Uplift targeting
  spends the budget on customers whose decision outreach can actually change.

Assumptions and limitations are in `reports/roi_assumptions.md`.
Chart: `outputs/roi/targeting_comparison.png`.
"""
    REPORT_PATH.write_text(text)


def main():
    df = load_strategy_scores()
    table = compare(df)
    print("Net benefit by strategy (real outcomes):")
    piv = table.pivot(index="outreach_share", columns="strategy", values="net_benefit")[[UPLIFT, CHURN, RANDOM]]
    print(piv.round(0).to_string())

    print(f"\nBootstrapping uplift vs churn risk ({N_BOOT} resamples)...")
    ci = bootstrap_uplift_vs_churn(df)
    for share, r in ci.items():
        print(f"  {share:>4} contacted: uplift - churn = ${r['mean_diff']:,.0f}  "
              f"[{r['ci_low']:,.0f}, {r['ci_high']:,.0f}]")

    ROI_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(ROI_DIR / "targeting_comparison.csv", index=False)
    (ROI_DIR / "targeting_comparison_ci.json").write_text(json.dumps(ci, indent=2))
    make_chart(table, ROI_DIR / "targeting_comparison.png")
    write_report(table, ci)
    print(f"\nSaved: {ROI_DIR}/targeting_comparison.csv, targeting_comparison_ci.json, "
          f"targeting_comparison.png, {REPORT_PATH}")


if __name__ == "__main__":
    main()