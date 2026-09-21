"""
Uplift-based targeting vs TRADITIONAL (non-uplift) targeting.

Question answered: if we can only contact the top X% of customers, does ranking
them by predicted UPLIFT retain more customers than ranking them by predicted
CHURN RISK (the conventional approach)?

How the comparison is kept fair
-------------------------------
- Same 10,000 customers, each scored out-of-fold (never by a model that saw them).
- Same folds (cross_validate.make_folds) and same per-fold scaling.
- Same features (everything except `treatment`).
- Same yardstick: incremental retained customers, estimated from the randomized
  treated-vs-control gap among the customers each strategy would contact
  (uplift_metrics.uplift_curve).
- Same budgets, and the same resampled customers for every bootstrap draw.

Definition of "traditional targeting" (edit BASELINE_* below if the team defines it differently)
------------------------------------------------------------------------------------------------
Contact the customers with the HIGHEST predicted probability of churning:
  churn_risk = 1 - P(retained | features)
from a single Logistic Regression that ignores the treatment flag. This is what a
team without an experiment would build. A second variant, trained on control
customers only (a "pure" no-outreach risk model), is reported as a sensitivity check.

Outputs (outputs/):
  baseline_comparison.csv        incremental retained by budget, per strategy
  baseline_selection_profile.csv who each strategy selects at the headline budget
  baseline_summary.json          headline numbers + confidence intervals
  baseline_comparison.png        the chart for the sprint review
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import uplift_metrics as um
from cross_validate import CLEANED_PATH, get_scale_cols, make_folds
from train_models import OUTPUT_DIR

UPLIFT_MODELS = {
    "Uplift: Logistic Regression": "logistic_t_learner",
    "Uplift: Random Forest": "random_forest_t_learner",
}
BASELINE_ALL = "Traditional: churn risk"
BASELINE_CONTROL = "Traditional: churn risk (control-only model)"
RANDOM = "Random targeting"

HEADLINE_BUDGET = 0.30  # share of customers we can afford to contact (an assumption; revisit in Sprint 8 ROI work)
BUDGETS = np.round(np.arange(0.05, 1.0001, 0.05), 2)
N_BOOT = 1000
SYNTHETIC_PATH = Path("data/processed/synthetic_data.csv")
PROFILE_COLS = ["Age", "NumOfProducts", "engagement_score", "support_tickets", "previous_campaign_response"]


# ---------------------------------------------------------------- baseline scores
def oof_churn_risk(X, y, folds, scale_cols, feature_cols, control_only=False):
    """Out-of-fold churn risk from a treatment-blind Logistic Regression."""
    risk = np.full(len(X), np.nan)
    for train_pos, test_pos in folds:
        X_tr, X_te = X.iloc[train_pos].copy(), X.iloc[test_pos].copy()
        y_tr = y.iloc[train_pos]
        scaler = StandardScaler().fit(X_tr[scale_cols])  # train fold only
        X_tr[scale_cols] = scaler.transform(X_tr[scale_cols])
        X_te[scale_cols] = scaler.transform(X_te[scale_cols])

        fit_rows = (X_tr["treatment"] == 0).values if control_only else np.ones(len(X_tr), dtype=bool)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_tr.loc[fit_rows, feature_cols], y_tr[fit_rows])
        risk[test_pos] = 1 - model.predict_proba(X_te[feature_cols])[:, 1]
    assert not np.isnan(risk).any()
    return risk


# ---------------------------------------------------------------- comparison tables
def budget_table(scores, t, o):
    """Incremental retained customers at each budget, for every strategy."""
    n = len(t)
    rows = []
    for name, s in scores.items():
        fracs, inc = um.uplift_curve(s, t, o)
        for b in BUDGETS:
            i = int(round(b * 100)) - 1
            contacted = fracs[i] * n
            rows.append({
                "strategy": name,
                "budget_share_contacted": float(b),
                "incremental_per_1000_customers": float(inc[i] / n * 1000),
                "incremental_per_1000_contacted": float(inc[i] / contacted * 1000),
            })
    return pd.DataFrame(rows)


def curve_metrics(score, t, o):
    """Everything we bootstrap, from one pass over the uplift curve."""
    fracs, inc = um.uplift_curve(score, t, o)
    n = len(t)
    return {
        "inc_at_20": inc[19] / n * 1000,
        "inc_at_30": inc[29] / n * 1000,
        "auuc": float(np.mean(inc - fracs * inc[-1]) / n * 1000),
    }


def bootstrap_vs_baseline(scores, baseline_name, t, o):
    """Paired bootstrap: (strategy - baseline) for every uplift model."""
    rng = np.random.default_rng(42)
    n = len(t)
    draws = {name: {"inc_at_20": [], "inc_at_30": [], "auuc": []} for name in scores}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        res = {name: curve_metrics(s[idx], t[idx], o[idx]) for name, s in scores.items()}
        for name in scores:
            for k in draws[name]:
                draws[name][k].append(res[name][k] - res[baseline_name][k])
    out = {}
    for name in scores:
        if name == baseline_name:
            continue
        out[name] = {}
        for k, arr in draws[name].items():
            arr = np.array(arr)
            lo, hi = um.ci95(arr)
            out[name][k] = {"diff": float(np.nanmean(arr)), "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0)}
    return out


def top_k_mask(score, budget):
    k = int(np.ceil(budget * len(score)))
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-np.asarray(score), kind="stable")[:k]] = True
    return mask


def selection_profile(scores, ref, t, o):
    """Who does each strategy actually contact at the headline budget?"""
    syn = None
    if SYNTHETIC_PATH.exists():
        syn_all = pd.read_csv(SYNTHETIC_PATH)
        ids = ref["customer_index"].values
        if ids.max() < len(syn_all) and (syn_all.loc[ids, "treatment"].values == t).all():
            syn = syn_all.loc[ids].reset_index(drop=True)

    def describe(label, mask):
        row = {"group": label, "n_contacted": int(mask.sum()),
               "observed_uplift": um.observed_uplift(t[mask], o[mask]),
               "share_inactive_members": float((ref.loc[mask, "IsActiveMember"] == 0).mean()),
               "share_germany": float(ref.loc[mask, "Geography_Germany"].mean()),
               "share_3plus_products": float((ref.loc[mask, "NumOfProducts"] >= 3).mean())}
        for c in PROFILE_COLS:
            row[f"mean_{c}"] = float(ref.loc[mask, c].mean())
        if syn is not None:  # synthetic-data diagnostic only
            row["DIAGNOSTIC_mean_true_ite"] = float(syn.loc[mask, "ite"].mean())
            row["DIAGNOSTIC_sleeping_dogs_contacted"] = int((syn.loc[mask, "segment"] == "Sleeping Dog").sum())
        return row

    masks = {name: top_k_mask(s, HEADLINE_BUDGET) for name, s in scores.items() if name != RANDOM}
    rows = [describe("All customers", np.ones(len(t), dtype=bool))]
    rows += [describe(name, m) for name, m in masks.items()]

    names = list(masks)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a}  vs  {b}"] = float((masks[a] & masks[b]).sum() / masks[a].sum())
    return pd.DataFrame(rows), overlaps


# ---------------------------------------------------------------- chart
def make_chart(table, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed: skipping chart)")
        return False
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    styles = {
        "Uplift: Logistic Regression": dict(color="#1f77b4", lw=2.6),
        "Uplift: Random Forest": dict(color="#2ca02c", lw=2.0, ls="--"),
        BASELINE_ALL: dict(color="#d62728", lw=2.6),
        BASELINE_CONTROL: dict(color="#ff9896", lw=1.6, ls=":"),
        RANDOM: dict(color="#7f7f7f", lw=1.6, ls="-."),
    }
    for name, st in styles.items():
        d = table[table["strategy"] == name]
        x = np.concatenate([[0], d["budget_share_contacted"].values * 100])
        y = np.concatenate([[0], d["incremental_per_1000_customers"].values])
        ax.plot(x, y, label=name, **st)
    ax.axvline(HEADLINE_BUDGET * 100, color="black", lw=0.8, alpha=0.35)
    ax.set_xlabel("Share of customers contacted (%)")
    ax.set_ylabel("Extra customers retained by outreach\n(per 1,000 customers in the base)")
    ax.set_title("Who to contact? Ranking by uplift vs ranking by churn risk")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- main
def main():
    df = pd.read_csv(CLEANED_PATH)
    y = df["outcome"]
    X = df.drop(columns=["outcome"])
    feature_cols = [c for c in X.columns if c != "treatment"]
    scale_cols = get_scale_cols(X)
    folds = make_folds(X, y)
    t, o = X["treatment"].values, y.values

    # Uplift-model scores (out-of-fold), checked to be on exactly the same customers
    scores, ref = {}, None
    for label, model_name in UPLIFT_MODELS.items():
        path = OUTPUT_DIR / model_name / "oof_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}. Run cross_validate.py first.")
        p = pd.read_csv(path)
        same = (len(p) == len(X) and (p["customer_index"].values == X.index.values).all()
                and (p["treatment"].values == t).all() and (p["outcome"].values == o).all())
        if not same:
            raise ValueError(f"{path} is not aligned with cleaned_data.csv. Re-run cross_validate.py.")
        scores[label] = p["uplift_score"].values
        ref = p if ref is None else ref

    print("Fitting traditional churn-risk baselines (5-fold, same folds as the uplift models)...")
    scores[BASELINE_ALL] = oof_churn_risk(X, y, folds, scale_cols, feature_cols, control_only=False)
    scores[BASELINE_CONTROL] = oof_churn_risk(X, y, folds, scale_cols, feature_cols, control_only=True)
    scores[RANDOM] = np.random.default_rng(7).random(len(t))  # placeholder ordering; random line is analytic below

    # Random targeting = contact x% of customers, capture x% of the average effect
    fracs, inc_ref = um.uplift_curve(scores[BASELINE_ALL], t, o)
    ate_total = inc_ref[-1]

    table = budget_table({k: v for k, v in scores.items() if k != RANDOM}, t, o)
    n = len(t)
    rand_rows = pd.DataFrame({
        "strategy": RANDOM, "budget_share_contacted": BUDGETS,
        "incremental_per_1000_customers": BUDGETS * ate_total / n * 1000,
        "incremental_per_1000_contacted": ate_total / n * 1000,
    })
    table = pd.concat([table, rand_rows], ignore_index=True)

    print(f"\n{'=' * 84}\nIncremental retained customers per 1,000 customers in the base (n = {n})\n{'=' * 84}")
    pivot = table.pivot(index="budget_share_contacted", columns="strategy", values="incremental_per_1000_customers")
    order = [*UPLIFT_MODELS, BASELINE_ALL, BASELINE_CONTROL, RANDOM]
    show = pivot.loc[[b for b in [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0] if b in pivot.index], order]
    show.index = [f"{int(b * 100)}% contacted" for b in show.index]
    print(show.round(1).to_string())

    print("\nRunning paired bootstrap (same resampled customers for every strategy)...")
    strategies = {k: v for k, v in scores.items() if k != RANDOM}
    boot = {}
    labels = {"inc_at_20": "extra retained @20% budget", "inc_at_30": "extra retained @30% budget",
              "auuc": "average advantage over all budgets (AUUC)"}
    for baseline in [BASELINE_ALL, BASELINE_CONTROL]:
        res_all = bootstrap_vs_baseline(strategies, baseline, t, o)
        boot[baseline] = {m: res_all[m] for m in UPLIFT_MODELS}
        print(f"\n{'=' * 84}\nUplift minus [{baseline}]  (per 1,000 customers; 95% CI)\n{'=' * 84}")
        for name, res in boot[baseline].items():
            print(f"  {name}")
            for k, r in res.items():
                verdict = "significant" if r["significant"] else "not significant"
                print(f"    {labels[k]:<44} {r['diff']:+6.1f}  [{r['ci_low']:+6.1f}, {r['ci_high']:+6.1f}]  {verdict}")

    profile, overlaps = selection_profile(scores, ref, t, o)
    print(f"\n{'=' * 84}\nWho gets contacted at a {int(HEADLINE_BUDGET * 100)}% budget?\n{'=' * 84}")
    print(profile.round(3).T.to_string(header=False))
    print("\n  Overlap in customers contacted:")
    for k, v in overlaps.items():
        print(f"    {v:5.0%}  {k}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_DIR / "baseline_comparison.csv", index=False)
    profile.to_csv(OUTPUT_DIR / "baseline_selection_profile.csv", index=False)
    summary = {
        "traditional_baseline_definition": "Rank customers by predicted churn risk (1 - P(retained)) from a "
                                           "treatment-blind Logistic Regression; contact the highest-risk first.",
        "headline_budget_share": HEADLINE_BUDGET,
        "units": "extra retained customers per 1,000 customers in the base",
        "uplift_minus_baseline_95ci": boot,
        "overlap_of_contacted_customers_at_headline_budget": overlaps,
    }
    with open(OUTPUT_DIR / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    chart_ok = make_chart(table, OUTPUT_DIR / "baseline_comparison.png")
    print("\nSaved: baseline_comparison.csv, baseline_selection_profile.csv, baseline_summary.json"
          + (", baseline_comparison.png" if chart_ok else ""))


if __name__ == "__main__":
    main()