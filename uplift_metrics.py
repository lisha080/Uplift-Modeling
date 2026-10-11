"""
Shared uplift metrics.

Observed uplift = retention rate (treated) - retention rate (control)

- outcome: 1 = retained, 0 = churned (higher is better)
- treatment: 1 = received outreach, 0 = control
- quintile 0 = LOWEST predicted uplift, quintile 4 = HIGHEST predicted uplift
"""

import numpy as np
import pandas as pd

N_QUINTILES = 5


def assign_quintiles(score, n_groups=N_QUINTILES):
    score = np.asarray(score)
    ranks = np.argsort(np.argsort(score, kind="stable"), kind="stable")
    return (ranks * n_groups // len(score)).astype(int)


def observed_uplift(treatment, outcome):
    t = np.asarray(treatment)
    o = np.asarray(outcome)
    if (t == 1).sum() == 0 or (t == 0).sum() == 0:
        return np.nan
    return float(o[t == 1].mean() - o[t == 0].mean())


def quintile_table(score, treatment, outcome):
    s, t, o = np.asarray(score), np.asarray(treatment), np.asarray(outcome)
    q = assign_quintiles(s)
    rows = []
    for k in range(N_QUINTILES):
        m = q == k
        tm, om = t[m], o[m]
        rows.append({
            "quintile": k,
            "n_customers": int(m.sum()),
            "n_treated": int((tm == 1).sum()),
            "n_control": int((tm == 0).sum()),
            "mean_predicted_uplift": float(s[m].mean()),
            "treated_retention": float(om[tm == 1].mean()) if (tm == 1).any() else np.nan,
            "control_retention": float(om[tm == 0].mean()) if (tm == 0).any() else np.nan,
            "observed_uplift": observed_uplift(tm, om),
        })
    return pd.DataFrame(rows)


def uplift_curve(score, treatment, outcome, n_points=100):
    s, t, o = np.asarray(score), np.asarray(treatment), np.asarray(outcome)
    order = np.argsort(-s, kind="stable")
    t, o = t[order], o[order]
    n = len(s)

    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    r_t = np.cumsum(o * t)
    r_c = np.cumsum(o * (1 - t))

    fracs = np.arange(1, n_points + 1) / n_points
    k = np.ceil(fracs * n).astype(int) - 1  # index of last customer included

    valid = (n_t[k] > 0) & (n_c[k] > 0)
    rate_t = np.where(valid, r_t[k] / np.maximum(n_t[k], 1), 0.0)
    rate_c = np.where(valid, r_c[k] / np.maximum(n_c[k], 1), 0.0)
    incremental = (rate_t - rate_c) * (k + 1)
    return fracs, incremental


def auuc_per_1000(score, treatment, outcome):
    fracs, inc = uplift_curve(score, treatment, outcome)
    random_line = fracs * inc[-1]
    n = len(score)
    return float(np.mean(inc - random_line) / n * 1000)


def incremental_per_1000_at(score, treatment, outcome, budget):
    fracs, inc = uplift_curve(score, treatment, outcome)
    idx = int(round(budget * len(fracs))) - 1
    return float(inc[idx] / len(score) * 1000)


def predicted_quintile_spread(score):
    s = np.asarray(score)
    q = assign_quintiles(s)
    return float(s[q == N_QUINTILES - 1].mean() - s[q == 0].mean())


def core_metrics(score, treatment, outcome):
    s, t, o = np.asarray(score), np.asarray(treatment), np.asarray(outcome)
    q = assign_quintiles(s)
    top = observed_uplift(t[q == N_QUINTILES - 1], o[q == N_QUINTILES - 1])
    bottom = observed_uplift(t[q == 0], o[q == 0])
    return {
        "observed_top_quintile_uplift": top,
        "observed_bottom_quintile_uplift": bottom,
        "observed_quintile_spread": top - bottom,
        "auuc_per_1000": auuc_per_1000(s, t, o),
        "incremental_per_1000_at_20pct": incremental_per_1000_at(s, t, o, 0.20),
        "incremental_per_1000_at_30pct": incremental_per_1000_at(s, t, o, 0.30),
    }


def paired_bootstrap(scores_by_model, treatment, outcome, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    t_all, o_all = np.asarray(treatment), np.asarray(outcome)
    n = len(t_all)
    keys = ["observed_top_quintile_uplift", "observed_quintile_spread", "auuc_per_1000"]
    draws = {m: {k: [] for k in keys} for m in scores_by_model}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        t, o = t_all[idx], o_all[idx]
        for m, s in scores_by_model.items():
            res = core_metrics(np.asarray(s)[idx], t, o)
            for k in keys:
                draws[m][k].append(res[k])

    return {m: {k: np.array(v) for k, v in d.items()} for m, d in draws.items()}


def ci95(draws):
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)