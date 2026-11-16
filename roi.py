"""
Shared ROI math used by roi_simulation.py, targeting_comparison.py and dashboard.py.
"""

import numpy as np
import pandas as pd

CONTACT_COST = 20.0        # $ per customer contacted
RETENTION_VALUE = 300.0   # $ value of keeping one customer
OUTREACH_SHARES = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def n_contacted(n_customers, outreach_share):
    return int(np.ceil(outreach_share * n_customers))


def top_k_order(score):
    return np.argsort(-np.asarray(score, dtype=float), kind="stable")


def predicted_incremental(uplift_score, k):
    s = np.asarray(uplift_score, dtype=float)
    return float(s[top_k_order(s)[:k]].sum())


def observed_incremental(score, treatment, outcome, k):
    order = top_k_order(score)[:k]
    t = np.asarray(treatment)[order]
    o = np.asarray(outcome)[order]
    if k == 0 or (t == 1).sum() == 0 or (t == 0).sum() == 0:
        return np.nan
    return float((o[t == 1].mean() - o[t == 0].mean()) * k)


def money(incremental_retained, k, contact_cost=CONTACT_COST, retention_value=RETENTION_VALUE):
    cost = k * contact_cost
    benefit = incremental_retained * retention_value
    net = benefit - cost
    return {
        "cost": float(cost),
        "benefit": float(benefit),
        "net_benefit": float(net),
        "roi": float(net / cost) if cost > 0 else np.nan,
    }


def simulate(uplift_score, outreach_share, contact_cost=CONTACT_COST, retention_value=RETENTION_VALUE,
             treatment=None, outcome=None, rank_score=None):
    uplift_score = np.asarray(uplift_score, dtype=float)
    rank_score = uplift_score if rank_score is None else np.asarray(rank_score, dtype=float)
    k = n_contacted(len(uplift_score), outreach_share)
    contacted = top_k_order(rank_score)[:k]

    pred_inc = float(uplift_score[contacted].sum())
    row = {"outreach_share": float(outreach_share), "n_contacted": k,
           "contact_cost": float(contact_cost), "retention_value": float(retention_value),
           "predicted_incremental": pred_inc}
    row.update({f"predicted_{key}": v for key, v in money(pred_inc, k, contact_cost, retention_value).items()})

    if treatment is not None and outcome is not None:
        obs_inc = observed_incremental(rank_score, treatment, outcome, k)
        row["observed_incremental"] = obs_inc
        row.update({f"observed_{key}": v for key, v in money(obs_inc, k, contact_cost, retention_value).items()})
    return row


def roi_table(uplift_score, contact_cost=CONTACT_COST, retention_value=RETENTION_VALUE,
              treatment=None, outcome=None, rank_score=None, shares=OUTREACH_SHARES):
    return pd.DataFrame([
        simulate(uplift_score, s, contact_cost, retention_value, treatment, outcome, rank_score)
        for s in shares
    ])


def random_targeting(treatment, outcome, outreach_share, contact_cost=CONTACT_COST,
                     retention_value=RETENTION_VALUE):
    t, o = np.asarray(treatment), np.asarray(outcome)
    ate = o[t == 1].mean() - o[t == 0].mean()
    k = n_contacted(len(t), outreach_share)
    inc = float(ate * k)
    row = {"outreach_share": float(outreach_share), "n_contacted": k, "observed_incremental": inc}
    row.update({f"observed_{key}": v for key, v in money(inc, k, contact_cost, retention_value).items()})
    return row


def break_even_uplift(contact_cost=CONTACT_COST, retention_value=RETENTION_VALUE):
    return contact_cost / retention_value