"""
Checks roi.py against hand-worked examples.
"""

import numpy as np

import roi


def check(name, condition):
    assert condition, f"FAIL: {name}"
    print(f"PASS  {name}")


def main():
    r = roi.simulate(np.full(10, 0.10), 1.0, contact_cost=10, retention_value=500)
    check("contacted all 10", r["n_contacted"] == 10)
    check("1 extra customer kept", np.isclose(r["predicted_incremental"], 1.0))
    check("cost $100", np.isclose(r["predicted_cost"], 100))
    check("benefit $500", np.isclose(r["predicted_benefit"], 500))
    check("net benefit $400", np.isclose(r["predicted_net_benefit"], 400))
    check("ROI 4.0", np.isclose(r["predicted_roi"], 4.0))

    r = roi.simulate(np.array([0.1, 0.4, 0.2, 0.3]), 0.5, contact_cost=10, retention_value=100)
    check("contacts the highest scores first", np.isclose(r["predicted_incremental"], 0.7))
    check("net benefit 0.7 x $100 - 2 x $10 = $50", np.isclose(r["predicted_net_benefit"], 50))

    score = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.0])
    treatment = np.array([1, 0, 1, 0, 1, 0])
    outcome = np.array([1, 1, 1, 0, 0, 1])
    check("observed extra kept = 2", np.isclose(roi.observed_incremental(score, treatment, outcome, 4), 2.0))

    r = roi.simulate(np.array([0.1, 0.4, 0.2, 0.3]), 0.5, rank_score=np.array([4, 1, 3, 2]))
    check("rank_score picks customers 0 and 2 -> 0.3 extra kept", np.isclose(r["predicted_incremental"], 0.3))

    r = roi.random_targeting(np.array([1, 1, 1, 0, 0, 0]), np.array([1, 1, 0, 1, 0, 0]), 0.5,
                             contact_cost=10, retention_value=100)
    check("random targeting extra kept = 1", np.isclose(r["observed_incremental"], 1.0))

    check("break-even uplift = cost / value", np.isclose(roi.break_even_uplift(20, 300), 20 / 300))
    check("outreach size rounds up", roi.n_contacted(10_000, 0.05) == 500 and roi.n_contacted(7, 0.5) == 4)
    print("\nAll ROI checks passed.")


if __name__ == "__main__":
    main()