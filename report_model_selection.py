import json
from pathlib import Path

import pandas as pd

from train_models import OUTPUT_DIR

REPORT_PATH = Path("reports") / "model_selection_summary.md"
NAMES = {"logistic_t_learner": "Logistic Regression", "random_forest_t_learner": "Random Forest"}


def main():
    selection = json.loads((OUTPUT_DIR / "selection_summary.json").read_text())
    stability = json.loads((OUTPUT_DIR / "stability" / "stability_summary.json").read_text())
    comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv").set_index("model_name")

    lr = comparison.loc["logistic_t_learner"]
    rf = comparison.loc["random_forest_t_learner"]
    margin = next(m for m in selection["margins_vs_others"] if m["metric"] == "observed_quintile_spread")
    tied = not margin["decisive"]
    selected = NAMES[selection["selected_model"]]
    wins = stability["wins"]

    if tied:
        result = (f"The difference in quintile spread was only {abs(lr['observed_quintile_spread'] - rf['observed_quintile_spread']) * 100:.1f} pts and the "
                  f"bootstrap 95% interval ({margin['ci_low'] * 100:.1f} to {margin['ci_high'] * 100:.1f} pts) "
                  f"includes zero, so the two models are statistically tied.")
        decision = (f"Since the models perform about the same, we went with {selected} under our tie rule "
                    f"because it is simpler and easier to explain to stakeholders.")
    else:
        result = (f"The difference in quintile spread was {abs(lr['observed_quintile_spread'] - rf['observed_quintile_spread']) * 100:.1f} pts, and the "
                  f"bootstrap 95% interval does not include zero.")
        decision = f"{selected} performed better, so we selected it."

    text = f"""# Sprint 7 Model Selection

We compared the Logistic Regression and Random Forest T-Learners using 5-fold
cross-validation on all {int(lr['n_customers']):,} customers. Each model was evaluated on the
observed uplift in its top and bottom quintiles, using the agreed selection rule
(quintile spread, with top-quintile uplift as the tiebreaker).

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Top quintile observed uplift | {lr['observed_top_quintile_uplift'] * 100:.1f} pts | {rf['observed_top_quintile_uplift'] * 100:.1f} pts |
| Bottom quintile observed uplift | {lr['observed_bottom_quintile_uplift'] * 100:.1f} pts | {rf['observed_bottom_quintile_uplift'] * 100:.1f} pts |
| Quintile spread | {lr['observed_quintile_spread'] * 100:.1f} pts | {rf['observed_quintile_spread'] * 100:.1f} pts |
| Extra retained per 1,000 (top 30% contacted) | {lr['incremental_per_1000_at_30pct']:.1f} | {rf['incremental_per_1000_at_30pct']:.1f} |

{result} We also reran the comparison on {len(stability['seeds'])} different data splits, and
Logistic Regression won {wins.get('logistic_t_learner', 0)} while Random Forest won {wins.get('random_forest_t_learner', 0)}.

{decision} The ranked customer list and visualizations for Sprint 7 use its uplift scores.
"""

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(text)
    print(f"Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()