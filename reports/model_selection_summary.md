# Sprint 7 Model Selection

We compared the Logistic Regression and Random Forest T-Learners using 5-fold
cross-validation on all 10,000 customers. Each model was evaluated on the
observed uplift in its top and bottom quintiles, using the agreed selection rule
(quintile spread, with top-quintile uplift as the tiebreaker).

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Top quintile observed uplift | 25.7 pts | 24.2 pts |
| Bottom quintile observed uplift | 2.8 pts | 1.8 pts |
| Quintile spread | 22.9 pts | 22.4 pts |
| Extra retained per 1,000 (top 30% contacted) | 68.8 | 68.8 |

The difference in quintile spread was only 0.5 pts and the bootstrap 95% interval (-5.0 to 5.1 pts) includes zero, so the two models are statistically tied. We also reran the comparison on 5 different data splits, and
Logistic Regression won 3 while Random Forest won 2.

Since the models perform about the same, we went with Logistic Regression under our tie rule because it is simpler and easier to explain to stakeholders. The ranked customer list and visualizations for Sprint 7 use its uplift scores.
