# Final Model Configuration (Sprint 8)

- **Model:** Logistic Regression T-Learner (one model for contacted customers, one for non-contacted)
- **Setting:** C = 0.1, L1 penalty (tuned)
- **Features:** all customer columns except the treatment flag, numeric columns standardized
- **Uplift score:** chance of staying if contacted minus chance of staying if not contacted

## Tuning

We tried 12 settings on the 8,000 training customers and checked the best
one on the 2,000 holdout customers. Tuned setting was not worse than the default on the holdout customers.

| Metric (all 10,000 customers) | Sprint 7 | Final |
|---|---|---|
| Quintile spread | 22.9 pts | 23.1 pts |
| Top quintile observed uplift | 25.7 pts | 26.1 pts |
| Extra retained per 1,000 (top 20% contacted) | 51.5 | 52.2 |
| AUUC per 1,000 | 22.4 | 23.0 |

## Stability across data splits

We reran the final model and the Sprint 7 model on 5 different data splits.

- The final model had a higher quintile spread than the Sprint 7 model in 5 of 5 splits
  (average 23.6 pts vs 22.5 pts).
- Its quintile spread ranged from 23.0 to 25.5 pts across splits.
- Customer rankings were very similar across splits (rank correlation 0.97).
  88% of Priority 1 customers stayed in Priority 1, and 99%
  of customers never moved more than one segment.

The improvement from tuning is small but consistent. The rankings the retention team would use do not depend much on how the data was split.
