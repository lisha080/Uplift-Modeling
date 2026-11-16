# Stability Across Random Seeds (Sprint 10)

The final model (Logistic Regression T-Learner, tuned) was retrained with 5 different random seeds.
Each seed changes how customers are split into the 5 cross-validation folds, so every customer is scored
by a differently trained model. We then checked whether the business results change.

| Seed | Quintile spread | Priority 1 overlap | Best outreach | Net benefit at 30% | Gain vs churn risk at 30% |
|---|---|---|---|---|---|
| 42 | 23.1 pts | 100% | Top 70% | $147,602 | $45,173 |
| 7 | 25.5 pts | 87% | Top 70% | $148,563 | $46,133 |
| 123 | 23.1 pts | 88% | Top 70% | $156,832 | $54,402 |
| 2024 | 23.0 pts | 87% | Top 70% | $166,185 | $63,755 |
| 31337 | 23.1 pts | 86% | Top 70% | $146,923 | $44,494 |

Seed 42 is the split used for the saved results and the dashboard. Net benefit and gain use real outcomes,
$20 per contact and $300 per customer kept.

## What stays the same

- **Ranking:** average rank correlation between seeds is 0.973. On average
  88% of Priority 1 customers are the same between any two seeds, and
  99% of customers stay within one priority group across all seeds.
- **Recommended outreach size:** top 70% across seeds.
- **Net benefit at 30% outreach:** $146,923 to $166,185 (average $153,221).
- **Gain over churn-risk targeting at 30%:** $44,494 to $63,755; uplift targeting
  wins with every seed.

## Conclusion

The recommendations do not depend on a lucky random split.
The dashboard uses seed 42; any seed would lead to the same targeting decisions.
