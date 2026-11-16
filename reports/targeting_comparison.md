# Uplift-Based vs Traditional Targeting (Sprint 9)

All three strategies contact the same number of customers at the same cost
($20 per contact, $300 per customer kept). Only *who* is contacted changes:

- **Uplift targeting**: customers whose chance of staying rises most with outreach (final tuned model).
- **Churn-risk targeting**: customers most likely to leave, the traditional approach.
- **Random targeting**: a random sample of customers.

Extra customers kept are measured from real outcomes in the experiment, not from model predictions,
so each strategy is judged fairly.

## Results

| Customers contacted | Uplift: extra kept / net benefit | Churn risk: extra kept / net benefit | Random: extra kept / net benefit |
|---|---|---|---|
| 10% | 266 / $59,672 | 264 / $59,252 | 124 / $17,149 |
| 20% | 522 / $116,497 | 372 / $71,582 | 248 / $34,299 |
| 30% | 692 / $147,602 | 541 / $102,430 | 371 / $51,448 |
| 50% | 944 / $183,064 | 840 / $152,004 | 619 / $85,747 |
| 70% | 1,111 / $193,341 | 1,064 / $179,220 | 867 / $120,046 |
| 100% | 1,238 / $171,494 | 1,238 / $171,494 | 1,238 / $171,494 |

## Is the difference real?

Bootstrap with 500 resamples of the 10,000 customers:

| Customers contacted | Uplift minus churn risk (net benefit) | 95% interval | Clear winner? |
|---|---|---|---|
| 20% | $44,283 | $18,937 to $69,259 | Yes, uplift |
| 30% | $47,188 | $20,695 to $78,340 | Yes, uplift |
| 50% | $29,967 | -$677 to $58,674 | Not clear |

## Takeaways

- With 30% of customers contacted, uplift targeting keeps 692 extra customers, compared with
  541 for churn-risk targeting and 371 for random targeting. That is
  $45,173 more net benefit than churn-risk targeting for the same budget.
- Best outreach size for each strategy (real outcomes): uplift 75% ($197,655),
  churn risk 80% ($184,233), random 100% ($171,494).
- Customers at highest risk of leaving are not always the ones outreach helps most. Uplift targeting
  spends the budget on customers whose decision outreach can actually change.

Assumptions and limitations are in `reports/roi_assumptions.md`.
Chart: `outputs/roi/targeting_comparison.png`.
