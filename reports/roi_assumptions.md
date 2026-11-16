# ROI Simulation: Assumptions and Results (Sprint 9)

## What we calculate

For a campaign that contacts the top N customers ranked by uplift score:

- **Cost** = N x contact cost
- **Extra customers kept** = customers who stay *because of* the outreach
- **Benefit** = extra customers kept x value of keeping one customer
- **Net benefit** = benefit - cost
- **ROI** = net benefit / cost (100% means every $1 spent returns $2)

## Assumptions

| Assumption | Value used | Notes |
|---|---|---|
| Cost to contact one customer | $20 | Staff time, offer and channel cost. Illustrative; replace with the sponsor's real figure. |
| Value of keeping one customer | $300 | Profit from a customer who would otherwise have left. Illustrative. |
| Customers in the simulation | 10,000 | Every customer is scored by a model that never saw them (5-fold cross-validation). |
| Uplift score meaning | Probability | An uplift of 0.20 means outreach raises the chance of staying by 20 points. |
| Who gets contacted | Highest uplift first | One contact per customer, no capacity limits. |
| Break-even uplift | 0.067 | Contact cost / retention value. Customers below this cost more to contact than they return. |

Limitations:
- The data is synthetic (Kaggle churn data plus generated treatment and behavior columns), so dollar
  amounts show how the method works, not what a real campaign would earn.
- Costs and values are the same for every customer. In practice, high-balance customers are worth more.
- The simulation assumes outreach works the same way it did in the experiment.

## Checking the math (#88)

Every number is calculated two ways:

- **Predicted**: adds up the model's uplift scores for the customers contacted.
- **Observed**: uses what actually happened. Among the customers contacted, we compare those who
  received outreach in the experiment with those who did not.

If the model is well calibrated the two should be close. Largest gap in extra customers kept across
all outreach sizes: 89 customers
(18% of the observed value). The formulas are also checked
against hand-worked cases in `test_roi.py`.

## Results

| Customers contacted | Extra customers kept (predicted / observed) | Cost | Net benefit (predicted / observed) | ROI (predicted / observed) |
|---|---|---|---|---|
| 10% (1,000) | 313 / 266 | $20,000 | $73,961 / $59,672 | 370% / 298% |
| 20% (2,000) | 546 / 522 | $40,000 | $123,814 / $116,497 | 310% / 291% |
| 30% (3,000) | 737 / 692 | $60,000 | $161,049 / $147,602 | 268% / 246% |
| 50% (5,000) | 1,023 / 944 | $100,000 | $207,008 / $183,064 | 207% / 183% |
| 70% (7,000) | 1,198 / 1,111 | $140,000 | $219,546 / $193,341 | 157% / 138% |
| 100% (10,000) | 1,230 / 1,238 | $200,000 | $168,934 / $171,494 | 84% / 86% |

- **Best outreach size:** contact the top **70%** (7,000 customers).
  Predicted net benefit **$219,546** (observed $193,341),
  ROI 157%.
- Contacting everyone gives a net benefit of $168,934,
  so targeting saves money on customers who would not benefit.

Chart: `outputs/roi/roi_curve.png`. Full table: `outputs/roi/roi_by_outreach_size.csv`.
