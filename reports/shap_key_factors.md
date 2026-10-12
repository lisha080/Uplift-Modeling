# Key Factors Driving Predicted Uplift (SHAP)

SHAP values were computed for 10,000 customers using the final tuned model. A SHAP value
shows how much one factor raises or lowers a customer's predicted uplift, in percentage points.

| Rank | Factor | Average effect | Direction |
|---|---|---|---|
| 1 | Active member | 6.1 pts | No = more benefit from outreach |
| 2 | Responded to a past campaign | 3.3 pts | Yes = more benefit from outreach |
| 3 | Number of products | 2.7 pts | Lower = more benefit from outreach |
| 4 | Engagement score | 2.7 pts | Lower = more benefit from outreach |
| 5 | Age | 2.7 pts | Higher = more benefit from outreach |
| 6 | Based in Germany | 2.3 pts | Yes = more benefit from outreach |

The biggest factors, in order, are active member, responded to a past campaign, number of products, engagement score, age and based in Germany. Together they account for 78% of the
total effect. Factors such as estimated salary, account balance, support tickets and has a credit card have almost no effect.

These results match the customer groups that showed the largest benefit in the outreach test
(Sprint 7), so the model is picking up real differences between customers.

Plots: `outputs/shap/shap_summary.png` and `outputs/shap/shap_top_factors.png`
