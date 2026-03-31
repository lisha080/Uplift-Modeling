import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

RAW_PATH = Path("data/raw/churn_modeling.csv")
OUTPUT_PATH = Path("data/processed/synthetic_data.csv")

# Load and drop non-predictive columns
df = pd.read_csv(RAW_PATH)
df = df.drop(columns=["RowNumber", "Surname", "Exited"])

# Behavioral features 
df["engagement_score"] = np.random.randint(10, 100, size=len(df))
df["login_frequency"] = np.random.randint(1, 15, size=len(df))
df["purchase_count"] = np.random.randint(0, 20, size=len(df))

# Support tickets: higher for disengaged/inactive customers
support_base = (
    1
    + 0.04 * (100 - df["engagement_score"])
    + 0.5 * (1 - df["IsActiveMember"])
    + np.random.normal(0, 1.0, size=len(df))
)
df["support_tickets"] = np.clip(np.round(support_base), 0, 10).astype(int)

# Previous campaign response: correlated with engagement and tenure
campaign_score = (
    0.03 * df["engagement_score"]
    + 0.15 * df["Tenure"]
    + 0.8 * df["IsActiveMember"]
    - 0.2 * df["support_tickets"]
    + np.random.normal(0, 1.0, size=len(df))
)
campaign_prob = 1 / (1 + np.exp(-0.08 * (campaign_score - campaign_score.mean())))
df["previous_campaign_response"] = (np.random.rand(len(df)) < campaign_prob).astype(int)

# Treatment assignment (RCT: pure random 50/50)
df["treatment"] = np.random.binomial(1, 0.5, size=len(df))

# Probability of churning WITHOUT outreach
logit = (
    -0.5
    + 0.04 * (df["Age"] - 38)
    + 0.6 * (1 - df["IsActiveMember"])
    + 0.3 * (df["NumOfProducts"] == 1).astype(int)
    + 1.2 * (df["NumOfProducts"] >= 3).astype(int)
    + 0.5 * (df["Geography"] == "Germany").astype(int)
    - 0.1 * (df["Geography"] == "Spain").astype(int)
    + 0.3 * (df["Gender"] == "Female").astype(int)
    + 0.2 * (df["Balance"] == 0).astype(int)
    - 0.003 * (df["CreditScore"] - 650)
    - 0.05 * df["Tenure"]
    # Behavioral features influence baseline churn
    - 0.008 * df["engagement_score"]
    - 0.03 * df["login_frequency"]
    + 0.05 * df["support_tickets"]
    - 0.2 * df["previous_campaign_response"]
)
df["p_churn_base"] = 1 / (1 + np.exp(-logit))

# Positive ITE = outreach reduces churn. Negative = outreach backfires.
ite = np.full(len(df), 0.02)

# Persuadable signals
ite += 0.20 * ((df["Age"] > 45) & (df["IsActiveMember"] == 0)).astype(int)
ite += 0.08 * ((df["Age"] > 40) & (df["Age"] <= 45)).astype(int)
ite += 0.12 * ((df["NumOfProducts"] == 1) & (df["IsActiveMember"] == 0)).astype(int)
ite += 0.06 * (df["Geography"] == "Germany").astype(int)
ite += 0.05 * ((df["Balance"] > 50000) & (df["Balance"] < 150000)).astype(int)
ite += 0.07 * (df["CreditScore"] < 550).astype(int)
ite += 0.08 * (df["engagement_score"] < 30).astype(int)
ite += 0.05 * (df["support_tickets"] >= 4).astype(int)
ite -= 0.04 * (df["previous_campaign_response"] == 0).astype(int)

# Sleeping dog signals
ite -= 0.15 * (df["NumOfProducts"] >= 3).astype(int)
ite -= 0.06 * ((df["Tenure"] >= 8) & (df["IsActiveMember"] == 1)).astype(int)

# Sure thing dampening
sure_thing = (df["Age"] < 30) & (df["IsActiveMember"] == 1) & (df["CreditScore"] > 700)
ite = np.where(sure_thing, np.maximum(ite * 0.2, 0.01), ite)

df["ite"] = np.clip(ite, -0.20, 0.35)

# Generate observed outcome 
p_churn = np.where(
    df["treatment"] == 0,
    df["p_churn_base"],
    np.clip(df["p_churn_base"] - df["ite"], 0, 1)
)
df["outcome"] = (np.random.rand(len(df)) > p_churn).astype(int)  # 1=retained, 0=churned

df["segment"] = np.select(
    [
        df["ite"] < -0.02,
        (df["ite"] < 0.05) & (df["p_churn_base"] < 0.25),
        (df["ite"] < 0.05) & (df["p_churn_base"] >= 0.25),
    ],
    ["Sleeping Dog", "Sure Thing", "Lost Cause"],
    default="Persuadable"
)

# Save dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

control = df[df["treatment"] == 0]
treated = df[df["treatment"] == 1]
ate = treated["outcome"].mean() - control["outcome"].mean()

print("Synthetic dataset created successfully.")
print(f"Saved to: {OUTPUT_PATH}")
print(f"\nRows: {len(df)}, Columns: {len(df.columns)}")
print(f"Treatment rate: {df['treatment'].mean():.3f}")
print(f"Retention rate: {df['outcome'].mean():.3f}")
print(f"Observed ATE:   {ate:.4f} ({ate*100:.1f}pp)")
print(f"\nSegment breakdown:")
print(df["segment"].value_counts().to_string())
