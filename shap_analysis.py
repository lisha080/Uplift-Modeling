import contextlib
import io
import json
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler

from cross_validate import CLEANED_PATH, get_scale_cols
from train_models import OUTPUT_DIR, train_t_learner

SHAP_DIR = OUTPUT_DIR / "shap"
N_EXPLAIN = None     # None = explain every customer (about 2 minutes); set e.g. 2000 for a quick run
N_BACKGROUND = 100   # reference customers SHAP compares against
SEED = 42

LABELS = {
    "CreditScore": "Credit score",
    "Gender": "Gender",
    "Age": "Age",
    "Tenure": "Years as a customer",
    "Balance": "Account balance",
    "NumOfProducts": "Number of products",
    "HasCrCard": "Has a credit card",
    "IsActiveMember": "Active member",
    "EstimatedSalary": "Estimated salary",
    "engagement_score": "Engagement score",
    "login_frequency": "Login frequency",
    "purchase_count": "Purchase count",
    "support_tickets": "Support tickets",
    "previous_campaign_response": "Responded to a past campaign",
    "Geography_Germany": "Based in Germany",
    "Geography_Spain": "Based in Spain",
}


def fit_final_model(X, y, config, scale_cols, feature_cols):
    scaler = StandardScaler().fit(X[scale_cols])
    X_scaled = X.copy()
    X_scaled[scale_cols] = scaler.transform(X[scale_cols])
    with contextlib.redirect_stdout(io.StringIO()):
        m_t, m_c, _ = train_t_learner(X_scaled, y, feature_cols, config["model_type"], config["params"])

    # Takes unscaled customer data, so SHAP results are in real units (Age = 45, not 0.6)
    pos = [feature_cols.index(c) for c in scale_cols]

    def uplift(data):
        data = np.array(data, dtype=float)
        data[:, pos] = (data[:, pos] - scaler.mean_) / scaler.scale_
        return m_t.predict_proba(data)[:, 1] - m_c.predict_proba(data)[:, 1]

    return uplift


def direction(feature, values, shap_values):
    corr = np.corrcoef(values, shap_values)[0, 1] if values.std() > 0 else 0
    if abs(corr) < 0.1:
        return "No clear direction"
    if set(np.unique(values)) <= {0, 1}:
        return f"{'Yes' if corr > 0 else 'No'} = more benefit from outreach"
    return f"{'Higher' if corr > 0 else 'Lower'} = more benefit from outreach"


def describe(feature, value, row):
    """Plain-language description of one customer trait."""
    v = int(value) if float(value).is_integer() else value
    text = {
        "IsActiveMember": "Active member" if v else "Inactive member",
        "previous_campaign_response": "Responded to a past campaign" if v else "Never responded to a campaign",
        "HasCrCard": "Has a credit card" if v else "No credit card",
        "Gender": "Male" if v else "Female",
        "Geography_Germany": f"Based in {row['country']}",
        "Geography_Spain": f"Based in {row['country']}",
        "engagement_score": f"Engagement score {v}/100",
        "NumOfProducts": f"Holds {v} product{'s' if v != 1 else ''}",
        "Age": f"Age {v}",
        "CreditScore": f"Credit score {v}",
        "Tenure": f"{v} year{'s' if v != 1 else ''} as a customer",
        "purchase_count": f"{v} purchases",
        "login_frequency": f"{v} logins",
        "support_tickets": f"{v} support tickets",
        "Balance": "Zero balance" if v == 0 else f"Balance {value:,.0f}",
        "EstimatedSalary": f"Salary {value:,.0f}",
    }
    return text.get(feature, f"{feature} {v}")


def customer_reasons(shap_df, data, feature_cols, n=2):
    """Plain-language reasons for each customer's score. Above-average customers get the traits
    that push their score up; below-average customers get the traits that push it down."""
    data = data.reset_index(drop=True).copy()
    data["country"] = np.select([data["Geography_Germany"] == 1, data["Geography_Spain"] == 1],
                                ["Germany", "Spain"], default="France")
    values = shap_df[feature_cols].values
    reasons = []
    for i in range(len(values)):
        sign = 1 if values[i].sum() >= 0 else -1
        picked = []
        for j in np.argsort(-values[i] * sign):
            if values[i, j] * sign <= 0:
                break
            text = describe(feature_cols[j], data.loc[i, feature_cols[j]], data.loc[i])
            if text not in picked:
                picked.append(text)
            if len(picked) == n:
                break
        reasons.append(", ".join(picked))
    return pd.DataFrame({"customer_index": shap_df["customer_index"], "main_reasons": reasons})


def main():
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    config = json.loads((OUTPUT_DIR / "final_model" / "final_model_config.json").read_text())
    df = pd.read_csv(CLEANED_PATH)
    y = df["outcome"]
    X = df.drop(columns=["outcome"])
    feature_cols = [c for c in X.columns if c != "treatment"]
    scale_cols = get_scale_cols(X)

    print(f"Fitting the final model ({config['model_type']}, {config['params']}) on all {len(X):,} customers...")
    uplift = fit_final_model(X, y, config, scale_cols, feature_cols)

    if N_EXPLAIN:
        sample = np.sort(np.random.default_rng(SEED).choice(len(X), N_EXPLAIN, replace=False))
    else:
        sample = np.arange(len(X))
    background = X[feature_cols].sample(N_BACKGROUND, random_state=SEED).values
    explainer = shap.PermutationExplainer(uplift, shap.maskers.Independent(background), seed=SEED)

    print(f"Computing SHAP values for {len(sample):,} customers...")
    result = explainer(X.loc[sample, feature_cols].values, silent=True)

    # SHAP values should add up to each customer's uplift score
    scores = uplift(X.loc[sample, feature_cols].values)
    error = np.abs(result.values.sum(axis=1) + result.base_values - scores).max()
    assert error < 1e-6, f"SHAP values do not add up to the uplift scores (max error {error})"

    shap_df = pd.DataFrame(result.values, columns=feature_cols)
    shap_df.insert(0, "base_value", result.base_values)
    shap_df.insert(0, "uplift_score", scores)
    shap_df.insert(0, "customer_index", sample)

    rows = []
    for c in feature_cols:
        rows.append({
            "feature": c,
            "label": LABELS.get(c, c),
            "avg_impact_pts": float(np.abs(shap_df[c]).mean() * 100),
            "direction": direction(c, X.loc[sample, c].values, shap_df[c].values),
        })
    importance = pd.DataFrame(rows).sort_values("avg_impact_pts", ascending=False).reset_index(drop=True)

    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(SHAP_DIR / "shap_values.csv", index=False)
    importance.to_csv(SHAP_DIR / "feature_importance.csv", index=False)
    reasons = customer_reasons(shap_df, X.loc[sample], feature_cols)
    reasons.to_csv(SHAP_DIR / "customer_reasons.csv", index=False)

    print(f"Check passed: SHAP values add up to each customer's uplift score\n")
    print("Top factors (average impact on uplift, in percentage points):")
    for i, r in importance.head(10).iterrows():
        print(f"  {i + 1:>2}. {r['label']:<30} {r['avg_impact_pts']:5.2f}   {r['direction']}")
    print(f"\nExample reasons: {reasons['main_reasons'].iloc[0]} | {reasons['main_reasons'].iloc[1]}")
    print(f"Saved {SHAP_DIR}/shap_values.csv, feature_importance.csv and customer_reasons.csv")


if __name__ == "__main__":
    main()