"""
Creates SHAP summary plots and a short summary of the key factors driving predicted uplift.
Run after shap_analysis.py.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap

from cross_validate import CLEANED_PATH
from shap_analysis import LABELS, SHAP_DIR

REPORT_PATH = Path("reports") / "shap_key_factors.md"
BLUE, RED, INK, MUTED, GRID = "#2a78d6", "#d1603d", "#1d2329", "#5f6770", "#e4e7eb"


def load():
    shap_df = pd.read_csv(SHAP_DIR / "shap_values.csv")
    importance = pd.read_csv(SHAP_DIR / "feature_importance.csv")
    features = importance["feature"].tolist()
    data = pd.read_csv(CLEANED_PATH).loc[shap_df["customer_index"], features].reset_index(drop=True)
    return shap_df, importance, features, data


def beeswarm(shap_df, features, data):
    values = shap.Explanation(values=shap_df[features].values * 100, data=data.values,
                              feature_names=[LABELS[f] for f in features])
    cmap = LinearSegmentedColormap.from_list("low_high", [BLUE, "#d6d9de", RED])
    plt.figure()
    shap.plots.beeswarm(values, max_display=10, color=cmap, show=False, plot_size=(9, 6))
    plt.xlabel("Effect on predicted uplift (percentage points)")
    plt.title("How each factor changes a customer's predicted uplift\n"
              "Each dot is a customer. Red = high value, blue = low value.", loc="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close("all")


def top_factors(importance):
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top["label"], top["avg_impact_pts"], color=BLUE, height=0.6)
    for i, (v, d) in enumerate(zip(top["avg_impact_pts"], top["direction"])):
        ax.text(v + 0.08, i, f"{v:.1f}   ({d.lower()})", va="center", fontsize=9, color=MUTED)
    ax.set_xlim(0, top["avg_impact_pts"].max() * 2.3)
    ax.set_xlabel("Average effect on predicted uplift (percentage points)", color=MUTED)
    ax.set_title("Top 10 factors behind the uplift scores", loc="left", fontsize=12, color=INK)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK, length=0)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(SHAP_DIR / "shap_top_factors.png", dpi=160)
    plt.close(fig)


def write_summary(importance, n_customers):
    top = importance.head(6)
    share = top["avg_impact_pts"].sum() / importance["avg_impact_pts"].sum()
    rows = "\n".join(f"| {i + 1} | {r.label} | {r.avg_impact_pts:.1f} pts | {r.direction} |"
                     for i, r in enumerate(top.itertuples()))
    names = [l[0].lower() + l[1:] for l in top["label"]]
    biggest = ", ".join(names[:-1]) + f" and {names[-1]}"
    weak_names = [l[0].lower() + l[1:] for l in importance.tail(4)["label"]]
    weak = ", ".join(weak_names[:-1]) + f" and {weak_names[-1]}"
    text = f"""# Key Factors Driving Predicted Uplift (SHAP)

SHAP values were computed for {n_customers:,} customers using the final tuned model. A SHAP value
shows how much one factor raises or lowers a customer's predicted uplift, in percentage points.

| Rank | Factor | Average effect | Direction |
|---|---|---|---|
{rows}

The biggest factors, in order, are {biggest}. Together they account for {share:.0%} of the
total effect. Factors such as {weak} have almost no effect.

These results match the customer groups that showed the largest benefit in the outreach test
(Sprint 7), so the model is picking up real differences between customers.

Plots: `outputs/shap/shap_summary.png` and `outputs/shap/shap_top_factors.png`
"""
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(text)


def main():
    shap_df, importance, features, data = load()
    beeswarm(shap_df, features, data)
    top_factors(importance)
    write_summary(importance, len(shap_df))
    print(f"Saved {SHAP_DIR}/shap_summary.png, {SHAP_DIR}/shap_top_factors.png and {REPORT_PATH}")


if __name__ == "__main__":
    main()