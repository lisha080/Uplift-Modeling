import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import uplift_metrics as um
from train_models import OUTPUT_DIR

RANK_DIR = OUTPUT_DIR / "ranking"
OUT_PATH = OUTPUT_DIR / "charts" / "uplift_charts.html"
FINAL_PATH = OUTPUT_DIR / "final_model" / "oof_predictions.csv"
SHAP_PATH = OUTPUT_DIR / "shap" / "feature_importance.csv"

# Plain-language meaning of each SHAP factor, used when the model's direction matches
INSIGHTS = {
    ("IsActiveMember", "No"): ("Inactive members gain the most from outreach.",
                               "A message aimed at getting them active again is likely to work well."),
    ("previous_campaign_response", "Yes"): ("Customers who responded to a past campaign tend to respond again.",
                                            "They are worth contacting."),
    ("engagement_score", "Lower"): ("Customers with low engagement benefit more.",
                                    "They are drifting away, and contact can bring them back."),
    ("NumOfProducts", "Lower"): ("Customers with fewer products benefit more.",
                                 "Customers who already hold several products gain little from outreach."),
    ("Age", "Higher"): ("Older customers respond better to outreach than younger ones.", ""),
    ("Geography_Germany", "Yes"): ("Customers in Germany respond better than customers in France or Spain.", ""),
}

INK, MUTED, GRID = "#1d2329", "#5f6770", "#e4e7eb"
SEGMENT_COLORS = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4"]
MODEL_COLORS = {"logistic_t_learner": "#2a78d6", "random_forest_t_learner": "#e0703a"}
MODEL_NAMES = {"logistic_t_learner": "Method A: Logistic Regression", "random_forest_t_learner": "Method B: Random Forest"}
PER_100 = "Extra customers kept per 100 contacted"

# Benefit groups for the first chart (extra customers kept per 100 contacted)
BANDS = [
    ("Outreach may backfire", -np.inf, 0, "#d1603d"),
    ("Little benefit (0 to 5)", 0, 5, "#9ec5f4"),
    ("Some benefit (5 to 15)", 5, 15, "#4f8fdc"),
    ("Large benefit (15 or more)", 15, np.inf, "#1c5cab"),
]


def style(fig, x_title, y_title, height=400, legend=False):
    fig.update_layout(
        template="none", height=height, margin=dict(l=70, r=30, t=20, b=70),
        font=dict(family="Helvetica Neue, Arial, sans-serif", size=14, color=MUTED),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=legend,
        hoverlabel=dict(bgcolor="white", font_size=14, font_color=INK),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(color=INK)),
    )
    fig.update_xaxes(title=x_title, showgrid=False, linecolor=GRID, ticks="", tickfont=dict(color=INK))
    fig.update_yaxes(title=y_title, gridcolor=GRID, zerolinecolor="#b9bfc6", ticks="")
    return fig


def benefit_groups(ranked):
    per_100 = ranked["uplift_score"] * 100
    rows = []
    for label, lo, hi, color in BANDS:
        n = int(((per_100 >= lo) & (per_100 < hi)).sum())
        rows.append({"group": label, "n_customers": n, "color": color})
    return pd.DataFrame(rows)


def benefit_chart(groups, total):
    fig = go.Figure(go.Bar(
        x=groups["group"], y=groups["n_customers"], marker_color=groups["color"], width=0.6,
        text=[f"{n:,}" for n in groups["n_customers"]], textposition="outside",
        textfont=dict(size=15, color=INK),
        customdata=(groups["n_customers"] / total * 100).values,
        hovertemplate="<b>%{x}</b><br>%{y:,} customers (%{customdata:.0f}% of all customers)<extra></extra>",
    ))
    fig.update_yaxes(range=[0, groups["n_customers"].max() * 1.18])
    return style(fig, "Expected benefit from contacting the customer", "Number of customers")


def segment_chart(summary):
    labels = [f"{s}<br>{n:,} customers" for s, n in zip(summary["priority_segment"], summary["n_customers"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=summary["observed_uplift"] * 100, name="What actually happened",
        marker_color=SEGMENT_COLORS, width=0.55,
        text=[f"{v * 100:.0f}" for v in summary["observed_uplift"]], textposition="inside",
        textfont=dict(color=["white", "white", "white", INK, INK], size=15), insidetextanchor="start",
        textangle=0, constraintext="none",
        customdata=summary["recommended_action"].values,
        hovertemplate="<b>%{customdata}</b><br>About %{y:.0f} extra customers kept per 100 contacted<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=summary["mean_predicted_uplift"] * 100, name="What the model expected",
        mode="markers", marker=dict(symbol="line-ew", size=34, line=dict(width=3, color=INK)),
        hovertemplate="The model expected about %{y:.0f} extra per 100<extra></extra>",
    ))
    fig.update_yaxes(rangemode="tozero")
    return style(fig, "", PER_100, height=420, legend=True)


def customer_groups(ranked):
    traits = {
        "Membership": ranked["IsActiveMember"].map({1: "Active member", 0: "Inactive member"}),
        "Country": ranked["Geography"],
        "Age": pd.cut(ranked["Age"], [0, 35, 45, 200], right=False, labels=["Under 35", "35 to 44", "45 and over"]),
        "Products held": pd.cut(ranked["NumOfProducts"], [0, 1, 2, 99], labels=["1 product", "2 products", "3 or more products"]),
        "Engagement": pd.cut(ranked["engagement_score"], [0, 30, 60, 101], right=False,
                             labels=["Low engagement", "Medium engagement", "High engagement"]),
    }
    rows = []
    for trait, values in traits.items():
        for group, g in ranked.groupby(values, observed=True):
            rows.append({"trait": trait, "group": str(group), "n_customers": len(g),
                         "extra_per_100": um.observed_uplift(g["treatment"], g["outcome"]) * 100})
    return pd.DataFrame(rows)


def groups_chart(groups):
    traits = list(dict.fromkeys(groups["trait"]))
    sizes = [int((groups["trait"] == t).sum()) for t in traits]
    fig = make_subplots(rows=len(traits), cols=1, shared_xaxes=True, subplot_titles=traits,
                        row_heights=sizes, vertical_spacing=0.06)
    for i, trait in enumerate(traits, start=1):
        g = groups[groups["trait"] == trait]
        fig.add_trace(go.Bar(
            y=g["group"], x=g["extra_per_100"], orientation="h", width=0.7,
            marker_color=["#1c5cab" if v >= 15 else "#5598e7" if v >= 5 else "#b8cde6" for v in g["extra_per_100"]],
            text=[f"{v:.0f}" if v >= 1 else "less than 1" for v in g["extra_per_100"]], textposition="outside",
            textfont=dict(color=INK, size=13), customdata=g["n_customers"], cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>About %{x:.0f} extra customers kept per 100 contacted"
                          "<br>%{customdata:,} customers in this group<extra></extra>",
        ), row=i, col=1)
        fig.update_yaxes(autorange="reversed", row=i, col=1)
    fig = style(fig, "", "", height=150 + 46 * len(groups))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, range=[0, groups["extra_per_100"].max() * 1.15])
    fig.update_xaxes(title=PER_100, row=len(traits), col=1)
    fig.update_annotations(font=dict(size=14, color=MUTED), x=0, xanchor="left")
    fig.update_layout(margin=dict(l=150, r=40, t=30, b=60))
    return fig


def priority1_profile(ranked):
    p1 = ranked[ranked["priority_segment"] == "Priority 1"]
    rows = [
        ("Inactive members", lambda d: f"{(d['IsActiveMember'] == 0).mean():.0%}"),
        ("Based in Germany", lambda d: f"{(d['Geography'] == 'Germany').mean():.0%}"),
        ("Hold only 1 product", lambda d: f"{(d['NumOfProducts'] == 1).mean():.0%}"),
        ("Responded to a past campaign", lambda d: f"{d['previous_campaign_response'].mean():.0%}"),
        ("Average age", lambda d: f"{d['Age'].mean():.0f}"),
        ("Average engagement score (out of 100)", lambda d: f"{d['engagement_score'].mean():.0f}"),
    ]
    body = "".join(f"<tr><td>{label}</td><td>{f(p1)}</td><td>{f(ranked)}</td></tr>" for label, f in rows)
    return f"<table><tr><th></th><th>Priority 1</th><th>All customers</th></tr>{body}</table>"


def capacity_table(summary):
    kept = (summary["observed_uplift"] * summary["n_customers"]).cumsum()
    total = kept.iloc[-1]
    body = ""
    for i, row in summary.iterrows():
        who = "Priority 1 only" if i == 0 else f"Priority 1 to {i + 1}" if i < len(summary) - 1 else "Everyone"
        body += (f"<tr><td>{who}</td><td>{int(row['cumulative_customers']):,}</td>"
                 f"<td>about {kept[i]:,.0f}</td><td>{kept[i] / total:.0%}</td></tr>")
    return ("<table><tr><th>Who we contact</th><th>Customers contacted</th><th>Extra customers who stay</th>"
            f"<th>Share of the total benefit</th></tr>{body}</table>"), kept, total


def drivers_section(importance):
    items = []
    for r in importance.head(6).itertuples():
        key = (r.feature, r.direction.split(" ")[0])
        if key in INSIGHTS:
            headline, detail = INSIGHTS[key]
            items.append(f"<li><b>{headline}</b> {detail}</li>")
    weak = [(l[0].lower() + l[1:]).replace("has a", "having a") for l in importance.tail(4)["label"]]
    items.append(f"<li><b>Some things make almost no difference.</b> {', '.join(weak[:-1]).capitalize()} and "
                 f"{weak[-1]} barely change how much a customer benefits, so they are not useful for "
                 f"choosing who to contact.</li>")
    return (f"<ul class='insights'>{''.join(items)}</ul>"
            "<p class='note'>The ranked customer list (ranked_customers.csv) includes the two main reasons for "
            "each customer's score, for example \"Inactive member, Engagement score 12/100\". These explain "
            "why the model scored the customer the way it did. They are not proven causes of why a customer "
            "stays or leaves.</p>")


def model_chart(quintiles, selected):
    fig = go.Figure()
    labels = {4: "Top 20%", 3: "Next 20%", 2: "Middle 20%", 1: "Lower 20%", 0: "Bottom 20%"}
    for model, g in quintiles.groupby("model_name"):
        g = g.sort_values("quintile", ascending=False)
        name = MODEL_NAMES[model] + (" (used)" if model == selected else "")
        fig.add_trace(go.Bar(
            x=[labels[q] for q in g["quintile"]], y=g["observed_uplift"] * 100, name=name,
            marker_color=MODEL_COLORS[model],
            hovertemplate=f"<b>{name}</b><br>%{{x}} of customers<br>About %{{y:.0f}} extra kept per 100 contacted<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    return style(fig, "Customers ranked by each method, best first", PER_100, height=420, legend=True)


def check_consistency(ranked, summary, quintiles, groups, cust_groups, selected):
    counts = ranked["priority_segment"].value_counts()
    for row in summary.itertuples():
        assert counts[row.priority_segment] == row.n_customers, f"{row.priority_segment} count does not match"
    pred = pd.read_csv(FINAL_PATH if FINAL_PATH.exists() else OUTPUT_DIR / selected / "oof_predictions.csv")
    q = um.quintile_table(pred["uplift_score"], pred["treatment"], pred["outcome"]).sort_values("quintile", ascending=False)
    assert (q["observed_uplift"].round(6).values == summary["observed_uplift"].round(6).values).all(), \
        "Segment chart does not match the model's scores"
    assert groups["n_customers"].sum() == len(ranked), "Benefit groups do not add up to all customers"
    for trait, g in cust_groups.groupby("trait"):
        assert g["n_customers"].sum() == len(ranked), f"{trait} groups do not add up to all customers"


def build_page(sections):
    blocks = []
    first_chart = True
    for title, text, content in sections:
        if isinstance(content, str):
            body = content
        else:
            body = content.to_html(full_html=False, include_plotlyjs=first_chart, config={"displayModeBar": False})
            first_chart = False
        blocks.append(f"<section><h2>{title}</h2><p>{text}</p>{body}</section>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Who Should We Contact?</title>
<style>
  body {{ margin: 0; background: #f5f6f7; color: {INK}; font-family: "Helvetica Neue", Arial, sans-serif; }}
  main {{ max-width: 920px; margin: 0 auto; padding: 32px 20px 60px; }}
  h1 {{ font-size: 28px; margin: 0 0 8px; }}
  .sub {{ color: {MUTED}; margin: 0 0 28px; line-height: 1.6; font-size: 16px; max-width: 70ch; }}
  section {{ background: white; border: 1px solid {GRID}; border-radius: 6px; padding: 22px 22px 8px; margin-bottom: 22px; }}
  h2 {{ font-size: 19px; margin: 0 0 6px; }}
  section p {{ color: {MUTED}; margin: 0 0 10px; line-height: 1.6; font-size: 15px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0 18px; font-size: 15px; }}
  th, td {{ text-align: left; padding: 9px 12px; border-bottom: 1px solid {GRID}; }}
  th {{ color: {MUTED}; font-weight: 600; font-size: 13px; }}
  td:not(:first-child), th:not(:first-child) {{ text-align: right; }}
  .insights {{ margin: 6px 0 14px; padding-left: 20px; line-height: 1.6; font-size: 15px; color: {INK}; }}
  .insights li {{ margin-bottom: 8px; }}
  section p.note {{ font-size: 14px; margin-bottom: 16px; }}
</style></head>
<body><main>
<h1>Who should the retention team contact?</h1>
<p class="sub">Reaching out to a customer helps some people stay, but not everyone. Some customers would stay
anyway, and a few are more likely to leave if contacted. These charts show which customers benefit most,
so outreach goes where it makes a difference. Hover over any bar for details.</p>
{''.join(blocks)}
</main></body></html>"""


def main():
    selected = json.loads((OUTPUT_DIR / "selection_summary.json").read_text())["selected_model"]
    ranked = pd.read_csv(RANK_DIR / "ranked_customers.csv")
    summary = pd.read_csv(RANK_DIR / "segment_summary.csv")
    quintiles = pd.read_csv(OUTPUT_DIR / "quintile_table.csv")
    groups = benefit_groups(ranked)
    cust_groups = customer_groups(ranked)
    check_consistency(ranked, summary, quintiles, groups, cust_groups, selected)
    capacity_html, kept, total = capacity_table(summary)

    def group_value(trait, group):
        row = cust_groups[(cust_groups["trait"] == trait) & (cust_groups["group"] == group)].iloc[0]
        return row["extra_per_100"]

    p1, p5 = summary.iloc[0], summary.iloc[-1]
    n_backfire = int(groups.iloc[0]["n_customers"])
    sections = [
        ("Most customers benefit from outreach, but not equally",
         f"Each bar shows how many of our {len(ranked):,} customers fall into each group, based on the model's "
         f"estimate. {n_backfire:,} customers may be more likely to leave if we contact them.",
         benefit_chart(groups, len(ranked))),
        ("Start with Priority 1 customers",
         f"We split customers into 5 equal groups of {int(p1['n_customers']):,}. In our test, contacting 100 "
         f"Priority 1 customers kept about {p1['observed_uplift'] * 100:.0f} extra customers, compared with about "
         f"{p5['observed_uplift'] * 100:.0f} for Priority 5. If we can only contact some customers, start with "
         f"Priority 1 and work down.",
         segment_chart(summary)),
        ("Which kinds of customers benefit most?",
         f"Inactive members, customers in Germany, older customers and customers with low engagement respond best "
         f"to outreach. Contacting 100 inactive members kept about {group_value('Membership', 'Inactive member'):.0f} "
         f"extra customers, compared with about {group_value('Membership', 'Active member'):.0f} for active members. "
         f"Customers with 3 or more products barely respond at all, so outreach to them is mostly wasted. "
         f"This is a small group, so treat that number as a rough guide.",
         groups_chart(cust_groups)),
        ("What do Priority 1 customers look like?",
         "Compared with the customer base as a whole, Priority 1 customers are more often inactive, older, based in "
         "Germany and holding a single product. This can help the retention team tailor the message.",
         priority1_profile(ranked)),
    ]
    if SHAP_PATH.exists():
        sections.append((
            "Why does the model pick these customers?",
            "We used SHAP, a method that shows which customer traits drive each score. "
            "These are the traits that matter most, from strongest to weakest.",
            drivers_section(pd.read_csv(SHAP_PATH)),
        ))
    sections += [
        ("How many customers should we contact?",
         f"Each step down the priority list adds fewer extra customers. Contacting Priority 1 and 2 "
         f"({int(summary['cumulative_customers'].iloc[1]):,} customers) gets about {kept[1] / total:.0%} of the "
         f"benefit of contacting everyone, with less than half the effort. Numbers are based on the results of our "
         f"outreach test.",
         capacity_html),
        ("We tested two methods, and both picked the right customers",
         "Each method ranked customers from most to least likely to benefit. For both, the customers ranked at the "
         "top really did benefit the most. The two performed about the same, so we use Method A because its "
         "results are easier to explain.",
         model_chart(quintiles, selected)),
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_page(sections))
    print("Check passed: chart values match the ranked customer list and model evaluation")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()