"""
Retention outreach dashboard.
Run with: streamlit run dashboard.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import roi
from dashboard_data import SHAP_IMAGES, explain_customer, filter_customers, load_all, missing_files
from targeting_comparison import CHURN, RANDOM, UPLIFT, compare

st.set_page_config(page_title="Retention Outreach", layout="wide")

SEGMENT_ORDER = ["Priority 1", "Priority 2", "Priority 3", "Priority 4", "Priority 5"]
NAVY, STEEL, SILVER, RUST = "#1F3A5F", "#5B7DB1", "#B8C2CC", "#B4553A"
INK, MUTED, GRID = "#1D2329", "#5F6B76", "#EEF1F4"
STRATEGY_STYLE = {UPLIFT: (NAVY, "solid", 3), CHURN: (RUST, "dash", 2), RANDOM: (SILVER, "dot", 2)}
CHART_CONFIG = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "showTips": False}

CSS = """
<style>
#MainMenu, footer, [data-testid="stDecoration"] {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 1240px;}
h1, h2, h3 {letter-spacing: -0.01em; color: #1D2329;}
h3 {font-size: 1.05rem !important; font-weight: 600 !important; margin-top: 1.2rem;}
[data-testid="stMetric"] {border: 1px solid #E3E7EC; border-radius: 6px; padding: 14px 16px; background: #FFFFFF;}
[data-testid="stMetricLabel"] p {font-size: 0.78rem; color: #5F6B76; text-transform: uppercase; letter-spacing: 0.04em;}
[data-testid="stMetricValue"] {font-size: 1.55rem; font-weight: 600; color: #1D2329;}
.stTabs [data-baseweb="tab-list"] {gap: 28px; border-bottom: 1px solid #E3E7EC;}
.stTabs [data-baseweb="tab"] {font-weight: 500; padding: 8px 0;}
.header {border-bottom: 1px solid #E3E7EC; padding-bottom: 14px; margin-bottom: 18px;}
.header .title {font-size: 1.6rem; font-weight: 650; color: #1D2329;}
.header .sub {font-size: 0.85rem; color: #5F6B76; margin-top: 2px;}
.detail-head {display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 6px;}
.detail-title {font-size: 1.2rem; font-weight: 650; color: #1D2329;}
.pill {display: inline-block; font-size: 0.75rem; padding: 3px 10px; margin-left: 6px; border-radius: 999px;
       background: #F4F6F8; color: #1D2329; border: 1px solid #E3E7EC;}
.pill.go {background: #1F3A5F; color: #FFFFFF; border-color: #1F3A5F;}
.pill.stop {background: #F6E9E5; color: #B4553A; border-color: #EBCFC6;}
.label {font-size: 0.75rem; color: #5F6B76; text-transform: uppercase; letter-spacing: 0.04em; margin: 8px 0 4px;}
.big {font-size: 1.5rem; font-weight: 650; color: #1F3A5F;}
.reasons {list-style: none; padding: 0; margin: 0 0 6px;}
.reasons li {display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #EEF1F4;
             font-size: 0.9rem; color: #1D2329;}
.reasons .pts {color: #1F3A5F; font-weight: 600;}
.reasons.neg .pts {color: #B4553A;}
.traits {display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 14px 0 8px;}
.traits div {background: #F7F8FA; border-radius: 6px; padding: 8px 10px;}
.traits span {display: block; font-size: 0.72rem; color: #5F6B76;}
.traits b {font-size: 0.95rem; font-weight: 600; color: #1D2329;}
.note {font-size: 0.8rem; color: #5F6B76; line-height: 1.45; margin: 4px 0 8px;}
</style>
"""

DRIVER_GROUPS = [
    ("Inactive members", lambda d: d["IsActiveMember"] == 0, "Re-activation offer"),
    ("Responded to a past campaign", lambda d: d["previous_campaign_response"] == 1, "Include in every campaign wave"),
    ("Hold a single product", lambda d: d["NumOfProducts"] == 1, "Retention offer; skip customers with 3+ products"),
    ("Below-median engagement", lambda d: d["engagement_score"] < d["engagement_score"].median(), "Re-engagement message"),
    ("Aged 45 and over", lambda d: d["Age"] >= 45, "Prioritise for direct contact"),
    ("Based in Germany", lambda d: d["Geography"] == "Germany", "Weight regional budget toward Germany"),
]

PROFILE = [
    ("Inactive members", lambda d: (d["IsActiveMember"] == 0).mean(), "{:.0%}"),
    ("Responded to a past campaign", lambda d: d["previous_campaign_response"].mean(), "{:.0%}"),
    ("Average number of products", lambda d: d["NumOfProducts"].mean(), "{:.2f}"),
    ("Average engagement score", lambda d: d["engagement_score"].mean(), "{:.0f}"),
    ("Average age", lambda d: d["Age"].mean(), "{:.0f}"),
    ("Based in Germany", lambda d: (d["Geography"] == "Germany").mean(), "{:.0%}"),
]


@st.cache_data
def get_data():
    return load_all()


@st.cache_data
def comparison_table(contact_cost, retention_value):
    return compare(get_data().strategy_scores, contact_cost, retention_value)


def dollars(x):
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def observed_per_100(d):
    t, o = d["treatment"], d["outcome"]
    return (o[t == 1].mean() - o[t == 0].mean()) * 100


def chart(fig, x_title="", y_title="", height=340, margin=None):
    fig.update_layout(
        template="none", height=height, margin=margin or dict(l=56, r=16, t=36, b=48), dragmode=False,
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=12, color=MUTED),
        plot_bgcolor="white", paper_bgcolor="white", bargap=0.35,
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=12, color=INK)),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#E3E7EC"))
    fig.update_xaxes(title=x_title, showgrid=False, fixedrange=True, linecolor="#D5DBE1", ticks="")
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=True, zerolinecolor="#D5DBE1", fixedrange=True)
    st.plotly_chart(fig, config=CHART_CONFIG)


GLOSSARY = """
- **Uplift score**: how much outreach raises a customer's chance of staying. +20 pts means 20 more customers
  kept for every 100 like them who are contacted.
- **Priority group**: customers split into five equal groups by uplift score. Priority 1 benefits most.
- **Observed vs predicted**: predicted comes from the model; observed is measured from real campaign results
  (customers who were contacted vs those who were not).
- **Net benefit**: value of the extra customers kept minus the cost of contacting everyone selected.
- **ROI**: net benefit divided by cost. 100% means every $1 spent comes back as $2.
"""


def header(data):
    cfg = data.final_config
    st.markdown(
        f"<div class='header'><div class='title'>Retention Outreach</div><div class='sub'>"
        f"Uplift model (Logistic Regression T-Learner, C = {cfg['params']['C']}) &nbsp;·&nbsp; "
        f"{len(data.ranked):,} customers scored out-of-sample</div></div>", unsafe_allow_html=True)

    best = data.roi_table.loc[data.roi_table["predicted_net_benefit"].idxmax()]
    comp = comparison_table(roi.CONTACT_COST, roi.RETENTION_VALUE).set_index(["strategy", "outreach_share"])
    lead = comp.loc[(UPLIFT, 0.3), "net_benefit"] - comp.loc[(CHURN, 0.3), "net_benefit"]
    p1 = data.segments.set_index("priority_segment").loc["Priority 1"]

    c = st.columns(4)
    c[0].metric("Customers scored", f"{len(data.ranked):,}")
    c[1].metric("Priority 1 uplift", f"+{p1.observed_uplift * 100:.1f} pts",
                help="Extra customers kept per 100 contacted in the top 20%, from real outcomes.")
    c[2].metric("Recommended outreach", f"Top {best.outreach_share:.0%}",
                help=f"Highest expected net benefit at ${roi.CONTACT_COST:,.0f} per contact and "
                     f"${roi.RETENTION_VALUE:,.0f} per customer kept.")
    c[3].metric("Gain vs churn-risk targeting", dollars(lead),
                help="Extra net benefit at 30% outreach, same budget, from real outcomes.")
    with st.expander("How to read this dashboard"):
        st.markdown(GLOSSARY)


def priority_tab(data):
    seg = data.segments.set_index("priority_segment").loc[SEGMENT_ORDER].reset_index()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Extra customers kept per 100 contacted")
        fig = go.Figure()
        fig.add_bar(x=seg.priority_segment, y=seg.observed_uplift * 100, name="Observed", marker_color=NAVY,
                    hovertemplate="%{x}: %{y:.1f}<extra>Observed</extra>")
        fig.add_bar(x=seg.priority_segment, y=seg.mean_predicted_uplift * 100, name="Predicted", marker_color=SILVER,
                    hovertemplate="%{x}: %{y:.1f}<extra>Predicted</extra>")
        fig.update_layout(barmode="group")
        chart(fig)
    with right:
        st.subheader("Priority groups")
        table = pd.DataFrame({
            "Group": seg.priority_segment,
            "Customers": seg.n_customers.map("{:,}".format),
            "Observed uplift": (seg.observed_uplift * 100).map("{:+.1f} pts".format),
            "Action": seg.recommended_action,
        })
        st.dataframe(table, hide_index=True)
        st.markdown("<div class='note'>Each group is 20% of customers, ranked by expected benefit from outreach. "
                    "Observed uplift is measured from real campaign outcomes.</div>", unsafe_allow_html=True)

    st.subheader("Target list")
    ranked = data.ranked
    f = st.columns(4)
    groups = f[0].multiselect("Priority group", SEGMENT_ORDER, default=["Priority 1"])
    countries = f[1].multiselect("Country", sorted(ranked["Geography"].unique()))
    activity = f[2].selectbox("Membership", ["All", "Active", "Inactive"])
    response = f[3].selectbox("Past campaign", ["All", "Responded", "Did not respond"])
    g = st.columns([2, 2, 2, 2])
    products = g[0].multiselect("Number of products", sorted(ranked["NumOfProducts"].unique()))
    age_min, age_max = int(ranked["Age"].min()), int(ranked["Age"].max())
    ages = g[1].slider("Age", age_min, age_max, (age_min, age_max))
    search = g[3].text_input("Find customer ID", placeholder="e.g. 15743617").strip()

    if search:
        view = filter_customers(ranked[ranked["CustomerId"].astype(str).str.contains(search, regex=False)])
        st.markdown("<div class='note'>Showing search results across all customers; filters are ignored.</div>",
                    unsafe_allow_html=True)
    else:
        view = filter_customers(ranked, groups, countries, activity, response, products, ages)
    if view.empty:
        st.info("No customers match.")
        return

    extra = view["uplift_score"].sum()
    net = extra * roi.RETENTION_VALUE - len(view) * roi.CONTACT_COST
    m = st.columns(4)
    m[0].metric("Customers selected", f"{len(view):,}")
    m[1].metric("Average uplift", f"+{view['uplift_score'].mean() * 100:.1f} pts")
    m[2].metric("Extra kept if all contacted", f"{extra:,.0f}")
    m[3].metric("Net benefit if all contacted", dollars(net),
                help=f"${roi.CONTACT_COST:,.0f} per contact, ${roi.RETENTION_VALUE:,.0f} per customer kept.")

    show = pd.DataFrame({
        "Rank in selection": view.rank_in_selection,
        "Overall rank": view.priority_rank,
        "Customer ID": view.CustomerId.astype(str),
        "Group": view.priority_segment,
        "Uplift (pts)": (view.uplift_score * 100).round(1),
        "Stay if contacted": view.retention_with_outreach.map("{:.0%}".format),
        "Stay if not contacted": view.retention_without_outreach.map("{:.0%}".format),
        "Main reasons": view.main_reasons,
    })
    st.markdown("<div class='note'>Click a row to see that customer's details below.</div>", unsafe_allow_html=True)
    event = st.dataframe(show, hide_index=True, height=320, on_select="rerun",
                         selection_mode=["single-row", "single-cell"], key="target_table")
    st.download_button("Export list (CSV)", view.to_csv(index=False), "target_list.csv")

    sel = event.selection if event else {}
    rows = list(sel.get("rows", [])) + [c[0] for c in sel.get("cells", [])]
    idx = rows[0] if rows and rows[0] < len(view) else 0
    customer_detail(data, int(view.iloc[idx]["CustomerId"]))


def reason_list(factors, positive):
    f = factors[factors.effect_pts > 0] if positive else factors[factors.effect_pts < 0]
    f = f.head(4 if positive else 2)
    if f.empty:
        return "<li>None</li>"
    return "".join(f"<li>{label}<span class='pts'>{pts:+.1f} pts</span></li>" for label, pts in zip(f.label, f.effect_pts))


def customer_detail(data, cid):
    r = data.ranked.set_index("CustomerId").loc[cid]
    raw = data.customers.loc[cid]
    ex = explain_customer(data, cid)
    gain = r.uplift_score * roi.RETENTION_VALUE - roi.CONTACT_COST
    worth = gain > 0
    verdict = f"<span class='pill {'go' if worth else 'stop'}'>{'Contact' if worth else 'Do not contact'}</span>"

    with st.container(border=True):
        st.markdown(
            f"<div class='detail-head'><div><div class='detail-title'>Customer {cid}</div>"
            f"<div class='note'>{raw.Geography} · Age {raw.Age} · {raw.Tenure} year{'' if raw.Tenure == 1 else 's'} as a customer</div></div>"
            f"<div><span class='pill'>{r.priority_segment}</span><span class='pill'>Rank {int(r.priority_rank):,} "
            f"of {len(data.ranked):,}</span>{verdict}</div></div>", unsafe_allow_html=True)

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown("<div class='label'>Chance of staying</div>", unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=[r.retention_with_outreach * 100, r.retention_without_outreach * 100],
                y=["Contacted", "Not contacted"], orientation="h", marker_color=[NAVY, SILVER],
                text=[f"{r.retention_with_outreach:.0%}", f"{r.retention_without_outreach:.0%}"],
                textposition="outside", hoverinfo="skip"))
            fig.update_xaxes(range=[0, 110], showticklabels=False, gridcolor="white")
            fig.update_layout(bargap=0.35)
            fig.update_yaxes(tickfont=dict(size=13, color=INK))
            chart(fig, height=130, margin=dict(l=110, r=10, t=6, b=6))
            st.markdown(
                f"<div class='big'>{r.uplift_score * 100:+.1f} pts</div>"
                f"<div class='note'>from outreach. Expected value {dollars(r.uplift_score * roi.RETENTION_VALUE)} "
                f"vs {dollars(roi.CONTACT_COST)} cost ({dollars(gain)} net).</div>", unsafe_allow_html=True)
        with right:
            st.markdown(
                f"<div class='label'>Why outreach helps <span style='text-transform:none'>(from SHAP)</span></div>"
                f"<ul class='reasons'>{reason_list(ex['factors'], True)}</ul>"
                f"<div class='label'>Works against</div><ul class='reasons neg'>{reason_list(ex['factors'], False)}</ul>",
                unsafe_allow_html=True)

        traits = [("Active member", "Yes" if raw.IsActiveMember else "No"),
                  ("Past campaign", "Responded" if raw.previous_campaign_response else "No response"),
                  ("Products", raw.NumOfProducts), ("Engagement", f"{raw.engagement_score}/100"),
                  ("Support tickets", raw.support_tickets), ("Balance", f"${raw.Balance:,.0f}"),
                  ("Credit score", raw.CreditScore), ("Logins / month", raw.login_frequency)]
        st.markdown("<div class='traits'>" + "".join(
            f"<div><span>{k}</span><b>{v}</b></div>" for k, v in traits) + "</div>", unsafe_allow_html=True)

        with st.expander("Full SHAP breakdown (for analysts)"):
            f = ex["factors"].iloc[::-1]
            fig = go.Figure(go.Bar(x=f.effect_pts, y=f.label, orientation="h",
                                   marker_color=[NAVY if v > 0 else RUST for v in f.effect_pts],
                                   hovertemplate="%{y}: %{x:+.1f} pts<extra></extra>"))
            fig.add_vline(x=0, line_color="#9AA5B1", line_width=1)
            chart(fig, "Effect on uplift score (pts)", height=300)
            st.markdown(f"<div class='note'>Starts from the average customer ({ex['base_pts']:+.1f} pts); other traits "
                        f"add {ex['other_pts']:+.1f} pts, for {ex['uplift_pts']:+.1f} pts. The breakdown comes from "
                        f"the model trained on all customers; the ranking uses a model that never saw this customer "
                        f"({r.uplift_score * 100:+.1f} pts), so the two can differ slightly.</div>",
                        unsafe_allow_html=True)


def drivers_tab(data):
    ranked = data.ranked
    st.markdown("<div class='note' style='font-size:0.88rem'>Based on <b>SHAP analysis</b> of the model: which "
                "customer traits push the uplift score up or down, checked against real campaign results.</div>",
                unsafe_allow_html=True)

    st.subheader("Customer groups that respond best")
    rows = []
    for name, rule, action in DRIVER_GROUPS:
        mask = rule(ranked)
        rows.append({"Group": name, "Share of customers": f"{mask.mean():.0%}",
                     "Uplift in group": f"{observed_per_100(ranked[mask]):+.1f} pts",
                     "Everyone else": f"{observed_per_100(ranked[~mask]):+.1f} pts",
                     "Suggested approach": action})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.markdown(f"<div class='note'>Uplift = extra customers kept per 100 contacted, from real outcomes. "
                f"Across all customers it is {observed_per_100(ranked):+.1f} pts.</div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("What moves the uplift score (SHAP)")
        imp = data.importance.head(8).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.avg_impact_pts, y=imp.label, orientation="h", marker_color=STEEL,
                               customdata=imp.direction,
                               hovertemplate="%{y}: %{x:.1f} pts<br>%{customdata}<extra></extra>"))
        chart(fig, "Average effect on score (pts)", height=330)
    with right:
        st.subheader("Who is in Priority 1")
        p1 = ranked[ranked["priority_segment"] == "Priority 1"]
        st.dataframe(pd.DataFrame([{"Trait": name, "Priority 1": fmt.format(f(p1)), "All customers": fmt.format(f(ranked))}
                                   for name, f, fmt in PROFILE]), hide_index=True)

    with st.expander("SHAP plots (for analysts)"):
        st.markdown("<div class='note'>Feature effects on the uplift score from SHAP. They describe patterns the "
                    "model learned, not proven causes of retention.</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if SHAP_IMAGES["summary"].exists():
            c1.image(str(SHAP_IMAGES["summary"]))
        if SHAP_IMAGES["top_factors"].exists():
            c2.image(str(SHAP_IMAGES["top_factors"]))


def planner_tab(data):
    c = st.columns(3)
    share = c[0].slider("Customers contacted (%)", 5, 100, 30, step=5) / 100
    cost = c[1].number_input("Cost per contact ($)", min_value=1.0, max_value=500.0,
                             value=float(roi.CONTACT_COST), step=1.0)
    value = c[2].number_input("Value per customer kept ($)", min_value=10.0, max_value=10000.0,
                              value=float(roi.RETENTION_VALUE), step=10.0)

    s = data.strategy_scores
    r = roi.simulate(s["uplift_final"].values, share, cost, value,
                     treatment=s["treatment"].values, outcome=s["outcome"].values)
    m = st.columns(5)
    m[0].metric("Customers contacted", f"{r['n_contacted']:,}")
    m[1].metric("Extra customers kept", f"{r['predicted_incremental']:,.0f}")
    m[2].metric("Campaign cost", dollars(r["predicted_cost"]))
    m[3].metric("Net benefit", dollars(r["predicted_net_benefit"]))
    m[4].metric("ROI", f"{r['predicted_roi']:.0%}")

    table = roi.roi_table(s["uplift_final"].values, cost, value)
    best = table.loc[table["predicted_net_benefit"].idxmax()]
    st.markdown(f"<div class='note'>Model estimate. Measured on real outcomes: {r['observed_incremental']:,.0f} extra "
                f"customers kept, net benefit {dollars(r['observed_net_benefit'])}. At these costs the best size is "
                f"the top {best.outreach_share:.0%} ({dollars(best.predicted_net_benefit)}).</div>",
                unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    comp = comparison_table(cost, value)
    with left:
        st.subheader("Net benefit by targeting method")
        fig = go.Figure()
        for name in [UPLIFT, CHURN, RANDOM]:
            color, dash, width = STRATEGY_STYLE[name]
            d = comp[comp["strategy"] == name]
            fig.add_scatter(x=d.outreach_share * 100, y=d.net_benefit, name=name.replace(" targeting", ""),
                            mode="lines", line=dict(color=color, dash=dash, width=width),
                            hovertemplate="%{x:.0f}%: $%{y:,.0f}<extra>" + name + "</extra>")
        fig.add_vline(x=share * 100, line_color="#9AA5B1", line_width=1)
        fig.update_yaxes(tickprefix="$", tickformat="~s")
        fig.update_xaxes(ticksuffix="%")
        chart(fig, "Customers contacted", "")
    with right:
        st.subheader(f"At {share:.0%} contacted")
        at = comp[comp["outreach_share"].round(2) == round(share, 2)]
        st.dataframe(pd.DataFrame({
            "Method": at.strategy.str.replace(" targeting", ""),
            "Extra kept": at.observed_incremental.round(0).astype(int),
            "Net benefit": at.net_benefit.map(dollars),
            "ROI": at.roi.map("{:.0%}".format),
        }), hide_index=True)
        st.markdown("<div class='note'>Same budget for every method; only who is contacted changes. "
                    "Measured on real outcomes.</div>", unsafe_allow_html=True)

    with st.expander("Assumptions"):
        st.markdown(
            "- Cost and value are the same for every customer and are placeholders for the sponsor's figures.\n"
            "- An uplift score of 0.20 means outreach raises a customer's chance of staying by 20 points.\n"
            "- Data is synthetic: figures show how the method performs, not expected revenue.\n"
            "- Details: `reports/roi_assumptions.md`, `reports/targeting_comparison.md`.")


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    missing = missing_files()
    if missing:
        st.error("Missing results. Run these scripts first:\n\n"
                 + "\n".join(f"- `{s}` (creates `{f}`)" for f, s in missing))
        st.stop()
    data = get_data()
    header(data)
    t1, t2, t3 = st.tabs(["Priority customers", "Why customers respond", "Campaign planner"])
    with t1:
        priority_tab(data)
    with t2:
        drivers_tab(data)
    with t3:
        planner_tab(data)


main()