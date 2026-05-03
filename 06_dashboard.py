import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from shared_style import (
    ERA_COLORS, ERA_ORDER, ERA_LABELS, ERA_LABELS_FULL, ERA_PRICES, PRICE_ERAS,
    PROMO_COLOR, ACCENT_COLOR, NEUTRAL_COLOR,
    POSITIVE_COLOR, NEGATIVE_COLOR, ANNUAL_COLOR,
    BG_CHART, BG_HOVER, BORDER, GRID, TEXT_PRIMARY, TEXT_SECONDARY,
    FONT_FAMILY, SOURCE_NOTE, PRELIMINARY_NOTE, PRELIMINARY_START, CUTOFF_DATE,
    style_fig, add_era_vlines, hex_rgba,
    era_for_month, fmt_pln, fmt_pct, fmt_pval, fmt_months, fmt_period,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KDS Subscription Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stMetricValue"] {
      font-size: 2rem !important; font-weight: 700 !important;
      font-family: Inter, sans-serif !important;
  }
  [data-testid="stMetricLabel"] {
      font-size: 0.78rem !important; font-weight: 500 !important;
      letter-spacing: 0.04em; text-transform: uppercase; color: #94A3B8 !important;
  }
  [data-testid="stMetricDelta"] { font-size: 0.82rem !important; color: #4ADE80 !important; }
  [data-testid="stMetricDeltaIcon"] { display: none !important; }
  h2 { font-family: Inter, sans-serif; font-weight: 700; letter-spacing: -0.02em; }
  h3 { font-family: Inter, sans-serif; font-weight: 600; color: #94A3B8;
       font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.06em;
       margin-top: 1.6rem; }
  .caption-note { font-size: 0.78rem; color: #94A3B8; margin-top: 0.3rem;
                  font-family: Inter, sans-serif; }
  hr { border-color: #2E3347; margin: 1.4rem 0; }
  .block-container { padding-top: 1.8rem; }
</style>
""", unsafe_allow_html=True)

RAW_PATH       = "data/KDS Transactions.xlsx"
CUSTOMERS_PATH = "data/df_customers.csv"

# ── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    cust = pd.read_csv(CUSTOMERS_PATH, parse_dates=["entry_date", "last_payment"])
    cust["entry_month"] = cust["entry_month"].apply(lambda x: pd.Period(x, "M"))
    df   = pd.read_excel(RAW_PATH)
    df.columns = ["date", "client", "amount"]
    df["tx_month"]    = df["date"].dt.to_period("M")
    df["tx_month_ts"] = df["tx_month"].dt.to_timestamp()
    first_tx          = df.groupby("client")["date"].min()
    df["entry_month"] = df["client"].map(first_tx).dt.to_period("M")
    df["entry_era"]   = df["entry_month"].apply(era_for_month)
    return cust, df

cust, df_raw = load_data()
monthly = cust[~cust["is_annual"]].copy()
annual  = cust[cust["is_annual"]].copy()

# ── SHARED HELPERS ────────────────────────────────────────────────────────────
CUTOFF_PERIOD = pd.Period("2026-03", freq="M")

def retention_at_offset(cust_subset, df_transactions, offsets):
    """For a customer subset, return dict offset→retention% using valid denominator.

    valid_n for offset M = subscribers whose entry_month + M <= cutoff (Mar 2026).
    Only they had enough time to reach that offset — using total would deflate rates.
    """
    ids = set(cust_subset["client_id"])
    tx  = df_transactions[df_transactions["client"].isin(ids)].copy()
    cohort_map   = cust_subset.set_index("client_id")["entry_month"].to_dict()
    tx["cohort"] = tx["client"].map(cohort_map)
    tx["offset"] = (tx["tx_month"] - tx["cohort"]).apply(lambda x: x.n)
    entry_months = cust_subset["entry_month"].values
    result = {}
    for mo in offsets:
        valid_n = int(
            sum(1 for em in entry_months if (CUTOFF_PERIOD - em).n >= mo)
        )
        active  = tx[tx["offset"] == mo]["client"].nunique()
        result[mo] = round(active / valid_n * 100, 1) if valid_n > 0 else None
    return result

def add_km_trace_to(fig, grp, color, label, show_ci=True):
    if len(grp) < 5:
        return None
    kmf = KaplanMeierFitter()
    kmf.fit(grp["km_duration"], event_observed=grp["km_event"])
    t = kmf.survival_function_.index.tolist()
    s = kmf.survival_function_.iloc[:, 0].tolist()
    fig.add_trace(go.Scatter(
        x=t, y=s, mode="lines",
        name=f"{label}  (n={len(grp)})",
        line=dict(color=color, width=2.5),
        hovertemplate=(
            f"<b>{label}</b><br>"
            f"Month %{{x:.0f}}: %{{y:.0%}} still subscribed<extra></extra>"
        ),
    ))
    if show_ci:
        ci_u = kmf.confidence_interval_.iloc[:, 1].tolist()
        ci_l = kmf.confidence_interval_.iloc[:, 0].tolist()
        t_ci = kmf.confidence_interval_.index.tolist()
        r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        fig.add_trace(go.Scatter(
            x=t_ci+t_ci[::-1], y=ci_u+ci_l[::-1],
            fill="toself", fillcolor=f"rgba({r},{g},{b},0.10)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
    return kmf.median_survival_time_

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    _logo_col1, _logo_col2 = st.columns(2)
    _logo_col1.image("kds_logo.png", use_container_width=True)
    _logo_col2.image("da_logo.png", use_container_width=True)
    st.markdown("## KDS Analytics")
    st.markdown(
        f"<div class='caption-note'>Data through {CUTOFF_DATE.strftime('%b %Y')}"
        f"<br>{len(cust):,} subscribers · {len(df_raw):,} payments</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Who signed up?", "Who stayed?",
         "How long did they stay?", "What did they generate?"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div class='caption-note'>249 PLN era: Sep '25 to Mar '26 only.<br>"
        "Long-term retention conclusions are not yet possible.<br><br>"
        "Visualisations comply with WCAG 2.1 AA<br>"
        "(min. contrast ratio 6.4:1).</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("KDS Subscription Analytics")
    st.markdown(
        "The price of a KDS subscription rose four times from **99 PLN** to **249 PLN** "
        "over 29 months (Nov 2023 to Mar 2026). "
        "This report analyses how each price change affected **acquisition, retention, and "
        "lifetime value**, and how promotional discounts shaped subscriber behaviour."
    )

    st.markdown(
        "**Did more customers sign up as prices rose?** Yes. "
        "Acquisition accelerated with each price era: "
        "the 249 PLN era added subscribers at the fastest monthly rate of all four eras. "
        "Price did not suppress demand.\n\n"
        "**Did subscribers leave faster?** Yes. "
        "Median tenure dropped from 6 months at 99 PLN to 2 months at 199 PLN, "
        "a 67% fall. Higher price, shorter stay.\n\n"
        "**Did promotions help or hurt?** Neither decisively. "
        "Promo subscribers stay slightly longer but generate lower lifetime revenue. "
        "The LTV gap is not statistically significant in any era. "
        "The one clear signal: first-month retention is lower for promo subscribers "
        "in the 99 and 169 PLN eras."
    )

    with st.expander("Source data: format and sample"):
        st.markdown(
            "Raw input: a single Excel file with one row per payment transaction "
            f"(**4,227 transactions**, **{len(cust):,} unique subscribers**)."
        )
        sample = pd.DataFrame({
            "date":   ["2023-11-05 18:04:23", "2023-11-05 18:18:50", "2023-11-05 18:23:21"],
            "client": [1, 2, 3],
            "amount": [35.60, 89.00, 378.25],
        })
        st.dataframe(sample, use_container_width=False, hide_index=True)


    st.markdown("---")

    total           = len(cust)
    cutoff_ts       = pd.Timestamp("2026-03-31")
    _active_period  = pd.Period("2026-03", freq="M")
    monthly_active  = int(
        (monthly["last_payment"].dt.to_period("M") == _active_period).sum()
    )
    longplan_active = int(
        (annual["last_payment"] >= cutoff_ts - pd.DateOffset(years=1)).sum()
    )
    mrr          = int(monthly[monthly["is_active"]]["entry_price"].sum())
    rev          = int(cust["total_revenue"].sum())
    m3_vals = []
    for era in ERA_ORDER:
        ret = retention_at_offset(monthly[monthly["entry_era"] == era], df_raw, [3])
        if ret[3] is not None:
            m3_vals.append(ret[3])
    m3 = np.mean(m3_vals) if m3_vals else 0

    # promo KPI
    promo_sub     = monthly[monthly["is_promo"]]
    full_sub      = monthly[~monthly["is_promo"]]
    promo_pct     = len(promo_sub) / len(monthly) * 100
    monthly_rev   = int(monthly["total_revenue"].sum())
    longplan_rev  = int(annual["total_revenue"].sum())
    promo_rev     = int(promo_sub["total_revenue"].sum())
    full_rev      = int(full_sub["total_revenue"].sum())
    promo_rev_pct = promo_rev / monthly_rev * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total subscribers",      f"{total:,}")
    c2.metric("Monthly active",         f"{monthly_active:,}", delta="paid in Mar 2026",           delta_color="off")
    c3.metric(
        "Long plan active",
        f"{longplan_active:,}",
        delta="paid within last 12 months",
        delta_color="off",
    )
    c4.metric("Est. monthly revenue",   f"{mrr:,} PLN")
    c5.metric("Revenue, all time",      f"{rev:,} PLN")

    st.markdown("#### Revenue split & promo impact")
    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Monthly plans revenue",   f"{monthly_rev:,} PLN",  delta=f"{monthly_rev/rev*100:.0f}% of all revenue",   delta_color="off")
    c7.metric("Long plans revenue",      f"{longplan_rev:,} PLN", delta=f"{longplan_rev/rev*100:.0f}% of all revenue",  delta_color="off")
    c8.metric("Promo subscribers",       f"{len(promo_sub):,}",   delta=f"{promo_pct:.0f}% of monthly base",            delta_color="off")
    c9.metric("Promo revenue (monthly)", f"{promo_rev:,} PLN",    delta=f"{promo_rev_pct:.0f}% of monthly revenue",     delta_color="off")
    c10.metric("Full-price revenue",     f"{full_rev:,} PLN",     delta=f"{100-promo_rev_pct:.0f}% of monthly revenue", delta_color="off")
    st.markdown(
        "<div class='caption-note'>CAC (cost of acquisition) data is not available</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### New subscribers over time")
    signups = (
        cust.groupby("entry_month")
        .size()
        .reset_index(name="n")
    )
    signups["ts"] = signups["entry_month"].dt.to_timestamp()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=signups["ts"], y=signups["n"],
        mode="lines+markers",
        line=dict(color=ACCENT_COLOR, width=2),
        marker=dict(size=5, color=ACCENT_COLOR),
        hovertemplate="%{x|%b '%y}: <b>%{y}</b> new subscribers<extra></extra>",
        showlegend=False,
    ))
    add_era_vlines(fig)
    fig = style_fig(
        fig,
        title="New subscribers per month",
        subtitle="Vertical lines mark price increases",
        height=320,
        xlab="Month",
        ylab="New subscribers",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    st.markdown("### Key findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Subscription length dropped as prices rose**\n\n"
            "Subscribers who joined at 99 PLN stayed for a median of **6 months**. "
            "At 199 PLN, that fell to **2 months** (a 67% drop). "
            "The difference is statistically significant across every price increase.\n\n"
            "**The revenue sweet spot was 199 PLN**\n\n"
            "Despite shorter median tenure, the 199 PLN era generated the most total "
            "revenue. It combined a higher price with the largest subscriber base of "
            "any era."
        )
    with col2:
        st.markdown(
            "**249 PLN: too early to judge**\n\n"
            "The current era has only 7 months of data. Active rates look strong (56%), "
            "but most subscribers have not had enough time to churn yet.\n\n"
            "**Promo codes: LTV gap smaller than expected**\n\n"
            "Discount subscribers stay slightly longer, and the LTV difference vs "
            "full-price is statistically insignificant in every era. "
            "The one exception: first-month (M+1) retention is noticeably lower "
            "for promo subscribers in the 99 and 169 PLN eras."
        )
    st.markdown(
        "**The subscriber base is growing**\n\n"
        "Despite shorter retention, the raw number of new sign-ups has increased "
        "with each price era. The 249 PLN era recorded the highest acquisition pace "
        "of all four eras. Higher prices have not stopped demand."
    )
    st.markdown(
        "A higher price naturally raises the bar. Potential subscribers evaluate "
        "more carefully whether analytics is the right path for them, and it is "
        "easier to walk away. The data does not show dramatic churn spikes at any "
        "price point, but the downward trend in tenure is clear and worth watching. "
        "Retention started softening at 199 PLN, which is the threshold to monitor "
        "closely as the 249 PLN era matures. "
        "One early positive signal: M+3 retention for 249 PLN subscribers is "
        "currently higher than it was for 199 PLN at the same observation point."
    )
    st.markdown(
        f"<div class='caption-note'>⚠ {PRELIMINARY_NOTE}</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WHO SIGNED UP?
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Who signed up?":
    st.title("Who signed up?")
    st.markdown(
        "Acquisition grew with every price increase. "
        "The 249 PLN era recorded the highest monthly sign-up rate of all four eras, "
        "and September spikes in 2024 and 2025 point to campaign-driven demand "
        "independent of price. "
        "New course launches (including the Intro to Data Science in Python) "
        "likely brought in a different audience pool and supported continued growth. "
        "Demand held up. People kept signing up."
    )
    st.markdown("---")

    first_df = cust[["client_id", "entry_month", "entry_era", "is_promo"]].copy()
    first_df["entry_ts"] = first_df["entry_month"].dt.to_timestamp()

    breakdown = st.radio(
        "Show breakdown",
        ["By price era", "By price era + promo split"],
        horizontal=True,
    )

    fig = go.Figure()

    if breakdown == "By price era":
        acq = first_df.groupby(["entry_ts", "entry_era"]).size().reset_index(name="n")
        for era in ERA_ORDER:
            sub = acq[acq["entry_era"] == era]
            fig.add_trace(go.Bar(
                x=sub["entry_ts"], y=sub["n"],
                name=ERA_LABELS_FULL[era],
                marker_color=ERA_COLORS[era], marker_line=dict(width=0),
                hovertemplate=(
                    f"<b>{ERA_LABELS[era]}</b>"
                    f"<br>%{{x|%b '%y}}: %{{y}} new subscribers<extra></extra>"
                ),
            ))
        subtitle = (
            "September spikes in 2024 and 2025 suggest seasonal or "
            "campaign-driven demand, independent of price"
        )
    else:
        # stack: full price (solid) + promo (translucent same color)
        acq = (
            first_df
            .groupby(["entry_ts", "entry_era", "is_promo"])
            .size()
            .reset_index(name="n")
        )
        for era in ERA_ORDER:
            color = ERA_COLORS[era]
            r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
            for is_promo, seg_label, opacity in [
                (False, "Full price", 1.0),
                (True,  "Promo",      0.45),
            ]:
                sub = acq[(acq["entry_era"]==era) & (acq["is_promo"]==is_promo)]
                fig.add_trace(go.Bar(
                    x=sub["entry_ts"], y=sub["n"],
                    name=f"{ERA_LABELS[era]} ({seg_label})",
                    marker_color=f"rgba({r},{g},{b},{opacity})",
                    marker_line=dict(width=0),
                    legendgroup=era,
                    hovertemplate=(
                        f"<b>{ERA_LABELS[era]} · {seg_label}</b>"
                        f"<br>%{{x|%b '%y}}: %{{y}} subscribers<extra></extra>"
                    ),
                ))
        subtitle = (
            "Solid = full price · Transparent = promo discount · "
            "same color = same price era"
        )

    fig.add_annotation(
        x=pd.Timestamp("2024-07-15").timestamp()*1000, y=10,
        text="Sales gap<br>Jul – Aug '24",
        showarrow=True, arrowhead=2, arrowcolor=NEUTRAL_COLOR, arrowwidth=1.2,
        font=dict(size=10, color=NEUTRAL_COLOR, family=FONT_FAMILY),
        bgcolor=BG_CHART, bordercolor=BORDER, borderwidth=1,
    )
    style_fig(fig,
              title="New subscriber growth by price era",
              subtitle=subtitle,
              xlab="Month", ylab="New subscribers", height=480)
    fig.update_layout(
        barmode="stack",
        legend=dict(
            orientation="h", y=-0.26, x=0,
            title=dict(
                text="Price era · segment",
                font=dict(size=11, color=TEXT_SECONDARY),
            ),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<div class='caption-note'>"
        "⚠ <b>New sign-ups only.</b> Each bar counts subscribers whose "
        "<em>first ever payment</em> fell in that month. "
        "Returning or retained subscribers are not counted here. "
        "The chart below shows the full active base."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    col1, col2 = st.columns([3, 2])

    with col1:
        # Active monthly subscribers per month — split full price vs promo
        promo_ids = set(monthly[monthly["is_promo"]]["client_id"])
        full_ids  = set(monthly[~monthly["is_promo"]]["client_id"])
        monthly_ids = promo_ids | full_ids

        tx_monthly = df_raw[df_raw["client"].isin(monthly_ids)].copy()
        full_mo  = (
            tx_monthly[tx_monthly["client"].isin(full_ids)]
            .groupby("tx_month_ts")["client"].nunique()
        )
        promo_mo = (
            tx_monthly[tx_monthly["client"].isin(promo_ids)]
            .groupby("tx_month_ts")["client"].nunique()
        )
        all_mo   = tx_monthly.groupby("tx_month_ts")["client"].nunique()

        # Align all series to same index
        idx = all_mo.index
        full_mo  = full_mo.reindex(idx, fill_value=0)
        promo_mo = promo_mo.reindex(idx, fill_value=0)

        r, g, b = (
            int(ACCENT_COLOR[1:3], 16),
            int(ACCENT_COLOR[3:5], 16),
            int(ACCENT_COLOR[5:7], 16),
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=idx, y=full_mo.values,
            name="Full price",
            marker_color=ACCENT_COLOR, marker_line=dict(width=0), opacity=1.0,
            hovertemplate="%{x|%b '%y} · full price: %{y}<extra></extra>",
        ))
        fig2.add_trace(go.Bar(
            x=idx, y=promo_mo.values,
            name="Promo discount",
            marker_color=f"rgba({r},{g},{b},0.40)", marker_line=dict(width=0),
            hovertemplate="%{x|%b '%y} · promo: %{y}<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=idx, y=all_mo.values, mode="lines",
            name="Total",
            line=dict(color=NEUTRAL_COLOR, width=1.8, dash="dot"),
            hovertemplate="%{x|%b '%y} · total: %{y}<extra></extra>",
        ))
        add_era_vlines(fig2)
        style_fig(
            fig2,
            title="Active monthly subscribers per month",
            subtitle=(
                "All monthly-plan subscribers who paid in each month, "
                "full active base, not just new sign-ups"
            ),
            xlab="Month", ylab="Active subscribers", height=380,
        )
        fig2.update_layout(
            barmode="stack",
            legend=dict(orientation="h", y=-0.22, x=0),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(
            "<div class='caption-note'>"
            "Solid = full price · Transparent = promo discount · dotted line = total. "
            "The promo/full-price split is based on each subscriber's "
            "<b>entry price (locked in at sign-up)</b>: "
            "a subscriber who joined on a promo discount always counts as 'promo', "
            "even months later. "
            "This is why the promo segment grows over time as promo subscribers "
            "accumulate in the active base, "
            "while the top chart above shows only how many new promo sign-ups "
            "occurred each month. "
            "<br>Long plans (annual/semi-annual) are not shown. They pay once a year, "
            "so monthly counts are not meaningful."
            "</div>",
            unsafe_allow_html=True,
        )

    with col2:
        era_months_map = {"era_99":10,"era_169":2,"era_199":10,"era_249":7}
        rows = []
        for era in ERA_ORDER:
            sub   = monthly[monthly["entry_era"]==era]
            n     = len(sub)
            mo    = era_months_map[era]
            n_promo = sub["is_promo"].sum()
            rows.append({
                "Era":           ERA_LABELS[era],
                "Duration":      f"{mo} mo",
                "Total":         n,
                "Full price":    n - n_promo,
                "Promo":         n_promo,
                "% promo":       fmt_pct(n_promo/n*100),
                "New / month":   f"{n/mo:.1f}",
            })
        st.markdown("#### Acquisition by era: monthly plans")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown(
            "<div class='caption-note'>Use 'New / month' for fair comparison "
            "— eras have different durations.</div>", unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='caption-note'>"
            "Promo classification is based on each subscriber's first payment only. "
            "If the first payment was more than 5% below the official era price, "
            "the subscriber is flagged as promo permanently. "
            "This cannot be affected by later payments at a different amount."
            "</div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WHO STAYED?
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Who stayed?":
    st.title("Who stayed?")
    st.markdown(
        "Retention fell with each price increase. "
        "Subscribers who joined at 99 PLN were still paying at month 3 at a higher "
        "rate than any later era. At 199 PLN, most had churned within two months. "
        "For long-plan subscribers the picture is different: renewal rates are low "
        "across all eras (around 1 in 3), but those who do renew tend to be committed."
    )
    st.markdown("---")

    plan_filter = st.radio(
        "Plan type",
        ["Monthly plans", "Long plans (annual / semi-annual)"],
        horizontal=True,
    )

    # ── LONG PLANS: simple renewal rate bar ───────────────────────────────────
    if plan_filter == "Long plans (annual / semi-annual)":
        st.markdown(
            "Long plans require ~12 months before a subscriber can decide to renew. "
            "Only subscribers whose plan window had **already closed** before the "
            "data cutoff "
            "(Mar 2026) are included here. Showing them before that point would "
            "be meaningless."
        )
        st.markdown(
            "**era_249 long plans are not shown.** All 88 subscribers in this era "
            "joined "
            "from Sep 2025 onward and have not yet reached their 12-month renewal "
            "point. "
            "**era_169** has a small sample (n=17). Treat with caution."
        )
        st.markdown("---")

        CUTOFF_PERIOD = pd.Period("2026-03", freq="M")
        annual["renewal_month"] = annual["entry_month"] + 12
        observable = annual[annual["renewal_month"] <= CUTOFF_PERIOD].copy()

        renewal_rows = []
        for era in ["era_99", "era_169", "era_199"]:
            sub = observable[observable["entry_era"] == era]
            if len(sub) < 3:
                continue
            renewed = int(sub["is_active"].sum())
            n = len(sub)
            renewal_rows.append({
                "era":     era,
                "label":   ERA_LABELS[era],
                "n":       n,
                "renewed": renewed,
                "churned": n - renewed,
                "renewal_pct": renewed / n * 100,
            })

        fig = go.Figure()
        for row in renewal_rows:
            color = ERA_COLORS[row["era"]]
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fig.add_trace(go.Bar(
                x=[row["label"]], y=[row["renewal_pct"]],
                name=row["label"],
                marker_color=color, marker_line=dict(width=0),
                text=[f"{row['renewal_pct']:.0f}%<br>(n={row['n']})"],
                textposition="outside",
                textfont=dict(size=12, color=TEXT_SECONDARY),
                hovertemplate=(
                    f"<b>{row['label']}</b><br>"
                    f"Renewed: {row['renewed']} of {row['n']}"
                    f" ({row['renewal_pct']:.0f}%)<br>"
                    f"Churned: {row['churned']}<extra></extra>"
                ),
                showlegend=False,
            ))
        style_fig(
            fig,
            title="Long-plan renewal rate",
            subtitle=(
                "1 in 3 subscribers renews after the first year. "
                "Only subscribers whose 12-month window closed before "
                "Mar 2026 are counted"
            ),
            xlab="Price era", ylab="Renewal rate (%)", height=420,
        )
        fig.update_layout(yaxis=dict(range=[0, 55]))
        st.plotly_chart(fig, use_container_width=True)

        tbl = pd.DataFrame([{
            "Era":       r["label"],
            "Observable (n)": r["n"],
            "Renewed":   r["renewed"],
            "Churned":   r["churned"],
            "Renewal rate": fmt_pct(r["renewal_pct"]),
        } for r in renewal_rows])
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.markdown(
            "<div class='caption-note'>"
            "⚠ era_249 long plans: 0 observable (all joined Sep 2025 or later, "
            "renewal window opens Sep 2026+). "
            "era_169: n=17 only, interpret with caution. "
            "Data cutoff: Mar 2026."
            "</div>", unsafe_allow_html=True,
        )

    # ── MONTHLY PLANS: retention curves ───────────────────────────────────────
    else:
        st.markdown(
            "Retention curves show what share of each era's subscribers were still "
            "paying "
            "at 1, 2, 3… months after sign-up. Calculated as an era-level aggregate, "
            "which eliminates month-to-month noise of per-cohort views."
        )

        seg_filter = st.radio(
            "Subscriber segment",
            [
                "All subscribers",
                "Full price only",
                "Promo discount only",
                "Full price vs promo",
            ],
            horizontal=True,
        )

        # era_249 caps at M+3 — only offsets with valid_n > 0 are shown
        ERA_MAX_OFFSET = {"era_99": 12, "era_169": 12, "era_199": 12, "era_249": 3}
        OFFSETS = list(range(1, 13))
        MILESTONE_OFFSETS = [1, 3, 6, 12]

        fig = go.Figure()

        if seg_filter != "Full price vs promo":
            for era in ERA_ORDER:
                era_sub = monthly[monthly["entry_era"] == era]
                if seg_filter == "Full price only":
                    era_sub = era_sub[~era_sub["is_promo"]]
                elif seg_filter == "Promo discount only":
                    era_sub = era_sub[era_sub["is_promo"]]
                if len(era_sub) < 5:
                    continue
                max_off = ERA_MAX_OFFSET[era]
                offsets_era = [o for o in OFFSETS if o <= max_off]
                ret = retention_at_offset(era_sub, df_raw, offsets_era)
                x_vals = [o for o in offsets_era if ret.get(o) is not None]
                y_vals = [ret[o] for o in x_vals]
                color  = ERA_COLORS[era]
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode="lines+markers",
                    name=f"{ERA_LABELS[era]}  (n={len(era_sub)})",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=5, color=color),
                    fill="tozeroy",
                    fillcolor=f"rgba({r},{g},{b},0.04)",
                    hovertemplate=(
                        f"<b>{ERA_LABELS[era]}</b>"
                        f"<br>Month %{{x}}: %{{y:.0f}}% still paying<extra></extra>"
                    ),
                ))
            title_seg = {
                "All subscribers":    "all subscribers",
                "Full price only":    "full-price subscribers",
                "Promo discount only":"promo-discount subscribers",
            }.get(seg_filter, "")
            chart_title = f"Retention by price era: {title_seg}"
            subtitle    = (
                "Each point = % of subscribers who had enough time to reach "
                "that month and were still paying"
            )
        else:
            for seg_label, mask, color, dash in [
                ("Full price",     ~monthly["is_promo"], ACCENT_COLOR, "solid"),
                ("Promo discount",  monthly["is_promo"], PROMO_COLOR,  "dash"),
            ]:
                sub = monthly[mask]
                if len(sub) < 5:
                    continue
                ret    = retention_at_offset(sub, df_raw, OFFSETS)
                x_vals = [o for o in OFFSETS if ret.get(o) is not None]
                y_vals = [ret[o] for o in x_vals]
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode="lines+markers",
                    name=f"{seg_label}  (n={len(sub)})",
                    line=dict(color=color, width=2.5, dash=dash),
                    marker=dict(size=5, color=color),
                    hovertemplate=(
                        f"<b>{seg_label}</b>"
                        f"<br>Month %{{x}}: %{{y:.0f}}% still paying<extra></extra>"
                    ),
                ))
            chart_title = "Full price vs promo discount: overall retention comparison"
            subtitle    = (
                "Solid = full price · Dashed = promo discount · all eras combined"
            )

        style_fig(fig, title=chart_title, subtitle=subtitle,
                  xlab="Months since sign-up", ylab="Still subscribed (%)", height=480)
        fig.update_layout(
            yaxis=dict(range=[0, 105]),
            xaxis=dict(tickmode="array", tickvals=OFFSETS,
                       ticktext=[f"M+{o}" for o in OFFSETS]),
            legend=dict(orientation="h", y=-0.2, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "<div class='caption-note'>"
            "Denominator = subscribers who had enough time to reach each month. "
            "⚠ <b>249 PLN era:</b> shown only through M+3 "
            "(158 of 265 subscribers reached that offset). "
            "M+6 and beyond require data from Sep 2026 onward."
            "</div>", unsafe_allow_html=True,
        )

        # ── Milestone table ────────────────────────────────────────────────────
        st.markdown("#### Retention at key milestones")

        def fmt_ret(val, era, offset):
            """Format retention value; show — if offset exceeds era's observable
            window."""
            if offset > ERA_MAX_OFFSET.get(era, 12):
                return "—"
            if val is None:
                return "—"
            return fmt_pct(val)

        table_rows = []
        for era in ERA_ORDER:
            era_sub   = monthly[monthly["entry_era"] == era]
            full_sub  = era_sub[~era_sub["is_promo"]]
            promo_sub = era_sub[ era_sub["is_promo"]]

            for sub, seg in [
                (era_sub,   "All"),
                (full_sub,  "Full price"),
                (promo_sub, "Promo"),
            ]:
                if len(sub) < 5:
                    continue
                ret = retention_at_offset(sub, df_raw, MILESTONE_OFFSETS)
                row = {
                    "Era":     ERA_LABELS[era],
                    "Segment": seg,
                    "n":       len(sub),
                    "M+1":     fmt_ret(ret[1], era, 1),
                    "M+3":     fmt_ret(ret[3], era, 3),
                    "M+6":     fmt_ret(ret[6], era, 6),
                    "M+12":    fmt_ret(ret[12], era, 12),
                }
                table_rows.append(row)

        tbl_df = pd.DataFrame(table_rows)
        if seg_filter == "All subscribers":
            tbl_df = tbl_df[tbl_df["Segment"] == "All"]
        elif seg_filter == "Full price only":
            tbl_df = tbl_df[tbl_df["Segment"] == "Full price"]
        elif seg_filter == "Promo discount only":
            tbl_df = tbl_df[tbl_df["Segment"] == "Promo"]
        else:
            tbl_df = tbl_df[tbl_df["Segment"].isin(["Full price", "Promo"])]

        _drop_cols = (
            ["Segment"] if seg_filter != "Full price vs promo" else []
        )
        st.dataframe(
            tbl_df.drop(columns=_drop_cols),
            use_container_width=True, hide_index=True,
        )
        st.markdown(
            "<div class='caption-note'>"
            "n/a = not enough time elapsed for that era to reach this offset. "
            "249 PLN era: M+6 and M+12 are n/a (subscribers joined Sep 2025 to "
            "Mar 2026 at latest, "
            "M+3 is the last fully observable offset)."
            "</div>", unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOW LONG DID THEY STAY?
# ══════════════════════════════════════════════════════════════════════════════
elif page == "How long did they stay?":
    st.title("How long did they stay?")
    st.markdown(
        "Subscribers at 99 PLN lasted twice as long as those at 199 PLN. "
        "Kaplan-Meier survival curves below show the probability of still being "
        "subscribed after T months, with statistical significance confirmed by "
        "log-rank tests. Right-censoring is handled correctly: subscribers still "
        "active at the data cutoff are not counted as churned."
    )
    st.markdown("---")

    plan_filter = st.radio(
        "Plan type",
        ["Monthly plans", "Long plans (annual / semi-annual)"],
        horizontal=True,
    )

    # ── LONG PLANS: summary table only ────────────────────────────────────────
    if plan_filter == "Long plans (annual / semi-annual)":
        st.markdown(
            "Long-plan subscribers pay once a year. The KM survival curve is not "
            "suitable here: "
            "there is only one decision point per subscriber (renew or not at "
            "~12 months). "
            "The table below shows renewal rates for eras where that decision "
            "window has already closed."
        )
        st.markdown("---")

        CUTOFF_PERIOD = pd.Period("2026-03", freq="M")
        annual_copy = annual.copy()
        annual_copy["renewal_month"] = annual_copy["entry_month"] + 12
        observable = annual_copy[annual_copy["renewal_month"] <= CUTOFF_PERIOD]

        lp_rows = []
        for era in ERA_ORDER:
            sub_all = annual_copy[annual_copy["entry_era"] == era]
            sub_obs = observable[observable["entry_era"] == era]
            n_total  = len(sub_all)
            n_obs    = len(sub_obs)
            renewed  = int(sub_obs["is_active"].sum()) if n_obs > 0 else 0
            lp_rows.append({
                "Era":              ERA_LABELS[era],
                "Total (n)":        n_total,
                "Observable (n)":   n_obs if n_obs > 0 else "—",
                "Renewal rate":     (
                    fmt_pct(renewed / n_obs * 100) if n_obs > 0 else "N/A"
                ),
                "Renewed":          renewed if n_obs > 0 else "N/A",
                "Churned":          (n_obs - renewed) if n_obs > 0 else "N/A",
                "Median LTV":       fmt_pln(sub_all["total_revenue"].median()),
                "Avg LTV":          fmt_pln(sub_all["total_revenue"].mean()),
                "Median entry price": fmt_pln(sub_all["entry_price"].median()),
                "Avg entry price":  fmt_pln(sub_all["entry_price"].mean()),
            })

        st.dataframe(pd.DataFrame(lp_rows), use_container_width=True, hide_index=True)
        st.markdown(
            "<div class='caption-note'>"
            "Observable = subscribers whose 12-month plan window closed before "
            "Mar 2026. "
            "era_249: all joined Sep 2025 or later, renewal window opens Sep 2026+. "
            "era_169: n=17 only, interpret with caution. "
            "Long plans have no promo subscribers in this dataset."
            "</div>", unsafe_allow_html=True,
        )

    # ── MONTHLY PLANS: KM survival curves ─────────────────────────────────────
    else:
        view_mode = st.radio(
            "View",
            ["By price era", "Full price vs promo (overall)", "By era + promo split"],
            horizontal=True,
        )

        fig     = go.Figure()
        medians = {}
        avgs    = {}
        pvals   = {}
        color_map = {}

        if view_mode == "By price era":
            ref_grp = monthly[monthly["entry_era"] == "era_99"]
            for era in ERA_ORDER:
                grp = monthly[monthly["entry_era"] == era]
                m   = add_km_trace_to(fig, grp, ERA_COLORS[era], ERA_LABELS[era])
                if m is not None:
                    medians[era] = m
                    avgs[era]    = grp["km_duration"].mean()
                    color_map[era] = ERA_COLORS[era]
                    if era != "era_99":
                        res = logrank_test(
                            ref_grp["km_duration"], grp["km_duration"],
                            event_observed_A=ref_grp["km_event"],
                            event_observed_B=grp["km_event"],
                        )
                        pvals[era] = res.p_value
            chart_title = (
                "Higher prices, shorter tenure: "
                "99 PLN subscribers lasted twice as long"
            )
            ref_label   = "99 PLN"

        elif view_mode == "Full price vs promo (overall)":
            segs = {
                "Full price":    (monthly[~monthly["is_promo"]], ACCENT_COLOR),
                "Promo discount":(monthly[ monthly["is_promo"]], PROMO_COLOR),
            }
            ref_grp_km = segs["Full price"][0]
            color_map  = {s: c for s, (_, c) in segs.items()}
            for seg, (grp, color) in segs.items():
                m = add_km_trace_to(fig, grp, color, seg)
                if m is not None:
                    medians[seg] = m
                    avgs[seg]    = grp["km_duration"].mean()
                    if seg != "Full price":
                        res = logrank_test(
                            ref_grp_km["km_duration"], grp["km_duration"],
                            event_observed_A=ref_grp_km["km_event"],
                            event_observed_B=grp["km_event"],
                        )
                        pvals[seg] = res.p_value
            chart_title = (
                "Promo-code subscribers stay slightly longer, "
                "but generate less lifetime revenue"
            )
            ref_label   = "Full price"

        else:  # By era + promo split
            era_options = st.multiselect(
                "Filter eras",
                options=ERA_ORDER,
                default=ERA_ORDER,
                format_func=lambda e: ERA_LABELS[e],
            )
            ref_grp_era = monthly[
                (monthly["entry_era"]=="era_99") & (~monthly["is_promo"])
            ]
            for era in era_options:
                color = ERA_COLORS[era]
                r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                for is_promo, seg_label, dash in [
                    (False, "Full price", "solid"),
                    (True,  "Promo",      "dash"),
                ]:
                    grp = monthly[
                        (monthly["entry_era"]==era)
                        & (monthly["is_promo"]==is_promo)
                    ]
                    if len(grp) < 5: continue
                    kmf = KaplanMeierFitter()
                    kmf.fit(grp["km_duration"], event_observed=grp["km_event"])
                    t = kmf.survival_function_.index.tolist()
                    s = kmf.survival_function_.iloc[:, 0].tolist()
                    key = f"{era}_{seg_label}"
                    medians[key] = kmf.median_survival_time_
                    avgs[key]    = grp["km_duration"].mean()
                    color_map[key] = color
                    label = f"{ERA_LABELS[era]} · {seg_label}  (n={len(grp)})"
                    fig.add_trace(go.Scatter(
                        x=t, y=s, mode="lines",
                        name=label,
                        line=dict(color=color, width=2.5, dash=dash),
                        hovertemplate=(
                            f"<b>{label}</b>"
                            f"<br>Month %{{x:.0f}}: %{{y:.0%}}<extra></extra>"
                        ),
                    ))
                    if not is_promo and era != "era_99":
                        res = logrank_test(
                            ref_grp_era["km_duration"], grp["km_duration"],
                            event_observed_A=ref_grp_era["km_event"],
                            event_observed_B=grp["km_event"],
                        )
                        pvals[key] = res.p_value
            chart_title = (
                "Survival by era and pricing type "
                "(solid = full price, dashed = promo)"
            )
            ref_label   = "99 PLN full price"

        # median dotted lines
        for key, m in medians.items():
            if not pd.isna(m):
                c = color_map.get(key, NEUTRAL_COLOR)
                fig.add_shape(type="line", x0=m, x1=m, y0=0, y1=0.95,
                              line=dict(color=c, width=1.2, dash="dot"), opacity=0.6)

        median_parts = [
            f"{ERA_LABELS.get(k,k)}: {fmt_months(m)}"
            for k, m in medians.items() if view_mode != "By era + promo split"
        ]
        _subtitle = (
            "Median subscription length: " + " · ".join(median_parts)
            if median_parts else None
        )
        style_fig(
            fig,
            title=chart_title,
            subtitle=_subtitle,
            xlab="Months since sign-up",
            ylab="Probability of still subscribing",
            height=540,
        )
        fig.update_layout(
            yaxis=dict(tickformat=".0%", range=[0, 1.05]),
            xaxis=dict(range=[0, 15]),
            legend=dict(
                orientation="v", x=1.01, y=0.98,
                title=dict(
                    text="Group",
                    font=dict(size=11, color=TEXT_SECONDARY),
                ),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        if medians and view_mode != "By era + promo split":
            st.markdown("#### Statistical significance")
            rows = []
            for key, m in medians.items():
                p = pvals.get(key)
                rows.append({
                    "Group":         ERA_LABELS.get(key, key),
                    "Median tenure": fmt_months(m),
                    "Avg tenure":    fmt_months(avgs.get(key)),
                    f"vs {ref_label} (log-rank)": (
                        fmt_pval(p) if p is not None else "reference"
                    ),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown(
                "<div class='caption-note'>✓✓✓ = p < 0.001 · ✓✓ = p < 0.01 · "
                "✓ = p < 0.05 · "
                "— = not significant · log-rank test (non-parametric)</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div class='caption-note'>⚠ {PRELIMINARY_NOTE}</div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WHAT DID THEY GENERATE?
# ══════════════════════════════════════════════════════════════════════════════
elif page == "What did they generate?":
    st.title("What did they generate?")
    st.markdown(
        "Higher prices did not automatically mean higher revenue per subscriber. "
        "The 199 PLN era produced the most total revenue of any era, combining "
        "a strong price with the largest subscriber base. "
        "Promotional subscribers generate less lifetime revenue in every era, "
        "though the gap is not statistically significant. "
        "Cost of acquisition (CAC) data is not available, so promo ROI "
        "cannot be assessed from revenue alone."
    )
    st.markdown("---")

    # ── TOP: Monthly vs Long plans revenue banner ─────────────────────────────
    monthly_rev_total  = int(monthly["total_revenue"].sum())
    longplan_rev_total = int(annual["total_revenue"].sum())
    all_rev_total      = monthly_rev_total + longplan_rev_total
    cl1, cl2, cl3 = st.columns(3)
    cl1.metric("Total revenue, all plans",  f"{all_rev_total:,} PLN")
    cl2.metric(
        "Monthly plans",
        f"{monthly_rev_total:,} PLN",
        delta=f"{monthly_rev_total/all_rev_total*100:.0f}% of total",
    )
    cl3.metric(
        "Long plans",
        f"{longplan_rev_total:,} PLN",
        delta=f"{longplan_rev_total/all_rev_total*100:.0f}% of total",
    )
    st.markdown(
        "<div class='caption-note'>Long plans = annual / semi-annual "
        "(single payment > 700 PLN). "
        "Breakdown below covers monthly plans only.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── MONTHLY PLANS: revenue summary table with full / promo split ──────────
    st.markdown("### Monthly plans: revenue by price era")
    MILESTONE_OFFSETS = [1, 3, 6]
    summary_rows = []
    for era in ERA_ORDER:
        era_sub = monthly[monthly["entry_era"] == era]
        full_e  = era_sub[~era_sub["is_promo"]]
        promo_e = era_sub[ era_sub["is_promo"]]
        summary_rows.append({
            "Era":               ERA_LABELS[era],
            "Price":             f"{ERA_PRICES[era]} PLN",
            "Full price (n)":    len(full_e),
            "Promo (n)":         len(promo_e),
            "% promo":           (
                fmt_pct(len(promo_e) / len(era_sub) * 100)
                if len(era_sub) else "—"
            ),
            "Median LTV (full)": (
                fmt_pln(full_e["total_revenue"].median())
                if len(full_e) > 0 else "—"
            ),
            "Median LTV (promo)": (
                fmt_pln(promo_e["total_revenue"].median())
                if len(promo_e) > 0 else "—"
            ),
            "Total revenue":     fmt_pln(era_sub["total_revenue"].sum()),
            "Active":            fmt_pct(era_sub["is_active"].mean() * 100),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.markdown(
        "<div class='caption-note'>⚠ 249 PLN era: 7 months of data only. "
        "Revenue totals will grow.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    col1, col2 = st.columns([3, 2])

    with col1:
        # Grouped bar: Median LTV per era, full price vs promo
        fig = go.Figure()
        ltv_full  = []
        ltv_promo = []
        era_labels_list = []
        for era in ERA_ORDER:
            full_e  = monthly[(monthly["entry_era"]==era) & ~monthly["is_promo"]]
            promo_e = monthly[(monthly["entry_era"]==era) &  monthly["is_promo"]]
            if len(full_e) < 5 or len(promo_e) < 5:
                continue
            era_labels_list.append(ERA_LABELS[era])
            ltv_full.append(full_e["total_revenue"].median())
            ltv_promo.append(promo_e["total_revenue"].median())

        fig.add_trace(go.Bar(
            x=era_labels_list, y=ltv_full,
            name="Full price",
            marker_color=ACCENT_COLOR, marker_line=dict(width=0),
            text=[fmt_pln(v) for v in ltv_full], textposition="outside",
            textfont=dict(size=11, color=TEXT_SECONDARY),
            hovertemplate=(
                "<b>Full price</b><br>%{x}: %{y:,.0f} PLN median LTV<extra></extra>"
            ),
        ))
        fig.add_trace(go.Bar(
            x=era_labels_list, y=ltv_promo,
            name="Promo discount",
            marker_color=PROMO_COLOR, marker_line=dict(width=0),
            text=[fmt_pln(v) for v in ltv_promo], textposition="outside",
            textfont=dict(size=11, color=TEXT_SECONDARY),
            hovertemplate=(
                "<b>Promo discount</b>"
                "<br>%{x}: %{y:,.0f} PLN median LTV<extra></extra>"
            ),
        ))
        style_fig(
            fig,
            title="Median lifetime revenue: full price vs promo per era",
            subtitle=(
                "Promo LTV is comparable to full price, "
                "and higher in eras 199 and 249"
            ),
            xlab="Price era", ylab="Median LTV per subscriber (PLN)", height=440,
        )
        fig.update_layout(
            barmode="group",
            legend=dict(orientation="h", y=-0.22, x=0),
            yaxis=dict(range=[0, max(ltv_full + ltv_promo) * 1.25]),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rev_vals = [
            monthly[monthly["entry_era"]==e]["total_revenue"].sum()
            for e in ERA_ORDER
        ]
        fig2 = go.Figure()
        for era in ERA_ORDER:
            era_sub  = monthly[monthly["entry_era"]==era]
            full_rev  = era_sub[~era_sub["is_promo"]]["total_revenue"].sum()
            promo_rev = era_sub[ era_sub["is_promo"]]["total_revenue"].sum()
            color = ERA_COLORS[era]
            r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
            fig2.add_trace(go.Bar(
                x=[ERA_LABELS[era]], y=[full_rev],
                name=f"{ERA_LABELS[era]} (full)",
                marker_color=color, marker_line=dict(width=0),
                legendgroup=era, showlegend=True,
                hovertemplate=(
                    f"<b>{ERA_LABELS[era]} full price</b>"
                    f"<br>%{{y:,.0f}} PLN<extra></extra>"
                ),
            ))
            fig2.add_trace(go.Bar(
                x=[ERA_LABELS[era]], y=[promo_rev],
                name=f"{ERA_LABELS[era]} (promo)",
                marker_color=f"rgba({r},{g},{b},0.45)", marker_line=dict(width=0),
                legendgroup=era, showlegend=True,
                hovertemplate=(
                    f"<b>{ERA_LABELS[era]} promo</b>"
                    f"<br>%{{y:,.0f}} PLN<extra></extra>"
                ),
            ))
        style_fig(fig2,
                  title="Total revenue by era: full price vs promo",
                  subtitle="Solid = full price · Transparent = promo (same era color)",
                  ylab="Total revenue (PLN)", height=380)
        fig2.update_layout(
            barmode="stack", showlegend=False,
            yaxis=dict(range=[0, max(rev_vals) * 1.2]),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── PROMO ANALYSIS: detailed retention breakdown ───────────────────────────
    st.markdown("### Promo vs full price: retention breakdown")
    st.markdown(
        "Promo subscribers generate **comparable lifetime revenue**, "
        "but drop off faster in the first month, "
        "especially in earlier eras."
    )

    PROMO_MILESTONES = [1, 3, 6, 12]
    promo_table = []
    for era in ERA_ORDER:
        for seg_label, mask in [("Full price", ~monthly["is_promo"]),
                                 ("Promo discount", monthly["is_promo"])]:
            sub = monthly[(monthly["entry_era"]==era) & mask]
            if len(sub) < 5: continue
            ret = retention_at_offset(sub, df_raw, PROMO_MILESTONES)
            promo_table.append({
                "Era":            ERA_LABELS[era],
                "Segment":        seg_label,
                "n":              len(sub),
                "Median entry price": fmt_pln(sub["entry_price"].median()),
                "Avg entry price":   fmt_pln(sub["entry_price"].mean()),
                "Median LTV":        fmt_pln(sub["total_revenue"].median()),
                "Avg LTV":           fmt_pln(sub["total_revenue"].mean()),
                "M+1":            fmt_pct(ret[1]),
                "M+3":            fmt_pct(ret[3]),
                "M+6":            fmt_pct(ret[6]) if era != "era_249" else "N/A",
                "M+12":           (
                    fmt_pct(ret[12]) if era not in ("era_249",) else "N/A"
                ),
            })

    st.dataframe(pd.DataFrame(promo_table), use_container_width=True, hide_index=True)
    st.markdown(
        "<div class='caption-note'>Entry price = median first payment · "
        "M+6 and M+12 for 249 PLN era are N/A (insufficient observation window). "
        "⚠ CAC data not available; full promo ROI assessment requires "
        "acquisition cost data.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        # Retention bars: M+1 and M+3, full vs promo, per era
        fig3 = go.Figure()
        SEG_COLORS = {"Full price": ACCENT_COLOR, "Promo discount": PROMO_COLOR}
        for mo, opacity in [(1, 1.0), (3, 0.55)]:
            for seg_label, mask in [("Full price", ~monthly["is_promo"]),
                                     ("Promo discount", monthly["is_promo"])]:
                xvals, yvals = [], []
                for era in ERA_ORDER:
                    sub = monthly[(monthly["entry_era"]==era) & mask]
                    if len(sub) < 5: continue
                    ret = retention_at_offset(sub, df_raw, [mo])
                    xvals.append(ERA_LABELS[era])
                    yvals.append(ret[mo])
                color = SEG_COLORS[seg_label]
                r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                fig3.add_trace(go.Bar(
                    x=xvals, y=yvals,
                    name=f"{seg_label} · M+{mo}",
                    marker_color=f"rgba({r},{g},{b},{opacity})",
                    marker_line=dict(width=0),
                    text=[f"{v:.0f}%" for v in yvals], textposition="outside",
                    textfont=dict(size=10, color=TEXT_SECONDARY),
                    hovertemplate=(
                        f"<b>{seg_label} · M+{mo}</b>"
                        f"<br>%{{x}}: %{{y:.0f}}%<extra></extra>"
                    ),
                ))
        style_fig(
            fig3,
            title="Early retention: full price vs promo",
            subtitle=(
                "Solid bars = M+1 · Transparent = M+3 · "
                "Promo drops faster at M+1 in eras 99 and 169"
            ),
            xlab="Price era", ylab="Still subscribed (%)", height=420,
        )
        fig3.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 115]),
            legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=10)),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Revenue split: promo vs full, overall
        rev_data = {
            "Full price":    int(monthly[~monthly["is_promo"]]["total_revenue"].sum()),
            "Promo discount":int(monthly[ monthly["is_promo"]]["total_revenue"].sum()),
        }
        n_data = {
            "Full price":    len(monthly[~monthly["is_promo"]]),
            "Promo discount":len(monthly[ monthly["is_promo"]]),
        }
        fig4 = make_subplots(rows=1, cols=2,
                             subplot_titles=["Total revenue", "Subscribers"],
                             horizontal_spacing=0.18)
        seg_colors_list = [ACCENT_COLOR, PROMO_COLOR]
        for i, (seg, color) in enumerate(zip(rev_data.keys(), seg_colors_list)):
            fig4.add_trace(go.Bar(
                x=[seg], y=[rev_data[seg]],
                marker_color=color, marker_line=dict(width=0),
                text=[fmt_pln(rev_data[seg])], textposition="outside",
                textfont=dict(size=11, color=TEXT_SECONDARY),
                hovertemplate=f"<b>{seg}</b><br>%{{y:,.0f}} PLN<extra></extra>",
                showlegend=False,
            ), row=1, col=1)
            fig4.add_trace(go.Bar(
                x=[seg], y=[n_data[seg]],
                marker_color=color, marker_line=dict(width=0),
                text=[str(n_data[seg])], textposition="outside",
                textfont=dict(size=11, color=TEXT_SECONDARY),
                hovertemplate=f"<b>{seg}</b><br>%{{y}} subscribers<extra></extra>",
                showlegend=False,
            ), row=1, col=2)
        style_fig(
            fig4,
            title="Full price vs promo: revenue and headcount",
            subtitle=(
                "Despite nearly equal subscriber counts, "
                "revenue split is 54% / 46%"
            ),
            height=380,
        )
        fig4.update_layout(showlegend=False)
        for ann in fig4.layout.annotations:
            if ann.text in ["Total revenue", "Subscribers"]:
                ann.font = dict(size=12, color=TEXT_SECONDARY, family=FONT_FAMILY)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown(
            "<div class='caption-note'>⚠ CAC data not available. "
            "Promo ROI cannot be fully assessed from revenue alone.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Long plans: revenue summary")
    st.markdown(
        "Long-plan subscribers (annual / semi-annual) generated **"
        f"{int(annual['total_revenue'].sum()):,} PLN** in total revenue across "
        f"{len(annual)} subscribers. Renewal rates below are based only on "
        "subscribers "
        "whose 12-month window had closed before Mar 2026."
    )
    CUTOFF_PERIOD_LTV = pd.Period("2026-03", freq="M")
    annual_lp = annual.copy()
    annual_lp["renewal_month"] = annual_lp["entry_month"] + 12
    observable_lp = annual_lp[annual_lp["renewal_month"] <= CUTOFF_PERIOD_LTV]
    lp_rev_rows = []
    for era in ERA_ORDER:
        sub_all = annual_lp[annual_lp["entry_era"] == era]
        sub_obs = observable_lp[observable_lp["entry_era"] == era]
        n_obs   = len(sub_obs)
        renewed = int(sub_obs["is_active"].sum()) if n_obs > 0 else 0
        lp_rev_rows.append({
            "Era":                ERA_LABELS[era],
            "Total (n)":          len(sub_all),
            "Total revenue":      fmt_pln(sub_all["total_revenue"].sum()),
            "Median LTV":         fmt_pln(sub_all["total_revenue"].median()),
            "Avg LTV":            fmt_pln(sub_all["total_revenue"].mean()),
            "Median entry price": fmt_pln(sub_all["entry_price"].median()),
            "Avg entry price":    fmt_pln(sub_all["entry_price"].mean()),
            "Observable (n)":     n_obs if n_obs > 0 else "—",
            "Renewal rate":       (
                fmt_pct(renewed / n_obs * 100) if n_obs > 0 else "N/A"
            ),
            "M+12 retention":     (
                fmt_pct(renewed / n_obs * 100) if n_obs > 0 else "N/A"
            ),
        })
    st.dataframe(pd.DataFrame(lp_rev_rows), use_container_width=True, hide_index=True)
    st.markdown(
        "<div class='caption-note'>"
        "era_249 long plans: renewal window not yet reached (opens Sep 2026+). "
        "No promo subscribers in long plans. "
        "M+12 retention = renewal rate (single decision point per subscriber)."
        "</div>", unsafe_allow_html=True,
    )
