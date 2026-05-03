import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shared_style import (
    ERA_COLORS, ERA_ORDER, ERA_LABELS, ERA_LABELS_FULL,
    PROMO_COLOR, ANNUAL_COLOR, NEUTRAL_COLOR, ACCENT_COLOR,
    BG_CHART, GRID, BORDER, TEXT_PRIMARY, TEXT_SECONDARY,
    FONT_FAMILY, SOURCE_NOTE,
    style_fig, add_era_vlines, save_fig, era_for_month,
    hex_rgba, fmt_pln, fmt_pct, fmt_period, PRICE_ERAS,
)

RAW_PATH = "data/KDS Transactions.xlsx"
OUT_DIR = "outputs/01_eda"

print("Loading data...")
df = pd.read_excel(RAW_PATH)
df.columns = ["date", "client", "amount"]
df["month"] = df["date"].dt.to_period("M")
df["month_ts"] = df["month"].dt.to_timestamp()

first_tx = df.groupby("client")["date"].min().rename("entry_date")
df = df.merge(first_tx, on="client")
df["entry_month"] = df["entry_date"].dt.to_period("M")
df["entry_era"] = df["entry_month"].apply(era_for_month)

print("Generating EDA charts...\n")

print("[1/6] Payment amount distribution")

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=df["amount"],
    nbinsx=60,
    marker_color=ACCENT_COLOR,
    marker_line=dict(width=0),
    opacity=0.85,
    name="Payments",
    hovertemplate="Amount: %{x:.0f} PLN<br>Count: %{y}<extra></extra>",
))

for era in ERA_ORDER:
    price = next(p for l, _, _, p in PRICE_ERAS if l == era)
    fig.add_vline(
        x=price,
        line_dash="dot",
        line_color=ERA_COLORS[era],
        line_width=2,
        annotation_text=f"{int(price)} PLN tier",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color=ERA_COLORS[era],
    )

style_fig(
    fig,
    title=(
        "Most subscribers pay 99, 169, or 199 PLN"
        " — three dominant price tiers"
    ),
    subtitle=(
        "Amounts above 700 PLN are annual bundles"
        " (one payment covers ~10–12 months)"
    ),
    xlab="Monthly payment (PLN)",
    ylab="Number of payments",
    height=480,
)
fig.update_layout(showlegend=False)
save_fig(fig, OUT_DIR, "01_amount_distribution")

print("[2/6] Monthly volume & revenue")

monthly = df.groupby("month_ts").agg(
    n_tx=("amount", "count"),
    revenue=("amount", "sum"),
).reset_index()

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "Subscriptions processed",
        "Revenue collected (PLN)",
    ],
    vertical_spacing=0.10,
)
fig.add_trace(go.Bar(
    x=monthly["month_ts"], y=monthly["n_tx"],
    marker_color=ACCENT_COLOR, marker_line=dict(width=0),
    opacity=0.85, name="Subscriptions",
    hovertemplate="%{x|%b '%y}: %{y} payments<extra></extra>",
), row=1, col=1)
fig.add_trace(go.Bar(
    x=monthly["month_ts"], y=monthly["revenue"],
    marker_color=ERA_COLORS["era_199"], marker_line=dict(width=0),
    opacity=0.85, name="Revenue",
    hovertemplate="%{x|%b '%y}: %{y:,.0f} PLN<extra></extra>",
), row=2, col=1)

add_era_vlines(fig, row=1, col=1)
add_era_vlines(fig, row=2, col=1)

style_fig(
    fig,
    title=(
        "Revenue grew with each price increase"
        " — despite stable or declining transaction counts"
    ),
    subtitle="Dashed lines mark price transitions",
    height=620,
)
fig.update_layout(showlegend=False)
subplot_titles = {"Subscriptions processed", "Revenue collected (PLN)"}
for ann in fig.layout.annotations:
    if ann.text in subplot_titles:
        ann.font = dict(size=12, color=TEXT_SECONDARY, family=FONT_FAMILY)
save_fig(fig, OUT_DIR, "02_monthly_volume_revenue")

print("[3/6] New subscriber acquisition by era")

first_df = (
    df.drop_duplicates("client")[["client", "entry_month", "entry_era"]]
    .copy()
)
first_df["entry_ts"] = first_df["entry_month"].dt.to_timestamp()
acq = (
    first_df.groupby(["entry_ts", "entry_era"])
    .size()
    .reset_index(name="new_customers")
)

fig = go.Figure()
for era in ERA_ORDER:
    sub = acq[acq["entry_era"] == era]
    hover = (
        f"{ERA_LABELS[era]}"
        "<br>%{x|%b '%y}: %{y} new subscribers<extra></extra>"
    )
    fig.add_trace(go.Bar(
        x=sub["entry_ts"],
        y=sub["new_customers"],
        name=ERA_LABELS_FULL[era],
        marker_color=ERA_COLORS[era],
        marker_line=dict(width=0),
        hovertemplate=hover,
    ))

fig.add_annotation(
    x=pd.Timestamp("2024-07-15").timestamp() * 1000,
    y=8,
    text="Sales gap<br>Jul–Aug '24",
    showarrow=True,
    arrowhead=2,
    arrowcolor=NEUTRAL_COLOR,
    arrowwidth=1.2,
    font=dict(size=10, color=NEUTRAL_COLOR, family=FONT_FAMILY),
    align="center",
    bgcolor=BG_CHART,
    bordercolor=BORDER,
    borderwidth=1,
)

style_fig(
    fig,
    title="New subscriber growth by price era",
    subtitle=(
        "September spikes in 2024 and 2025 suggest seasonal"
        " or campaign-driven demand, independent of price"
    ),
    xlab="Month",
    ylab="New subscribers",
    height=500,
)
fig.update_layout(
    barmode="stack",
    legend=dict(
        title=dict(
            text="Price era",
            font=dict(size=11, color=TEXT_SECONDARY),
        ),
        orientation="h", y=-0.22, x=0,
    ),
)
save_fig(fig, OUT_DIR, "03_new_customers_by_era")

print("[4/6] Transactions per subscriber")

tx_per = df.groupby("client").size().reset_index(name="n_tx")
median_val = int(tx_per["n_tx"].median())

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=tx_per["n_tx"],
    nbinsx=29,
    marker_color=ERA_COLORS["era_249"],
    marker_line=dict(width=0),
    opacity=0.85,
    hovertemplate="Paid %{x} time(s): %{y} subscribers<extra></extra>",
))
fig.add_vline(
    x=median_val,
    line_dash="dash",
    line_color=ACCENT_COLOR,
    line_width=2,
    annotation_text=(
        f"Median — half of all subscribers<br>"
        f"make ≤{median_val} payments"
    ),
    annotation_position="top right",
    annotation_font_size=10,
    annotation_font_color=ACCENT_COLOR,
)

one_payment_pct = fmt_pct((tx_per["n_tx"] == 1).mean() * 100)
style_fig(
    fig,
    title=(
        "Most subscribers cancel within 2 payments"
        " — early churn is the central challenge"
    ),
    subtitle=(
        f"{one_payment_pct} of subscribers make only"
        " a single payment before cancelling"
    ),
    xlab="Payments made before cancelling",
    ylab="Subscribers",
    height=480,
)
fig.update_layout(showlegend=False)
save_fig(fig, OUT_DIR, "04_transactions_per_customer")

print("[5/6] Payment amount clustering")

unique_amounts = df["amount"].value_counts().reset_index()
unique_amounts.columns = ["amount", "freq"]


def classify_amount(amt):
    if amt > 700:
        return "Annual bundle"
    for label, _, _, price in PRICE_ERAS:
        if abs(amt - price) / price <= 0.05:
            return f"{int(price)} PLN — full price"
    return "Discounted / promo rate"


unique_amounts["tier"] = unique_amounts["amount"].apply(classify_amount)

tier_color_map = {
    "99 PLN — full price":     ERA_COLORS["era_99"],
    "169 PLN — full price":    ERA_COLORS["era_169"],
    "199 PLN — full price":    ERA_COLORS["era_199"],
    "249 PLN — full price":    ERA_COLORS["era_249"],
    "Annual bundle":           ANNUAL_COLOR,
    "Discounted / promo rate": PROMO_COLOR,
}

fig = go.Figure()
for tier, grp in unique_amounts.groupby("tier"):
    hover = (
        f"{tier}<br>"
        "Amount: %{x:.2f} PLN<br>"
        "Frequency: %{y}<extra></extra>"
    )
    fig.add_trace(go.Scatter(
        x=grp["amount"], y=grp["freq"],
        mode="markers",
        marker=dict(
            size=8,
            color=tier_color_map.get(tier, NEUTRAL_COLOR),
            opacity=0.85,
        ),
        name=tier,
        hovertemplate=hover,
    ))

for era in ERA_ORDER:
    price = next(p for l, _, _, p in PRICE_ERAS if l == era)
    fig.add_vline(
        x=price, line_dash="dot",
        line_color=ERA_COLORS[era],
        line_width=1.5, opacity=0.6,
    )

fig.add_vline(
    x=700, line_dash="dot",
    line_color=ANNUAL_COLOR, line_width=1.5, opacity=0.6,
    annotation_text="Annual plans above this line",
    annotation_position="top left",
    annotation_font_size=10,
    annotation_font_color=ANNUAL_COLOR,
)

style_fig(
    fig,
    title=(
        "Payment amounts cluster tightly around official prices"
        " — promo discounts are clearly visible"
    ),
    subtitle=(
        "Each dot is a unique payment amount;"
        " height shows how often it appears"
    ),
    xlab="Payment amount (PLN)",
    ylab="How often this amount appears",
    height=500,
)
fig.update_layout(
    legend=dict(
        title=dict(
            text="Payment type",
            font=dict(size=11, color=TEXT_SECONDARY),
        ),
        orientation="h", y=-0.22, x=0,
    ),
)
save_fig(fig, OUT_DIR, "05_amount_clustering")

print("[6/6] Cumulative subscriber growth")

acq_monthly = first_df.groupby("entry_ts").size().reset_index(name="new")
acq_monthly = acq_monthly.sort_values("entry_ts")
acq_monthly["cumulative"] = acq_monthly["new"].cumsum()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=acq_monthly["entry_ts"],
    y=acq_monthly["cumulative"],
    mode="lines",
    line=dict(color=ACCENT_COLOR, width=2.5),
    fill="tozeroy",
    fillcolor=hex_rgba(ACCENT_COLOR, 0.08),
    hovertemplate="%{x|%b '%y}: %{y} total subscribers<extra></extra>",
    name="Cumulative subscribers",
))

add_era_vlines(fig)

final_count = int(acq_monthly["cumulative"].iloc[-1])
style_fig(
    fig,
    title=(
        f"KDS reached {final_count:,} subscribers over 29 months"
        " — growth accelerated in each new price era"
    ),
    subtitle=(
        "Growth did not stall after price increases"
        " — each era attracted more subscribers than the previous"
    ),
    xlab="Month",
    ylab="Total subscribers",
    height=460,
)
fig.update_layout(showlegend=False)
save_fig(fig, OUT_DIR, "06_cumulative_growth")

print("[7/7] Retention milestones by era")

cust = pd.read_csv("data/df_customers.csv", parse_dates=["entry_date", "last_payment"])
cust["entry_month"] = cust["entry_month"].apply(lambda x: pd.Period(x, "M"))
monthly_cust = cust[~cust["is_annual"]].copy()

MILESTONES = [1, 3, 6, 12]
ret_rows = []
for era in ERA_ORDER:
    grp = monthly_cust[monthly_cust["entry_era"] == era]
    ids = set(grp["client_id"])
    tx_sub = df[df["client"].isin(ids)].copy()
    cohort_map = grp.set_index("client_id")["entry_month"].to_dict()
    tx_sub["cohort"] = tx_sub["client"].map(cohort_map)
    tx_sub["offset"] = (
        tx_sub["month"] - tx_sub["cohort"]
    ).apply(lambda x: x.n)
    total = len(grp)
    for mo in MILESTONES:
        active = tx_sub[tx_sub["offset"] == mo]["client"].nunique()
        ret_rows.append({
            "era":   era,
            "label": ERA_LABELS[era],
            "month": f"M+{mo}",
            "pct":   round(active / total * 100, 1) if total > 0 else 0,
        })

ret_df = pd.DataFrame(ret_rows)

fig = go.Figure()
for era in ERA_ORDER:
    sub = ret_df[ret_df["era"] == era]
    color = ERA_COLORS[era]
    hover = (
        f"<b>{ERA_LABELS[era]}</b><br>"
        "%{x}: %{y:.0f}% still paying<extra></extra>"
    )
    fig.add_trace(go.Bar(
        x=sub["month"],
        y=sub["pct"],
        name=ERA_LABELS[era],
        marker_color=color,
        marker_line=dict(width=0),
        text=[f"{v:.0f}%" for v in sub["pct"]],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_SECONDARY),
        hovertemplate=hover,
    ))

style_fig(
    fig,
    title=(
        "Retention drops sharply after month 1"
        " — and collapses after month 6 across all price eras"
    ),
    subtitle=(
        "Monthly plans only · each bar = share of that era's"
        " subscribers still paying at that milestone"
    ),
    xlab="Months since sign-up",
    ylab="Still subscribed (%)",
    height=500,
)
fig.update_layout(
    barmode="group",
    yaxis=dict(range=[0, 115]),
    legend=dict(
        title=dict(
            text="Price era",
            font=dict(size=11, color=TEXT_SECONDARY),
        ),
        orientation="h", y=-0.18, x=0,
    ),
)
save_fig(fig, OUT_DIR, "07_retention_milestones")

print("\nEDA complete. All charts saved to outputs/01_eda/")
