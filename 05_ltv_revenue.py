import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu
from shared_style import (
    ERA_COLORS, ERA_ORDER, ERA_LABELS, ERA_PRICES,
    PROMO_COLOR, ACCENT_COLOR, NEUTRAL_COLOR, POSITIVE_COLOR,
    BG_CHART, GRID, BORDER, TEXT_PRIMARY, TEXT_SECONDARY,
    FONT_FAMILY,
    style_fig, save_fig, hex_rgba, fmt_pln, fmt_pct, fmt_pval,
)

CUSTOMERS_PATH = "data/df_customers.csv"
OUT_DIR        = "outputs/05_ltv"

print("Loading data...")
cust = pd.read_csv(CUSTOMERS_PATH, parse_dates=["entry_date", "last_payment"])
cust["entry_month"] = cust["entry_month"].apply(lambda x: pd.Period(x, "M"))
monthly = cust[~cust["is_annual"]].copy()


def mw_pval(a, b):
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    return p


print("\n[1/4] LTV distribution by era (monthly plans)")

fig = go.Figure()
for era in ERA_ORDER:
    sub   = monthly[monthly["entry_era"] == era]["total_revenue"]
    color = ERA_COLORS[era]
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fig.add_trace(go.Violin(
        y=sub,
        name=ERA_LABELS[era],
        box_visible=True,
        meanline_visible=True,
        points="outliers",
        line_color=color,
        fillcolor=f"rgba({r},{g},{b},0.20)",
        marker=dict(size=3, color=color, opacity=0.6),
            hovertemplate=(
            f"<b>{ERA_LABELS[era]}</b><br>Revenue: %{{y:,.0f}} PLN<extra></extra>"
        ),
    ))

ref = monthly[monthly["entry_era"] == "era_99"]["total_revenue"]
sig_parts = []
for era in ERA_ORDER[1:]:
    grp = monthly[monthly["entry_era"] == era]["total_revenue"]
    p   = mw_pval(ref, grp)
    sig_parts.append(f"{ERA_LABELS[era]} vs 99 PLN: {fmt_pval(p)}")

style_fig(
    fig,
    title=(
        "Higher-priced eras don't always mean higher lifetime revenue"
        " — 169 PLN leads on median"
    ),
    subtitle=(
        "Box shows median & IQR · Violin shows full distribution"
        " · Dots are individual subscribers"
    ),
    ylab="Lifetime revenue per subscriber (PLN)",
    height=520,
)
fig.update_layout(
    showlegend=False,
    xaxis=dict(tickfont=dict(size=12, color=TEXT_SECONDARY)),
    annotations=list(fig.layout.annotations) + [
        dict(
            text="Mann-Whitney U test — " + "  |  ".join(sig_parts),
            xref="paper", yref="paper",
            x=0, y=-0.12,
            showarrow=False,
            font=dict(size=10, color=TEXT_SECONDARY, family=FONT_FAMILY),
            xanchor="left",
        ),
    ],
)
save_fig(fig, OUT_DIR, "01_ltv_violin_monthly")

print("[2/4] Revenue decomposition by era")

era_stats = []
for era in ERA_ORDER:
    sub = monthly[monthly["entry_era"] == era]
    era_stats.append({
        "era":        era,
        "label":      ERA_LABELS[era],
        "price":      ERA_PRICES[era],
        "n":          len(sub),
        "median_ltv": sub["total_revenue"].median(),
        "mean_ltv":   sub["total_revenue"].mean(),
        "total_rev":  sub["total_revenue"].sum(),
        "active_pct": sub["is_active"].mean() * 100,
    })
stats_df = pd.DataFrame(era_stats)
stats_df.to_csv(f"{OUT_DIR}/ltv_stats_table.csv", index=False)

print("\n  --- Lifetime Revenue by Era (monthly plans) ---")
for _, row in stats_df.iterrows():
    print(
        f"  {row['label']:10s}  n={row['n']:3d}  "
        f"median={fmt_pln(row['median_ltv']):15s}  "
        f"total={fmt_pln(row['total_rev'])}"
    )

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "Median lifetime revenue per subscriber",
        "Total revenue generated per price era",
    ],
    horizontal_spacing=0.14,
)
fig.add_trace(go.Bar(
    x=stats_df["label"], y=stats_df["median_ltv"],
    marker_color=[ERA_COLORS[e] for e in stats_df["era"]],
    marker_line=dict(width=0),
    text=[fmt_pln(v) for v in stats_df["median_ltv"]],
    textposition="outside",
    textfont=dict(size=11, color=TEXT_SECONDARY),
    hovertemplate="%{x}<br>Median LTV: %{y:,.0f} PLN<extra></extra>",
    name="Median LTV",
), row=1, col=1)
fig.add_trace(go.Bar(
    x=stats_df["label"], y=stats_df["total_rev"],
    marker_color=[ERA_COLORS[e] for e in stats_df["era"]],
    marker_line=dict(width=0),
    text=[fmt_pln(v) for v in stats_df["total_rev"]],
    textposition="outside",
    textfont=dict(size=11, color=TEXT_SECONDARY),
    hovertemplate="%{x}<br>Total revenue: %{y:,.0f} PLN<extra></extra>",
    name="Total revenue",
), row=1, col=2)

style_fig(
    fig,
    title=(
        "The 199 PLN era generated the most total revenue"
        " — the best balance of price and volume"
    ),
    subtitle="Note: 249 PLN era covers only 7 months — totals will continue to grow",
    height=520,
)
fig.update_layout(
    showlegend=False,
    annotations=list(fig.layout.annotations) + [
        dict(
            text="⚠ 249 PLN era — 7 months of data only",
            xref="paper", yref="paper",
            x=1, y=-0.12,
            showarrow=False,
            font=dict(size=10, color=ERA_COLORS["era_249"], family=FONT_FAMILY),
            xanchor="right",
        ),
    ],
)
for ann in fig.layout.annotations:
    if ann.text in [
        "Median lifetime revenue per subscriber",
        "Total revenue generated per price era",
    ]:
        ann.font = dict(size=12, color=TEXT_SECONDARY, family=FONT_FAMILY)
save_fig(fig, OUT_DIR, "02_revenue_decomposition")

print("[3/4] Promo vs full-price LTV by era")

promo_rows = []
for era in ERA_ORDER:
    for seg_label, is_promo in [("Full price", False), ("Promo discount", True)]:
        mask = (monthly["entry_era"] == era) & (monthly["is_promo"] == is_promo)
        sub = monthly[mask]
        if len(sub) < 5:
            continue
        promo_rows.append({
            "era":        era,
            "era_label":  ERA_LABELS[era],
            "segment":    seg_label,
            "n":          len(sub),
            "median_ltv": sub["total_revenue"].median(),
        })

promo_df = pd.DataFrame(promo_rows)
SEG_COLORS = {"Full price": ACCENT_COLOR, "Promo discount": PROMO_COLOR}

fig = go.Figure()
for seg in ["Full price", "Promo discount"]:
    sub = promo_df[promo_df["segment"] == seg]
    fig.add_trace(go.Bar(
        x=sub["era_label"], y=sub["median_ltv"],
        name=seg,
        marker_color=SEG_COLORS[seg],
        marker_line=dict(width=0),
        text=[fmt_pln(v) for v in sub["median_ltv"]],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_SECONDARY),
        hovertemplate=(
            f"<b>{seg}</b><br>%{{x}}: %{{y:,.0f}} PLN median<extra></extra>"
        ),
    ))

style_fig(
    fig,
    title="Promo subscribers generate less total revenue, despite staying slightly longer",
    subtitle="Lower entry price is not recovered through extended tenure in any era",
    xlab="Price era",
    ylab="Median lifetime revenue per subscriber (PLN)",
    height=480,
)
fig.update_layout(
    barmode="group",
    legend=dict(orientation="h", y=-0.18, x=0),
)
save_fig(fig, OUT_DIR, "03_ltv_promo_vs_standard")

print("[4/4] Subscriber acquisition by era")

era_months_map = {"era_99": 10, "era_169": 2, "era_199": 10, "era_249": 7}
acq_rows = []
for era in ERA_ORDER:
    n  = len(monthly[monthly["entry_era"] == era])
    mo = era_months_map[era]
    acq_rows.append({
        "era":       era,
        "label":     ERA_LABELS[era],
        "n":         n,
        "months":    mo,
        "avg_month": round(n / mo, 1),
    })
acq_df = pd.DataFrame(acq_rows)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "Total subscribers per price era",
        "Average new subscribers per month",
    ],
    horizontal_spacing=0.14,
)
fig.add_trace(go.Bar(
    x=acq_df["label"], y=acq_df["n"],
    marker_color=[ERA_COLORS[e] for e in acq_df["era"]],
    marker_line=dict(width=0),
    text=acq_df["n"], textposition="outside",
    textfont=dict(size=11, color=TEXT_SECONDARY),
    hovertemplate="%{x}: %{y} subscribers total<extra></extra>",
    name="Total",
), row=1, col=1)
fig.add_trace(go.Bar(
    x=acq_df["label"], y=acq_df["avg_month"],
    marker_color=[ERA_COLORS[e] for e in acq_df["era"]],
    marker_line=dict(width=0),
    text=[f"{v:.1f}" for v in acq_df["avg_month"]],
    textposition="outside",
    textfont=dict(size=11, color=TEXT_SECONDARY),
    hovertemplate="%{x}: %{y:.1f} new subscribers/month<extra></extra>",
    name="Avg/month",
), row=1, col=2)

style_fig(
    fig,
    title=(
        "Acquisition per month accelerated with each price era"
        " — price did not suppress demand"
    ),
    subtitle="Use monthly average for fair comparison — eras have different durations",
    height=500,
)
fig.update_layout(
    showlegend=False,
    annotations=list(fig.layout.annotations) + [
        dict(
            text="169 PLN era = 2 months · 249 PLN era = 7 months · all others = 10 months",
            xref="paper", yref="paper",
            x=0, y=-0.12,
            showarrow=False,
            font=dict(size=10, color=NEUTRAL_COLOR, family=FONT_FAMILY),
            xanchor="left",
        ),
    ],
)
for ann in fig.layout.annotations:
    if ann.text in [
        "Total subscribers per price era",
        "Average new subscribers per month",
    ]:
        ann.font = dict(size=12, color=TEXT_SECONDARY, family=FONT_FAMILY)
save_fig(fig, OUT_DIR, "04_acquisition_by_era")

print("\nLTV & revenue analysis complete. Charts saved to outputs/05_ltv/")
