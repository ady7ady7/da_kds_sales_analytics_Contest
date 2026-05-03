import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from shared_style import (
    ERA_COLORS, ERA_ORDER, ERA_LABELS, ERA_LABELS_FULL,
    PROMO_COLOR, ACCENT_COLOR, NEUTRAL_COLOR,
    BG_CHART, BORDER, TEXT_PRIMARY, TEXT_SECONDARY,
    FONT_FAMILY,
    style_fig, save_fig, hex_rgba, fmt_months, fmt_pval,
)

CUSTOMERS_PATH = "data/df_customers.csv"
OUT_DIR        = "outputs/04_survival"

print("Loading data...")
cust = pd.read_csv(CUSTOMERS_PATH, parse_dates=["entry_date", "last_payment"])
cust["entry_month"] = cust["entry_month"].apply(lambda x: pd.Period(x, "M"))
monthly = cust[~cust["is_annual"]].copy()
annual  = cust[cust["is_annual"]].copy()


def fit_km_traces(df_subset, group_col, color_map, label_map, show_ci=True):
    """Fit KM curves per group. Returns (traces list, medians dict)."""
    traces  = []
    medians = {}

    ordered = [g for g in ERA_ORDER if g in df_subset[group_col].unique()]
    extras = [g for g in df_subset[group_col].unique() if g not in ERA_ORDER]
    for group in ordered + extras:
        grp = df_subset[df_subset[group_col] == group]
        if len(grp) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(grp["km_duration"], event_observed=grp["km_event"])
        color = color_map.get(group, NEUTRAL_COLOR)
        n     = len(grp)
        label = label_map.get(group, str(group))
        medians[group] = kmf.median_survival_time_

        t = kmf.survival_function_.index.tolist()
        s = kmf.survival_function_.iloc[:, 0].tolist()
        traces.append(go.Scatter(
            x=t, y=s, mode="lines",
            name=f"{label}  (n={n})",
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f"<b>{label}</b><br>Month %{{x:.0f}}: "
                "%{y:.0%} still subscribed<extra></extra>"
            ),
        ))

        if show_ci:
            ci_u = kmf.confidence_interval_.iloc[:, 1].tolist()
            ci_l = kmf.confidence_interval_.iloc[:, 0].tolist()
            t_ci = kmf.confidence_interval_.index.tolist()
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            traces.append(go.Scatter(
                x=t_ci + t_ci[::-1],
                y=ci_u + ci_l[::-1],
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.10)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

    return traces, medians


def pairwise_logrank(df_subset, group_col, ref):
    ref_grp = df_subset[df_subset[group_col] == ref]
    pvals = {}
    for group in df_subset[group_col].unique():
        if group == ref:
            continue
        grp = df_subset[df_subset[group_col] == group]
        if len(grp) < 5:
            continue
        res = logrank_test(
            ref_grp["km_duration"], grp["km_duration"],
            event_observed_A=ref_grp["km_event"],
            event_observed_B=grp["km_event"],
        )
        pvals[group] = res.p_value
    return pvals


def add_median_vlines(fig, medians, color_map):
    for group, m in medians.items():
        if pd.isna(m):
            continue
        color = color_map.get(group, NEUTRAL_COLOR)
        fig.add_shape(
            type="line", x0=m, x1=m, y0=0, y1=0.95,
            line=dict(color=color, width=1.2, dash="dot"),
            opacity=0.6,
        )


# ── CHART 1: KM by era — monthly plans ────────────────────────────────────────
print("\n[1/3] KM curves by era (monthly plans)")

traces, medians = fit_km_traces(monthly, "entry_era", ERA_COLORS, ERA_LABELS)
pvals = pairwise_logrank(monthly, "entry_era", "era_99")

# Build readable median subtitle
median_parts = []
for era in ERA_ORDER:
    m = medians.get(era)
    label = ERA_LABELS[era]
    median_parts.append(f"{label}: {fmt_months(m)}")
median_subtitle = "Median subscription length — " + " · ".join(median_parts)

# Build significance annotation
sig_parts = []
for era in ERA_ORDER[1:]:
    if era in pvals:
        sig_parts.append(
            f"{ERA_LABELS[era]} vs 99 PLN: {fmt_pval(pvals[era])}"
        )
sig_text = "  |  ".join(sig_parts)

fig = go.Figure()
for t in traces:
    fig.add_trace(t)
add_median_vlines(fig, medians, ERA_COLORS)

style_fig(
    fig,
    title="Higher prices, shorter tenure — subscribers at 99 PLN lasted twice as long",
    subtitle=median_subtitle,
    xlab="Months since sign-up",
    ylab="Probability of still subscribing",
    height=540,
)
fig.update_layout(
    yaxis=dict(tickformat=".0%", range=[0, 1.05]),
    legend=dict(
        title=dict(text="Price era", font=dict(size=11, color=TEXT_SECONDARY)),
        orientation="v", x=1.01, y=0.98,
    ),
    annotations=list(fig.layout.annotations) + [
        dict(
            text=(
                f"Statistical significance (log-rank test) — {sig_text}"
            ),
            xref="paper", yref="paper",
            x=0, y=-0.12,
            showarrow=False,
            font=dict(size=10, color=TEXT_SECONDARY, family=FONT_FAMILY),
            xanchor="left",
        ),
    ],
)
save_fig(fig, OUT_DIR, "01_km_by_era_monthly")

print("\n  Median subscription lengths (monthly plans):")
for era in ERA_ORDER:
    m = medians.get(era)
    p = pvals.get(era)
    n = (monthly["entry_era"] == era).sum()
    p_str = fmt_pval(p) if p is not None else "reference"
    print(
        f"  {ERA_LABELS[era]:10s} (n={n:3d}): {fmt_months(m):15s} | {p_str}"
    )

# ── CHART 2: KM by era — annual plans ─────────────────────────────────────────
print("\n[2/3] KM curves by era (annual plans)")

traces_a, medians_a = fit_km_traces(annual, "entry_era", ERA_COLORS, ERA_LABELS)
pvals_a = pairwise_logrank(annual, "entry_era", "era_99")

fig = go.Figure()
for t in traces_a:
    fig.add_trace(t)
add_median_vlines(fig, medians_a, ERA_COLORS)

median_parts_a = []
for era in ERA_ORDER:
    m = medians_a.get(era)
    if m is not None:
        median_parts_a.append(f"{ERA_LABELS[era]}: {fmt_months(m)}")
if median_parts_a:
    subtitle_a = "Median subscription length — " + " · ".join(median_parts_a)
else:
    subtitle_a = (
        "Annual plans: most subscribers have not yet renewed"
        " — median not reached for several eras"
    )

style_fig(
    fig,
    title="Annual subscribers show stronger loyalty — most have not yet churned",
    subtitle=subtitle_a,
    xlab="Months since sign-up",
    ylab="Probability of still subscribing",
    height=540,
)
fig.update_layout(
    yaxis=dict(tickformat=".0%", range=[0, 1.05]),
    legend=dict(
        title=dict(text="Price era", font=dict(size=11, color=TEXT_SECONDARY)),
        orientation="v", x=1.01, y=0.98,
    ),
)
save_fig(fig, OUT_DIR, "02_km_by_era_annual")

# ── CHART 3: KM — promo vs standard, monthly plans ────────────────────────────
print("[3/3] KM curves: promo vs full-price (monthly plans)")

monthly["segment"] = monthly["is_promo"].map({True: "Promo discount", False: "Full price"})
SEG_COLORS = {"Full price": ACCENT_COLOR, "Promo discount": PROMO_COLOR}
SEG_LABELS = {"Full price": "Full price", "Promo discount": "Promo discount"}

traces_ps, medians_ps = fit_km_traces(monthly, "segment", SEG_COLORS, SEG_LABELS)
pvals_ps = pairwise_logrank(monthly, "segment", "Full price")

m_full  = medians_ps.get("Full price")
m_promo = medians_ps.get("Promo discount")
p_promo = list(pvals_ps.values())[0] if pvals_ps else None

subtitle_ps = (
    f"Median — Full price: {fmt_months(m_full)} · Promo discount: {fmt_months(m_promo)}"
)

fig = go.Figure()
for t in traces_ps:
    fig.add_trace(t)
add_median_vlines(fig, medians_ps, SEG_COLORS)

style_fig(
    fig,
    title="Promo-code subscribers stay slightly longer — but generate less lifetime revenue",
    subtitle=subtitle_ps,
    xlab="Months since sign-up",
    ylab="Probability of still subscribing",
    height=520,
)
fig.update_layout(
    yaxis=dict(tickformat=".0%", range=[0, 1.05]),
    legend=dict(orientation="h", y=-0.18, x=0),
    annotations=list(fig.layout.annotations) + [
        dict(
            text=(
                "Log-rank test (promo vs full price): "
                f"{fmt_pval(p_promo) if p_promo else '—'}"
            ),
            xref="paper", yref="paper",
            x=0, y=-0.13,
            showarrow=False,
            font=dict(size=10, color=TEXT_SECONDARY, family=FONT_FAMILY),
            xanchor="left",
        ),
    ],
)
save_fig(fig, OUT_DIR, "03_km_promo_vs_standard")

n_full  = (monthly["segment"] == "Full price").sum()
n_promo = (monthly["segment"] == "Promo discount").sum()
print(f"\n  Full price  (n={n_full}):    median {fmt_months(m_full)}")
print(f"  Promo       (n={n_promo}): median {fmt_months(m_promo)}")
if p_promo:
    print(f"  {fmt_pval(p_promo)}")

print("\nSurvival analysis complete. Charts saved to outputs/04_survival/")
