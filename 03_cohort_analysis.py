import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from shared_style import (
    ERA_COLORS, ERA_ORDER, ERA_LABELS, ERA_LABELS_FULL,
    BG_CHART, GRID, BORDER, TEXT_PRIMARY, TEXT_SECONDARY, NEUTRAL_COLOR,
    FONT_FAMILY, PRELIMINARY_START,
    style_fig, save_fig, era_for_month, fmt_period, fmt_pct,
    PRICE_ERAS,
)

RAW_PATH       = "data/KDS Transactions.xlsx"
CUSTOMERS_PATH = "data/df_customers.csv"
OUT_DIR        = "outputs/03_cohort"

ERA_TRANSITION_MONTHS = {
    "era_169": pd.Period("2024-09", "M"),
    "era_199": pd.Period("2024-11", "M"),
    "era_249": pd.Period("2025-09", "M"),
}

print("Loading data...")
df_raw = pd.read_excel(RAW_PATH)
df_raw.columns = ["date", "client", "amount"]
df_raw["tx_month"] = df_raw["date"].dt.to_period("M")

cust = pd.read_csv(CUSTOMERS_PATH, parse_dates=["entry_date", "last_payment"])
cust["entry_month"] = cust["entry_month"].apply(lambda x: pd.Period(x, "M"))


def build_retention_matrix(df_transactions, df_cust_subset):
    subset_ids = set(df_cust_subset["client_id"])
    tx = df_transactions[df_transactions["client"].isin(subset_ids)].copy()
    cohort_map = (
        df_cust_subset.set_index("client_id")["entry_month"].to_dict()
    )
    tx["cohort"] = tx["client"].map(cohort_map)
    tx["offset"] = (tx["tx_month"] - tx["cohort"]).apply(lambda x: x.n)
    tx = tx[tx["offset"] >= 0]
    cohort_sizes = df_cust_subset.groupby("entry_month")["client_id"].count()
    active_counts = tx.groupby(["cohort", "offset"])["client"].nunique()
    cohorts = sorted(cohort_sizes.index)
    max_offset = int(tx["offset"].max())
    matrix = pd.DataFrame(
        index=cohorts, columns=range(max_offset + 1), dtype=float
    )
    for cohort in cohorts:
        size = cohort_sizes.get(cohort, 0)
        if size == 0:
            continue
        for offset in range(max_offset + 1):
            active = active_counts.get((cohort, offset), 0)
            matrix.loc[cohort, offset] = round(active / size * 100, 1)
    matrix.index = [str(c) for c in matrix.index]
    return matrix, cohort_sizes


def make_heatmap(matrix, cohort_sizes, title, subtitle=None, plan_type="monthly"):
    cohort_periods = [pd.Period(idx, "M") for idx in matrix.index]

    y_labels = []
    for period in cohort_periods:
        n = cohort_sizes.get(period, 0)
        label = fmt_period(period)
        prefix = "⚠ " if period >= PRELIMINARY_START else ""
        y_labels.append(f"{prefix}{label}  ({n})")

    x_labels = [f"M+{c}" for c in matrix.columns]

    colorscale = [
        [0.00, BG_CHART],
        [0.10, "#1a3a4a"],
        [0.35, "#1d6e6e"],
        [0.65, "#25a89e"],
        [1.00, "#34D399"],
    ]

    annotations = []
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                text_color = TEXT_PRIMARY if val > 45 else TEXT_SECONDARY
                annotations.append(dict(
                    x=j, y=i,
                    text=f"{val:.0f}%",
                    showarrow=False,
                    font=dict(size=9, color=text_color, family=FONT_FAMILY),
                    xref="x", yref="y",
                ))

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values.tolist(),
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        zmin=0, zmax=100,
        colorbar=dict(
            title=dict(
                text="% still paying",
                font=dict(size=11, color=TEXT_SECONDARY),
            ),
            ticksuffix="%",
            tickfont=dict(size=10, color=TEXT_SECONDARY),
            bgcolor=BG_CHART,
            bordercolor=BORDER,
        ),
        hoverongaps=False,
        hovertemplate=(
            "Cohort: %{y}<br>%{x}<br><b>%{z:.0f}% still paying</b><extra></extra>"
        ),
    ))

    for era_label, era_start in ERA_TRANSITION_MONTHS.items():
        if era_start in cohort_periods:
            row_idx = cohort_periods.index(era_start)
            fig.add_shape(
                type="line",
                x0=-0.5, x1=len(matrix.columns) - 0.5,
                y0=row_idx - 0.5, y1=row_idx - 0.5,
                line=dict(color=ERA_COLORS[era_label], width=2, dash="dash"),
            )
            price = next(p for l, _, _, p in PRICE_ERAS if l == era_label)
            fig.add_annotation(
                x=len(matrix.columns) - 0.5,
                y=row_idx - 0.5,
                text=f"  ↑ price raised to {int(price)} PLN",
                showarrow=False,
                font=dict(
                    size=10, color=ERA_COLORS[era_label], family=FONT_FAMILY
                ),
                xanchor="left",
            )

    style_fig(
        fig,
        title=title,
        subtitle=subtitle,
        height=max(520, len(matrix) * 26 + 160),
    )
    fig.update_layout(
        margin=dict(l=180, r=140, t=90, b=100),
        annotations=fig.layout.annotations + (
            dict(
                text=(
                    "⚠ = fewer than 7 months of data"
                    " — right-censored, treat as preliminary"
                ),
                xref="paper", yref="paper",
                x=0, y=-0.07,
                showarrow=False,
                font=dict(
                    size=10, color=ERA_COLORS["era_249"], family=FONT_FAMILY
                ),
                xanchor="left",
            ),
            dict(
                text="M+1 = 1 month after sign-up",
                xref="paper", yref="paper",
                x=1, y=-0.07,
                showarrow=False,
                font=dict(size=10, color=NEUTRAL_COLOR, family=FONT_FAMILY),
                xanchor="right",
            ),
        ) + tuple(annotations),
    )
    return fig


# ── Monthly plan retention matrix ──────────────────────────────────────────────
print("\n[1/3] Monthly plan retention matrix")

monthly_cust = cust[~cust["is_annual"]].copy()
matrix_m, sizes_m = build_retention_matrix(df_raw, monthly_cust)
fig = make_heatmap(
    matrix_m, sizes_m,
    title="How many subscribers from each month are still paying — monthly plans",
    subtitle="Each row = one sign-up cohort. Darker cells = more subscribers still active.",
    plan_type="monthly",
)
save_fig(fig, OUT_DIR, "01_retention_monthly")

# ── Annual plan retention matrix ───────────────────────────────────────────────
print("[2/3] Annual plan retention matrix")

annual_cust = cust[cust["is_annual"]].copy()
matrix_a, sizes_a = build_retention_matrix(df_raw, annual_cust)
fig = make_heatmap(
    matrix_a, sizes_a,
    title="Annual subscribers show stronger loyalty — fewer but more committed",
    subtitle="Annual plans: one payment per year. Gaps between payments are normal, not churn.",
    plan_type="annual",
)
save_fig(fig, OUT_DIR, "02_retention_annual")

# ── M+3 retention by era summary ──────────────────────────────────────────────
print("[3/3] 3-month retention summary by era")

offset_target = 3
era_retention = []
for plan_label, matrix, sizes in [
    ("Monthly", matrix_m, sizes_m),
    ("Annual",  matrix_a, sizes_a),
]:
    for era in ERA_ORDER:
        era_cohorts = [
            idx for idx in matrix.index
            if era_for_month(pd.Period(idx, "M")) == era
            and pd.Period(idx, "M") in sizes.index
        ]
        if not era_cohorts or offset_target not in matrix.columns:
            continue
        vals = matrix.loc[era_cohorts, offset_target].dropna()
        if len(vals) == 0:
            continue
        era_retention.append({
            "plan_type": plan_label,
            "era": era,
            "m3_retention_avg": vals.mean(),
            "n_cohorts": len(vals),
        })

if era_retention:
    ret_df = pd.DataFrame(era_retention)
    fig = go.Figure()
    opacity_map = {"Monthly": 1.0, "Annual": 0.55}
    for plan_type in ["Monthly", "Annual"]:
        sub = ret_df[ret_df["plan_type"] == plan_type]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=[ERA_LABELS[e] for e in sub["era"]],
            y=sub["m3_retention_avg"],
            name=f"{plan_type} plans",
            marker_color=[ERA_COLORS[e] for e in sub["era"]],
            marker_line=dict(width=0),
            opacity=opacity_map[plan_type],
            text=[f"{v:.0f}%" for v in sub["m3_retention_avg"]],
            textposition="outside",
            textfont=dict(size=11, color=TEXT_SECONDARY),
            hovertemplate=(
            f"{plan_type}<br>%{{x}}: %{{y:.1f}}%"
            " still paying at month 3<extra></extra>"
        ),
        ))

    style_fig(
        fig,
        title="After 3 months, what share of subscribers were still paying?",
        subtitle="Solid bars = monthly plans · Transparent = annual plans",
        xlab="Price era",
        ylab="Still subscribed at month 3 (%)",
        height=480,
    )
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 115]),
        legend=dict(orientation="h", y=-0.18, x=0),
    )
    save_fig(fig, OUT_DIR, "03_m3_retention_by_era")

print("\nCohort analysis complete. Charts saved to outputs/03_cohort/")
