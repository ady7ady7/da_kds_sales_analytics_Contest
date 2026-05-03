"""
shared_style.py — Single source of truth for all visual and copy constants.
Every script imports from here. Change once, updates everywhere.
"""

import pandas as pd

# ── PALETTE ───────────────────────────────────────────────────────────────────
ERA_COLORS = {
    "era_99":  "#60A5FA",   # electric blue   — the original, established era
    "era_169": "#FB923C",   # neon amber      — transition era
    "era_199": "#34D399",   # emerald green   — growth era
    "era_249": "#F87171",   # soft red        — premium / current era
}

PROMO_COLOR    = "#A78BFA"  # lavender        — discounted subscribers
ANNUAL_COLOR   = "#94A3B8"  # slate           — annual plan segment
POSITIVE_COLOR = "#4ADE80"  # bright green    — active, growth, positive delta
NEGATIVE_COLOR = "#F87171"  # soft red        — churn, loss (same as era_249 — context distinguishes)
NEUTRAL_COLOR  = "#94A3B8"  # slate-grey      — secondary labels, captions
ACCENT_COLOR   = "#38BDF8"  # sky blue        — highlights, annotations

# Background constants (for chart surfaces — Streamlit bg set in config.toml)
BG_CHART   = "#1C1F2E"   # card / chart background
BG_HOVER   = "#252A3D"   # hover surface
BORDER     = "#2E3347"   # subtle card border
GRID       = "#1E2235"   # barely-visible gridlines
TEXT_PRIMARY   = "#F1F5F9"  # main text
TEXT_SECONDARY = "#94A3B8"  # labels, captions  — WCAG AA 6.37:1 on chart bg
SOURCE_COLOR   = "#94A3B8"  # source note       — unified with TEXT_SECONDARY, WCAG AA 6.37:1

# ── TYPOGRAPHY ────────────────────────────────────────────────────────────────
FONT_FAMILY = "Inter, sans-serif"

# ── ERA METADATA ──────────────────────────────────────────────────────────────
# Internal label → human-readable display name
ERA_LABELS = {
    "era_99":  "99 PLN",
    "era_169": "169 PLN",
    "era_199": "199 PLN",
    "era_249": "249 PLN*",   # * = preliminary
}

ERA_LABELS_FULL = {
    "era_99":  "99 PLN  (Nov '23 – Aug '24)",
    "era_169": "169 PLN  (Sep '24 – Oct '24)",
    "era_199": "199 PLN  (Nov '24 – Aug '25)",
    "era_249": "249 PLN  (Sep '25 – present)  ⚠",
}

ERA_ORDER = ["era_99", "era_169", "era_199", "era_249"]

ERA_PRICES = {
    "era_99":  99,
    "era_169": 169,
    "era_199": 199,
    "era_249": 249,
}

PRICE_ERAS = [
    ("era_99",  pd.Period("2023-11", "M"), pd.Period("2024-08", "M"), 99.0),
    ("era_169", pd.Period("2024-09", "M"), pd.Period("2024-10", "M"), 169.0),
    ("era_199", pd.Period("2024-11", "M"), pd.Period("2025-08", "M"), 199.0),
    ("era_249", pd.Period("2025-09", "M"), pd.Period("2099-12", "M"), 249.0),
]

ERA_TRANSITION_DATES = {
    "era_169": (pd.Timestamp("2024-09-01"), "169 PLN"),
    "era_199": (pd.Timestamp("2024-11-01"), "199 PLN"),
    "era_249": (pd.Timestamp("2025-09-01"), "249 PLN"),
}

PRELIMINARY_ERA = "era_249"
PRELIMINARY_START = pd.Period("2025-09", "M")
CUTOFF_DATE = pd.Timestamp("2026-03-31")

SOURCE_NOTE = "KDS Transactions · Nov 2023 – Mar 2026"
PRELIMINARY_NOTE = "249 PLN era covers Sep 2025 – Mar 2026 only. Long-term retention conclusions are not yet possible."

# ── FORMATTING HELPERS ────────────────────────────────────────────────────────
def fmt_pln(value, decimals=0):
    """Format a PLN value: 150968.0 → '150,968 PLN'"""
    if pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f} PLN"

def fmt_pct(value, decimals=0):
    """Format a percentage: 15.7 → '16%'"""
    if pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}%"

def fmt_pval(p):
    """Format a p-value with significance stars."""
    if p < 0.001: return "p < 0.001  ✓✓✓"
    if p < 0.01:  return f"p = {p:.3f}  ✓✓"
    if p < 0.05:  return f"p = {p:.3f}  ✓"
    return f"p = {p:.2f}  —"

def fmt_months(m):
    """Format a month count: 6.0 → '6 months', nan → '>data window'"""
    if pd.isna(m):
        return ">data window"
    return f"{m:.0f} months" if m >= 1 else f"{m*30:.0f} days"

def fmt_period(period):
    """Format a Period to readable month name: Period('2023-11') → 'Nov '23'"""
    ts = period.to_timestamp()
    return ts.strftime("%b '%y")

def era_for_month(period):
    for label, start, end, _ in PRICE_ERAS:
        if start <= period <= end:
            return label
    return "unknown"

def hex_rgba(hex_color, alpha=0.15):
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── CHART STYLING ─────────────────────────────────────────────────────────────
def style_fig(fig, title, subtitle=None, height=500, xlab=None, ylab=None):
    """Apply Data Noir dark theme to any Plotly figure."""
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += (
            f"<br><span style='font-size:11px;color:{TEXT_SECONDARY}'>"
            f"{subtitle}</span>"
        )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=title_text,
            font=dict(size=15, family=FONT_FAMILY, color=TEXT_PRIMARY),
            x=0,
            pad=dict(l=16),
        ),
        font=dict(family=FONT_FAMILY, size=12, color=TEXT_PRIMARY),
        plot_bgcolor=BG_CHART,
        paper_bgcolor=BG_CHART,
        height=height,
        margin=dict(l=60, r=30, t=80, b=60),
        xaxis=dict(
            showgrid=False,
            linecolor=BORDER,
            tickfont=dict(size=11, color=TEXT_SECONDARY),
            title_font=dict(size=12, color=TEXT_SECONDARY),
        ),
        yaxis=dict(
            gridcolor=GRID,
            gridwidth=1,
            linewidth=0,
            tickfont=dict(size=11, color=TEXT_SECONDARY),
            title_font=dict(size=12, color=TEXT_SECONDARY),
        ),
        hoverlabel=dict(
            bgcolor=BG_HOVER,
            bordercolor=BORDER,
            font=dict(family=FONT_FAMILY, size=12, color=TEXT_PRIMARY),
        ),
        legend=dict(
            font=dict(size=11, color=TEXT_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
        ),
    )
    if xlab:
        fig.update_xaxes(title_text=xlab)
    if ylab:
        fig.update_yaxes(title_text=ylab)
    return fig


def add_era_vlines(fig, row=None, col=None):
    """Add price transition vertical lines with formatted labels."""
    for era, (ts, label) in ERA_TRANSITION_DATES.items():
        kw = dict(row=row, col=col) if row else {}
        fig.add_vline(
            x=ts.timestamp() * 1000,
            line_dash="dash",
            line_color=ERA_COLORS[era],
            line_width=1.2,
            opacity=0.7,
            annotation_text=label,
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color=ERA_COLORS[era],
            **kw,
        )


def save_fig(fig, out_dir, name):
    """Save as both interactive HTML and static PNG."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{name}"
    fig.write_html(path + ".html")
    fig.write_image(path + ".png", width=1200, height=fig.layout.height or 500, scale=2)
    print(f"  Saved {name}")
