# KDS Subscription Analytics

**Live dashboard:** https://github.com/ady7ady7/da_kds_Contest_analysis *(update after deploy)*

Analysis of 4,227 transactions from 1,057 subscribers across 29 months (Nov 2023 – Mar 2026), covering four price increases: 99 → 169 → 199 → 249 PLN/month.

---

## Questions

**Are we gaining or losing subscribers?** Gaining. New sign-up pace grew with each price era. The 249 PLN era recorded the highest acquisition rate of all four. Higher prices have not stopped demand.

**Are subscribers churning faster?** Yes. Median subscription length dropped from 6 months at 99 PLN to 2 months at 199 PLN. The trend is consistent across every price increase and worth watching as the 249 PLN era matures. Early positive signal: M+3 retention for 249 PLN subscribers is currently higher than it was for 199 PLN at the same observation point.

**Do promo subscribers behave differently?** They pay less but stay slightly longer. Lifetime value difference vs full-price is statistically insignificant in every era. One exception: first-month (M+1) retention is noticeably lower for promo subscribers in the 99 and 169 PLN eras.

The best-performing era overall was **199 PLN**: highest total revenue, driven by a higher price combined with the largest subscriber volume of any era.
---

## Pipeline

```
00_validate.py            raw data validation
01_eda.py                 exploratory analysis
02_feature_engineering.py builds df_customers.csv (1 row = 1 subscriber)
03_cohort_analysis.py     cohort retention matrices
04_survival.py            Kaplan-Meier survival curves
05_ltv_revenue.py         LTV and revenue analysis
06_dashboard.py           Streamlit dashboard (final presentation)
shared_style.py           single source of truth: colors, formatters, chart style
```

---

## Setup

```bash
pip install -r requirements.txt
python 02_feature_engineering.py   # generates data/df_customers.csv
streamlit run 06_dashboard.py
```

---

## Data

Source file: `data/KDS Transactions.xlsx` — three columns: `date`, `client`, `amount`.
