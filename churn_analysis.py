"""
Customer Churn Analysis (subscription business)
Topics: data type classification | derived metrics | group by | trend analysis | correlation
Stack: Python, pandas, NumPy, Matplotlib, Seaborn
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
IMG = "images/"
rng = np.random.default_rng(21)
OBS_END = pd.Timestamp("2025-12-31")

# ------------------------------------------------------------------
# 0. DATA (synthetic; replace with pd.read_csv for real data)
# ------------------------------------------------------------------
n = 5000
df = pd.DataFrame({
    "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
    "signup_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 1095, n), unit="D"),
    "contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20]),
    "plan": rng.choice(["Basic", "Standard", "Premium"], n, p=[.4, .4, .2]),
    "payment_method": rng.choice(["Auto-pay (card)", "Auto-pay (bank)", "Manual / invoice", "UPI / wallet"], n, p=[.28, .22, .2, .3]),
    "support_tickets": rng.poisson(1.6, n),
    "late_payments": rng.poisson(0.7, n),
    "usage_hours": rng.gamma(4, 12, n).round(1),
})
base = df["plan"].map({"Basic": 30, "Standard": 55, "Premium": 85})
df["monthly_charges"] = (base + rng.normal(0, 6, n)).round(2).clip(lower=10)
df["autopay"] = df["payment_method"].str.startswith("Auto-pay")

# churn probability (tenure deliberately excluded: it is an outcome of churn)
logit = (-1.3
         + df["contract"].map({"Month-to-month": 1.2, "One year": -0.2, "Two year": -1.3})
         + 0.35 * df["support_tickets"] + 0.30 * df["late_payments"]
         + 0.015 * (df["monthly_charges"] - 55)
         - 0.012 * (df["usage_hours"] - 48)
         - 0.5 * df["autopay"])
df["churned"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

life_mean = np.where(df["contract"] == "Month-to-month", 7, np.where(df["contract"] == "One year", 13, 18))
life = np.maximum(1, rng.exponential(life_mean)).round().astype(int)
churn_date = df["signup_date"] + pd.to_timedelta(life * 30, unit="D")
df.loc[(df["churned"] == 1) & (churn_date > OBS_END), "churned"] = 0   # not yet churned within window
df["churn_date"] = churn_date.where(df["churned"] == 1)
end = df["churn_date"].fillna(OBS_END)
df["tenure_months"] = ((end - df["signup_date"]).dt.days / 30).round().astype(int).clip(lower=1)

# ------------------------------------------------------------------
# 1. DATA TYPE CLASSIFICATION
# ------------------------------------------------------------------
def classify(col, s):
    if pd.api.types.is_datetime64_any_dtype(s): return "Datetime"
    if col.endswith("_id"): return "Identifier"
    if pd.api.types.is_bool_dtype(s) or set(s.dropna().unique()) <= {0, 1}: return "Binary"
    if pd.api.types.is_numeric_dtype(s):
        return "Numeric - Discrete" if s.nunique() < 15 else "Numeric - Continuous"
    return "Categorical"

types = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values,
                      "classification": [classify(c, df[c]) for c in df.columns],
                      "unique_values": [df[c].nunique() for c in df.columns],
                      "missing": df.isna().sum().values})
print("=== 1. DATA TYPE CLASSIFICATION ===\n", types.to_string(index=False))
print(f"\nCustomers: {len(df):,} | Overall churn rate: {df.churned.mean():.1%}")

fig, ax = plt.subplots(figsize=(8, 4.5))
cnt = types["classification"].value_counts()
b = ax.barh(cnt.index, cnt.values, color=sns.color_palette("deep", len(cnt)))
ax.bar_label(b, padding=3); ax.set_title("Columns by Data Type Classification", fontweight="bold")
ax.set_xlabel("Number of columns")
plt.tight_layout(); plt.savefig(IMG + "1_data_types.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 2. DERIVED METRICS
# ------------------------------------------------------------------
df["tenure_group"] = pd.cut(df["tenure_months"], [0, 6, 12, 24, 100], labels=["0-6 mo", "7-12 mo", "13-24 mo", "25+ mo"])
df["charge_band"] = pd.qcut(df["monthly_charges"], 3, labels=["Low", "Medium", "High"])
df["lifetime_value"] = (df["monthly_charges"] * df["tenure_months"]).round(2)
df["tickets_per_year"] = (df["support_tickets"] / df["tenure_months"] * 12).round(2)
df["risk_score"] = ((df["contract"] == "Month-to-month").astype(int) * 2
                    + (df["support_tickets"] >= 3).astype(int)
                    + (df["late_payments"] >= 2).astype(int)
                    + (~df["autopay"]).astype(int)
                    + (df["usage_hours"] < 30).astype(int))
df["risk_band"] = pd.cut(df["risk_score"], [-1, 1, 3, 10], labels=["Low", "Medium", "High"])
print("\n=== 2. DERIVED METRICS (sample) ===\n",
      df[["customer_id", "tenure_group", "charge_band", "lifetime_value", "risk_score", "risk_band"]].head())

# ------------------------------------------------------------------
# 3. GROUP BY: churn rate by dimension
# ------------------------------------------------------------------
dims = ["contract", "plan", "payment_method", "tenure_group", "risk_band"]
rates = {d: df.groupby(d, observed=True)["churned"].mean().mul(100) for d in dims}
for d in dims: print(f"\nChurn % by {d}:\n{rates[d].round(1)}")
lost_rev = df[df.churned == 1].groupby("plan")["monthly_charges"].sum()
print("\nMonthly revenue lost by plan:\n", lost_rev.round(0))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
pal = {"contract": "#E76F51", "plan": "#264653", "payment_method": "#2A9D8F", "tenure_group": "#E9C46A"}
for ax, d in zip(axes.ravel(), ["contract", "plan", "payment_method", "tenure_group"]):
    r = rates[d]; bars = ax.bar(r.index.astype(str), r.values, color=pal[d])
    ax.bar_label(bars, fmt="%.1f%%", padding=2); ax.set_title(f"Churn Rate by {d.replace('_', ' ').title()}", fontweight="bold")
    ax.axhline(df.churned.mean() * 100, color="black", ls="--", lw=1, label="overall"); ax.set_ylabel("Churn %"); ax.legend()
    ax.tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.savefig(IMG + "2_group_by_churn.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 4. TREND ANALYSIS
# ------------------------------------------------------------------
df["signup_month"] = df["signup_date"].dt.to_period("M").dt.to_timestamp()
df["churn_month"] = df["churn_date"].dt.to_period("M").dt.to_timestamp()
new = df.groupby("signup_month").size()
churned_m = df.dropna(subset=["churn_month"]).groupby("churn_month").size()
trend = pd.concat([new.rename("new"), churned_m.rename("churned")], axis=1).fillna(0)
trend = trend[trend.index <= "2025-12-01"]
trend["net_adds"] = trend["new"] - trend["churned"]
trend["churned_3m_avg"] = trend["churned"].rolling(3).mean()
df["signup_qtr"] = df["signup_date"].dt.to_period("Q").astype(str)
cohort = df.groupby("signup_qtr")["churned"].mean().mul(100)
print("\n=== 4. TREND (last 6 months) ===\n", trend.tail(6))

fig, ax = plt.subplots(2, 1, figsize=(12, 8.5), height_ratios=[3, 2])
ax[0].plot(trend.index, trend["new"], label="New customers", color="#2A9D8F", lw=2)
ax[0].plot(trend.index, trend["churned"], label="Churned customers", color="#E76F51", lw=2)
ax[0].plot(trend.index, trend["churned_3m_avg"], label="Churned (3M avg)", color="black", ls="--")
ax[0].set_title("Monthly New vs Churned Customers", fontweight="bold"); ax[0].legend()
bars = ax[1].bar(cohort.index, cohort.values, color="#264653"); ax[1].bar_label(bars, fmt="%.0f%%", padding=2)
ax[1].set_title("Churn Rate by Signup Quarter (cohort view)", fontweight="bold"); ax[1].set_ylabel("Churn %")
ax[1].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(IMG + "3_trend.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 5. CORRELATION ANALYSIS
# ------------------------------------------------------------------
ncols = ["churned", "tenure_months", "monthly_charges", "support_tickets", "late_payments", "usage_hours", "lifetime_value"]
corr = df[ncols].assign(autopay=df["autopay"].astype(int)).corr()
cc = corr["churned"].drop("churned").sort_values()
print("\n=== 5. CORRELATION WITH CHURN ===\n", cc.round(2))

fig, axes = plt.subplots(1, 2, figsize=(15, 6), width_ratios=[1.3, 1])
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool), 1), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0])
axes[0].set_title("Correlation Matrix", fontweight="bold")
axes[1].barh(cc.index, cc.values, color=np.where(cc.values > 0, "#E76F51", "#2A9D8F"))
axes[1].axvline(0, color="black", lw=.8); axes[1].set_title("Correlation with Churn", fontweight="bold")
plt.tight_layout(); plt.savefig(IMG + "4_correlation.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 6. LINKEDIN DASHBOARD
# ------------------------------------------------------------------
fig = plt.figure(figsize=(16, 9))
fig.suptitle("Customer Churn Analysis  |  Python Data Analytics", fontsize=20, fontweight="bold", y=.98)
gs = fig.add_gridspec(2, 2, hspace=.38, wspace=.38)
a1 = fig.add_subplot(gs[0, 0]); r = rates["contract"]
bb = a1.bar(r.index, r.values, color="#E76F51"); a1.bar_label(bb, fmt="%.1f%%", padding=2)
a1.set_title("Group By: Churn Rate by Contract", fontweight="bold")
a2 = fig.add_subplot(gs[0, 1]); r = rates["risk_band"]
bb = a2.bar(r.index.astype(str), r.values, color=["#2A9D8F", "#E9C46A", "#E76F51"]); a2.bar_label(bb, fmt="%.1f%%", padding=2)
a2.set_title("Derived Metric: Churn Rate by Risk Band", fontweight="bold")
a3 = fig.add_subplot(gs[1, 0])
a3.plot(trend.index, trend["new"], color="#2A9D8F", lw=2, label="New"); a3.plot(trend.index, trend["churned"], color="#E76F51", lw=2, label="Churned")
a3.legend(); a3.set_title("Trend: New vs Churned Customers", fontweight="bold")
a4 = fig.add_subplot(gs[1, 1])
a4.barh(cc.index, cc.values, color=np.where(cc.values > 0, "#E76F51", "#2A9D8F")); a4.axvline(0, color="black", lw=.8)
a4.set_title("Correlation: Features vs Churn", fontweight="bold")
plt.savefig(IMG + "0_linkedin_dashboard.png", dpi=200, bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------
# INSIGHTS + EXPORT
# ------------------------------------------------------------------
print("\n=== KEY INSIGHTS ===")
print(f"- Overall churn: {df.churned.mean():.1%} ({df.churned.sum():,} of {len(df):,})")
print(f"- Month-to-month churn {rates['contract']['Month-to-month']:.1f}% vs Two year {rates['contract']['Two year']:.1f}%")
print(f"- High-risk band churn {rates['risk_band']['High']:.1f}% vs Low {rates['risk_band']['Low']:.1f}%")
print(f"- Share of customers in High risk band: {(df.risk_band=='High').mean():.1%}")
print(f"- Monthly revenue lost to churn: {df.loc[df.churned==1,'monthly_charges'].sum():,.0f} of {df.monthly_charges.sum():,.0f}")
print(f"- Autopay churn {df[df.autopay].churned.mean():.1%} vs manual {df[~df.autopay].churned.mean():.1%}")
out = df.drop(columns=["signup_month", "churn_month", "signup_qtr"]).copy()
for c in ["signup_date", "churn_date"]: out[c] = out[c].dt.date
for c in ["tenure_group", "charge_band", "risk_band"]: out[c] = out[c].astype(str)
out.to_csv("powerbi/churn_data.csv", index=False)
print("\nExported powerbi/churn_data.csv")
