import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bike-Sharing Usage Trends 2011-2012",
    page_icon="🚲",
    layout="wide",
)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path   = os.path.join(current_dir, "main_data.csv")
    df = pd.read_csv(file_path, parse_dates=["dteday"], index_col="dteday")
    # Ensure yr is stored as string label for display
    df["yr"] = df["yr"].astype(str)
    return df

df = load_data()

# ── Sidebar – date filter ─────────────────────────────────────────────────────
st.sidebar.title("🔎 Filters")
min_date = df.index.min().date()
max_date = df.index.max().date()
start_date, end_date = st.sidebar.date_input(
    "Select date range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date,
)
mask = (df.index.date >= start_date) & (df.index.date <= end_date)
dff  = df.loc[mask].copy()

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🚲 Bike-Sharing Usage Trends 2011–2012")
st.caption(
    f"Showing data from **{start_date}** to **{end_date}**  ({len(dff):,} days)"
)

# ── Key metrics ───────────────────────────────────────────────────────────────
total_rentals = int(dff["cnt"].sum())
avg_daily     = int(dff["cnt"].mean())
peak_day      = dff["cnt"].idxmax()
peak_cnt      = int(dff["cnt"].max())

col1, col2, col3 = st.columns(3)
col1.metric("🚴 Total Rentals",    f"{total_rentals:,}")
col2.metric("📅 Avg Daily Rentals", f"{avg_daily:,}")
col3.metric("🏆 Peak Day", f"{peak_day.strftime('%Y-%m-%d')} ({peak_cnt:,})")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# Q1 – Monthly & Yearly rental trend with growth rate
# ═════════════════════════════════════════════════════════════════════════════
st.header("Q1: Monthly & Yearly Rental Trend")

monthly = (
    dff.groupby(["yr", "mnth"])["cnt"]
    .sum()
    .reset_index()
    .rename(columns={"yr": "Year", "mnth": "Month", "cnt": "Rentals"})
)

fig1, ax1 = plt.subplots(figsize=(12, 4))
for yr, grp in monthly.groupby("Year"):
    ax1.plot(grp["Month"], grp["Rentals"], marker="o", label=str(yr))
ax1.set_title("Monthly Total Rentals by Year")
ax1.set_xlabel("Month")
ax1.set_ylabel("Total Rentals")
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"])
ax1.legend(title="Year")
st.pyplot(fig1)
plt.close(fig1)

# Year-over-year growth rate
yearly = dff.groupby("yr")["cnt"].sum().sort_index()
if len(yearly) == 2:
    growth = (yearly.iloc[1] - yearly.iloc[0]) / yearly.iloc[0] * 100
    col_a, col_b = st.columns(2)
    col_a.metric(f"{yearly.index[0]} Total Rentals", f"{int(yearly.iloc[0]):,}")
    col_b.metric(f"{yearly.index[1]} Total Rentals", f"{int(yearly.iloc[1]):,}",
                 delta=f"{growth:+.1f}% YoY")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# Q2 – Average daily rentals: workingday vs. holiday / weekend
# ═════════════════════════════════════════════════════════════════════════════
st.header("Q2: Average Rentals – Workingday vs. Holiday/Weekend")
st.info(
    "ℹ️ The dataset is **day-level** (no hourly breakdown). "
    "Averages shown are **daily** totals grouped by day type."
)

dff["day_type"] = dff["workingday"].map({1: "Working Day", 0: "Holiday/Weekend"})
q2 = dff.groupby("day_type")["cnt"].mean().reset_index()
q2.columns = ["Day Type", "Avg Daily Rentals"]

fig2, ax2 = plt.subplots(figsize=(6, 4))
bars2 = ax2.bar(q2["Day Type"], q2["Avg Daily Rentals"],
                color=["#4C72B0", "#DD8452"], width=0.5)
ax2.bar_label(bars2, fmt="%.0f", padding=4)
ax2.set_title("Average Daily Rentals by Day Type")
ax2.set_ylabel("Avg Daily Rentals")
st.pyplot(fig2)
plt.close(fig2)
st.dataframe(q2.style.format({"Avg Daily Rentals": "{:.0f}"}))

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# Q3 – Average daily rentals by season
# ═════════════════════════════════════════════════════════════════════════════
st.header("Q3: Average Daily Rentals by Season")

season_order = ["Spring", "Summer", "Fall", "Winter"]
present_seasons = [s for s in season_order if s in dff["season"].unique()]
q3 = (
    dff.groupby("season")["cnt"]
    .mean()
    .reindex(present_seasons)
    .reset_index()
)
q3.columns = ["Season", "Avg Daily Rentals"]

palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
fig3, ax3 = plt.subplots(figsize=(7, 4))
bars3 = ax3.bar(q3["Season"], q3["Avg Daily Rentals"],
                color=palette[:len(q3)], width=0.5)
ax3.bar_label(bars3, fmt="%.0f", padding=4)
ax3.set_title("Average Daily Rentals by Season")
ax3.set_ylabel("Avg Daily Rentals")
st.pyplot(fig3)
plt.close(fig3)
st.dataframe(q3.style.format({"Avg Daily Rentals": "{:.0f}"}))

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# EDA – Exploratory Data Analysis
# ═════════════════════════════════════════════════════════════════════════════
st.header("🔍 Exploratory Data Analysis")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Distribution", "Boxplot by Season", "Correlation Heatmap", "Count by Season"]
)

with tab1:
    fig_d, ax_d = plt.subplots(figsize=(8, 4))
    ax_d.hist(dff["cnt"], bins=40, color="#4C72B0", edgecolor="white")
    ax_d.set_title("Distribution of Daily Rentals")
    ax_d.set_xlabel("Daily Rentals")
    ax_d.set_ylabel("Frequency")
    st.pyplot(fig_d);  plt.close(fig_d)

with tab2:
    ordered = [s for s in season_order if s in dff["season"].unique()]
    dff_s = dff[dff["season"].isin(ordered)].copy()
    dff_s["season"] = pd.Categorical(dff_s["season"], categories=ordered, ordered=True)
    groups = [g["cnt"].values for _, g in dff_s.sort_values("season")
                                                .groupby("season", observed=True)]
    fig_b, ax_b = plt.subplots(figsize=(8, 4))
    ax_b.boxplot(groups, labels=ordered[:len(groups)], patch_artist=True)
    ax_b.set_title("Rental Distribution by Season")
    ax_b.set_ylabel("Daily Rentals")
    st.pyplot(fig_b);  plt.close(fig_b)

with tab3:
    num_cols = ["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
    corr = dff[num_cols].corr()
    fig_h, ax_h = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax_h, linewidths=0.5)
    ax_h.set_title("Correlation Heatmap")
    st.pyplot(fig_h);  plt.close(fig_h)

with tab4:
    season_counts = (dff["season"].value_counts()
                     .reindex([s for s in season_order if s in dff["season"].unique()]))
    fig_c, ax_c = plt.subplots(figsize=(7, 4))
    ax_c.bar(season_counts.index, season_counts.values,
             color=palette[:len(season_counts)], width=0.5)
    ax_c.set_title("Number of Days per Season")
    ax_c.set_ylabel("Days")
    st.pyplot(fig_c);  plt.close(fig_c)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# ARIMA – 6-month forecast (trained on the full dataset, not filtered)
# ═════════════════════════════════════════════════════════════════════════════
st.header("📈 ARIMA 6-Month Forecast")

monthly_ts = (
    df.groupby(["yr", "mnth"])["cnt"]
    .sum()
    .reset_index()
    .sort_values(["yr", "mnth"])
)
ts_values = monthly_ts["cnt"].values

try:
    model  = ARIMA(ts_values, order=(1, 1, 1))
    result = model.fit()
    n_forecast  = 6
    forecast    = result.forecast(steps=n_forecast)

    last_yr   = int(monthly_ts["yr"].iloc[-1])
    last_mnth = int(monthly_ts["mnth"].iloc[-1])
    future_labels = [
        f"{last_yr + (last_mnth + i - 1) // 12}-{(last_mnth + i - 1) % 12 + 1:02d}"
        for i in range(1, n_forecast + 1)
    ]

    hist_labels = [f"{row.yr}-{row.mnth:02d}" for _, row in monthly_ts.iterrows()]
    fig_f, ax_f = plt.subplots(figsize=(12, 4))
    ax_f.plot(hist_labels, ts_values,  marker="o", label="Historical",       color="#4C72B0")
    ax_f.plot(future_labels, forecast, marker="s", linestyle="--",
              label="Forecast (ARIMA)", color="#DD8452")
    ax_f.set_title("Monthly Rentals – ARIMA 6-Month Forecast")
    ax_f.set_xlabel("Month")
    ax_f.set_ylabel("Total Rentals")
    plt.setp(ax_f.get_xticklabels(), rotation=45, ha="right")
    ax_f.legend()
    st.pyplot(fig_f);  plt.close(fig_f)

    st.dataframe(
        pd.DataFrame({"Month": future_labels,
                      "Forecasted Rentals": forecast.astype(int)})
        .set_index("Month")
    )
except Exception as e:
    st.warning(f"ARIMA model could not be fitted: {e}")

st.markdown("---")
st.caption("Data source: UCI Bike Sharing Dataset | Dashboard by Streamlit")
