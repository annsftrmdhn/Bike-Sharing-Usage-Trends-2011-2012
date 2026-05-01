import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="🚲 Bike Sharing Dashboard", page_icon="🚲", layout="wide")

@st.cache_data
def load_data(path: str = "main_data.csv"):
    df = pd.read_csv(path, parse_dates=["dteday"], index_col="dteday")
    return df

df = load_data()

# Sidebar filter
min_date, max_date = df.index.min().date(), df.index.max().date()
start_date = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.sidebar.error("⚠️ Start date must be before end date."); st.stop()
mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
filtered = df.loc[mask]
if filtered.empty:
    st.warning("No data in selected date range."); st.stop()

# Header & metrics
st.title("🚲 Bike Sharing Dashboard")
st.markdown(f"Showing data from **{start_date}** to **{end_date}** ({len(filtered):,} days)")
total_rentals = int(filtered["cnt"].sum())
avg_daily     = round(filtered["cnt"].mean(), 1)
peak_date     = filtered["cnt"].idxmax().date()
peak_val      = int(filtered["cnt"].max())
col1, col2, col3 = st.columns(3)
col1.metric("Total Rentals",        f"{total_rentals:,}")
col2.metric("Avg Daily Rentals",    f"{avg_daily:,}")
col3.metric("Peak Day",             f"{peak_date}  ({peak_val:,})")
st.divider()

# Monthly trend
st.subheader("📈 Monthly Rental Trend")
monthly = (filtered.groupby([filtered.index.year, filtered.index.month])["cnt"].sum())
monthly.index.names = ["year", "month"]
monthly = monthly.reset_index()
monthly["date_label"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(monthly["date_label"], monthly["cnt"], marker="o", linewidth=1.8, markersize=4, color="#1565C0")
ax1.fill_between(monthly["date_label"], monthly["cnt"], alpha=0.12, color="#1565C0")
ax1.set_xlabel("Month"); ax1.set_ylabel("Total Rentals"); ax1.set_title("Monthly Rental Trend")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
st.pyplot(fig1); plt.close(fig1)

# ARIMA forecast
st.subheader("🔮 6-Month Rental Forecast (ARIMA)")
monthly_series = monthly.set_index("date_label")["cnt"].asfreq("MS")
if len(monthly_series) >= 12:
    try:
        model = ARIMA(monthly_series, order=(1, 1, 1)).fit()
        forecast = model.forecast(steps=6)
        last_date = monthly_series.index[-1]
        future_idx = pd.date_range(last_date, periods=7, freq="MS")[1:]
        forecast.index = future_idx
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(monthly_series.index, monthly_series.values, label="Historical", linewidth=1.8, color="#1565C0")
        ax2.plot(forecast.index, forecast.values, label="Forecast", linewidth=1.8, linestyle="--", color="#E53935")
        ax2.legend(); ax2.set_xlabel("Month"); ax2.set_ylabel("Total Rentals")
        ax2.set_title("Monthly Rentals — ARIMA(1,1,1) Forecast")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        st.pyplot(fig2); plt.close(fig2)
    except Exception as e:
        st.warning(f"ARIMA fitting failed: {e}")
else:
    st.info("Not enough data for forecasting (need ≥ 12 months).")

# Season barplot
st.subheader("🍂 Average Rentals by Season")
season_order  = ["Spring", "Summer", "Fall", "Winter"]
season_colors = ["#4CAF50", "#FF9800", "#F44336", "#2196F3"]
season_agg = (filtered.groupby("season")["cnt"].mean().reindex(season_order))
fig3, ax3 = plt.subplots(figsize=(7, 4))
bars = ax3.bar(season_order, season_agg.values, color=season_colors, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, season_agg.values):
    if not pd.isna(val):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40, f"{val:,.0f}", ha="center", va="bottom", fontsize=9)
ax3.set_xlabel("Season"); ax3.set_ylabel("Avg Daily Rentals"); ax3.set_title("Average Daily Rentals by Season")
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.1f}k"))
plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

# Hourly pattern (if available)
if "hr" in filtered.columns:
    st.subheader("⏰ Average Rentals by Hour of Day")
    hourly_agg = filtered.groupby("hr")["cnt"].mean()
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.fill_between(hourly_agg.index, hourly_agg.values, alpha=0.2, color="#E65100")
    ax4.plot(hourly_agg.index, hourly_agg.values, marker=".", markersize=5, linewidth=1.8, color="#E65100")
    ax4.set_xlabel("Hour of Day (0–23)"); ax4.set_ylabel("Avg Rentals"); ax4.set_title("Average Rentals by Hour of Day")
    ax4.set_xticks(range(0, 24, 2)); plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)

st.divider()
st.caption("📂 Data source: **main_data.csv** (dteday as DatetimeIndex, season as string labels)  |  Built with Streamlit")
