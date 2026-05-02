import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="🚲 Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide"
)

@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "main_data.csv")
    df = pd.read_csv(file_path, parse_dates=["dteday"], index_col="dteday")
    return df

data = load_data()

st.sidebar.header("🔎 Filters")
start_date = st.sidebar.date_input("Start Date", min_value=data.index.min(), max_value=data.index.max(), value=data.index.min())
end_date = st.sidebar.date_input("End Date", min_value=data.index.min(), max_value=data.index.max(), value=data.index.max())

if start_date > end_date:
    st.sidebar.error("Error: Start Date must be before End Date.")

filtered_data = data.loc[start_date:end_date].copy()

st.title("🚲 Bike Sharing Dashboard")
st.markdown("### Analisis interaktif data penyewaan sepeda (2011–2012)")

# Key metrics
st.header("📊 Key Metrics")
total_rentals = filtered_data["cnt"].sum()
avg_daily = filtered_data["cnt"].mean()
peak_day = filtered_data["cnt"].idxmax()
peak_value = filtered_data.loc[peak_day, "cnt"]

col1, col2, col3 = st.columns(3)
col1.metric("Total Rentals", f"{total_rentals:,}")
col2.metric("Average Daily Rentals", f"{avg_daily:.1f}")
col3.metric("Peak Day", peak_day.strftime("%Y-%m-%d"), f"{int(peak_value):,} rentals")

# Monthly rental trend
st.header("📈 Monthly Rental Trend")
monthly_rentals = filtered_data["cnt"].resample("ME").sum()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly_rentals.index, monthly_rentals.values, marker='o', color='steelblue')
ax.set_xlabel("Month")
ax.set_ylabel("Total Rentals")
ax.set_title("Monthly Bike Rental Trend (2011–2012)")
plt.xticks(rotation=45)
st.pyplot(fig)

# Forecasting
st.header("🔮 Forecasting Next 6 Months")
if len(monthly_rentals) >= 12:
    model = ARIMA(monthly_rentals, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=6)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(monthly_rentals.index, monthly_rentals.values, label="Historical Data", color='steelblue')
    ax2.plot(forecast.index, forecast.values, label="Forecast", linestyle='--', color='orange')
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Total Rentals")
    ax2.set_title("Forecasted Bike Rentals for Next 6 Months")
    ax2.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig2)
else:
    st.warning("Data tidak cukup untuk melakukan forecasting.")

# Rentals by season — warna highlight bar tertinggi
st.header("🌦️ Rentals by Season")
season_order = ["Spring", "Summer", "Fall", "Winter"]
season_avg = filtered_data.groupby("season")["cnt"].mean()\
    .reindex(season_order).dropna()

if season_avg.empty:
    st.warning("Tidak ada data season untuk rentang tanggal ini.")
else:
    max_season = season_avg.idxmax()
    colors = ["tomato" if s == max_season else "steelblue" for s in season_avg.index]

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    bars = ax3.bar(season_avg.index, season_avg.values, color=colors)
    ax3.set_ylabel("Average Rentals")
    ax3.set_title("Average Rentals by Season (merah = tertinggi)")
    ax3.legend(handles=[
        plt.Rectangle((0,0),1,1, color='tomato', label='Tertinggi'),
        plt.Rectangle((0,0),1,1, color='steelblue', label='Lainnya')
    ])
    st.pyplot(fig3)

st.markdown("---")
st.caption("📂 Data source: main_data.csv | Built with Streamlit")