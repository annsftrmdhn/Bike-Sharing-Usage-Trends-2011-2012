# dashboard.py — Corrected Streamlit Bike Sharing Dashboard
import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide",
)

# ── Data loading ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load main_data.csv from the same directory as this script."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath  = os.path.join(base_dir, "main_data.csv")
    df = pd.read_csv(filepath, parse_dates=["dteday"])
    df.set_index("dteday", inplace=True)
    # Ensure yr column is string for display ("2011" / "2012")
    df["yr"] = df["yr"].astype(str)
    return df

df = load_data()

# ── Sidebar — Date filter ───────────────────────────────────────────────────────
st.sidebar.header("⚙️ Filter Data")

min_date = df.index.min().date()
max_date = df.index.max().date()

date_range = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Safe unpacking — guard against single-date selection
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.sidebar.warning("⚠️ Pilih rentang tanggal yang valid (dua tanggal).")
    st.stop()

if start_date > end_date:
    st.sidebar.error("❌ Tanggal mulai harus sebelum tanggal akhir.")
    st.stop()

# Apply filter — dff is used consistently everywhere below
dff = df.loc[str(start_date) : str(end_date)].copy()

if dff.empty:
    st.warning("⚠️ Tidak ada data untuk rentang tanggal yang dipilih.")
    st.stop()

# ── Title ───────────────────────────────────────────────────────────────────────
st.title("🚲 Dashboard Analisis Bike Sharing (2011–2012)")
st.markdown(
    f"Menampilkan data dari **{start_date}** hingga **{end_date}** "
    f"({len(dff):,} hari)"
)

# ── Key Metrics (all from dff) ──────────────────────────────────────────────────
total_rentals = int(dff["cnt"].sum())
avg_daily     = round(dff["cnt"].mean(), 2)
peak_row      = dff["cnt"].idxmax()
peak_day      = peak_row.strftime("%Y-%m-%d")
peak_cnt      = int(dff.loc[peak_row, "cnt"])

col1, col2, col3 = st.columns(3)
col1.metric("🚲 Total Penyewaan",    f"{total_rentals:,}")
col2.metric("📅 Rata-rata Harian",   f"{avg_daily:,.1f}")
col3.metric("🏆 Puncak Penyewaan",   f"{peak_cnt:,} ({peak_day})")

st.markdown("---")

# ── Monthly & Yearly Trend (dff) ────────────────────────────────────────────────
st.subheader("📈 Tren Bulanan Penyewaan Sepeda")

monthly = (
    dff.assign(year=dff.index.year, month=dff.index.month)
    .groupby(["year", "month"])["cnt"]
    .sum()
    .reset_index()
)
monthly["period"] = pd.to_datetime(
    monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
)
monthly.sort_values("period", inplace=True)

fig1, ax1 = plt.subplots(figsize=(10, 4), constrained_layout=True)
for yr_val, grp in monthly.groupby("year"):
    ax1.plot(grp["period"], grp["cnt"], marker="o", label=str(yr_val))
ax1.set_title("Tren Bulanan Penyewaan Sepeda")
ax1.set_xlabel("Periode")
ax1.set_ylabel("Total Penyewaan")
ax1.legend(title="Tahun")
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
st.pyplot(fig1)
plt.close(fig1)

st.markdown("---")

# ── Working Day vs Holiday (dff) ────────────────────────────────────────────────
st.subheader("📊 Rata-rata Penyewaan: Hari Kerja vs Hari Libur/Akhir Pekan")

wd_avg = dff.groupby("workingday")["cnt"].mean().rename({0: "Libur/Weekend", 1: "Hari Kerja"})

fig2, ax2 = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax2.bar(wd_avg.index, wd_avg.values, color=["#E07B54", "#4C72B0"])
ax2.set_title("Rata-rata Penyewaan per Tipe Hari")
ax2.set_xlabel("Tipe Hari")
ax2.set_ylabel("Rata-rata Penyewaan")
for i, v in enumerate(wd_avg.values):
    ax2.text(i, v + 20, f"{v:,.0f}", ha="center", fontsize=10)
st.pyplot(fig2)
plt.close(fig2)

st.markdown("---")

# ── Average Rentals by Season (dff) ────────────────────────────────────────────
st.subheader("🌸 Rata-rata Penyewaan per Musim")

season_order  = ["Spring", "Summer", "Fall", "Winter"]
season_avg    = (
    dff.groupby("season")["cnt"]
    .mean()
    .reindex(season_order)
    .dropna()
)

if not season_avg.empty:
    fig3, ax3 = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax3.bar(season_avg.index, season_avg.values, color=sns.color_palette("Set2", len(season_avg)))
    ax3.set_title("Rata-rata Penyewaan per Musim")
    ax3.set_xlabel("Musim")
    ax3.set_ylabel("Rata-rata Penyewaan")
    for i, v in enumerate(season_avg.values):
        ax3.text(i, v + 20, f"{v:,.0f}", ha="center", fontsize=10)
    st.pyplot(fig3)
    plt.close(fig3)
else:
    st.info("Tidak ada data musim untuk rentang tanggal yang dipilih.")

st.markdown("---")

# ── EDA Tabs (dff) ──────────────────────────────────────────────────────────────
st.subheader("🔍 Eksplorasi Data (EDA)")

tab1, tab2, tab3, tab4 = st.tabs([
    "Distribusi", "Boxplot per Musim", "Heatmap Korelasi", "Jumlah per Musim"
])

with tab1:
    st.markdown("**Distribusi Total Penyewaan Harian**")
    fig_t1, ax_t1 = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax_t1.hist(dff["cnt"], bins=30, color="#4C72B0", edgecolor="white")
    ax_t1.set_title("Distribusi Total Penyewaan Harian")
    ax_t1.set_xlabel("Jumlah Penyewaan")
    ax_t1.set_ylabel("Frekuensi")
    st.pyplot(fig_t1)
    plt.close(fig_t1)

with tab2:
    st.markdown("**Boxplot Penyewaan per Musim**")
    seasons_present = [s for s in season_order if s in dff["season"].values]
    if seasons_present:
        fig_t2, ax_t2 = plt.subplots(figsize=(7, 4), constrained_layout=True)
        data_by_season = [dff[dff["season"] == s]["cnt"].values for s in seasons_present]
        ax_t2.boxplot(data_by_season, labels=seasons_present, patch_artist=True)
        ax_t2.set_title("Boxplot Penyewaan per Musim")
        ax_t2.set_xlabel("Musim")
        ax_t2.set_ylabel("Jumlah Penyewaan")
        st.pyplot(fig_t2)
        plt.close(fig_t2)
    else:
        st.info("Tidak ada data musim.")

with tab3:
    st.markdown("**Heatmap Korelasi**")
    num_cols = dff.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) >= 2:
        fig_t3, ax_t3 = plt.subplots(figsize=(9, 7), constrained_layout=True)
        sns.heatmap(dff[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax_t3)
        ax_t3.set_title("Heatmap Korelasi Fitur Numerik")
        st.pyplot(fig_t3)
        plt.close(fig_t3)
    else:
        st.info("Tidak cukup kolom numerik untuk heatmap.")

with tab4:
    st.markdown("**Total Penyewaan per Musim**")
    season_cnt = (
        dff.groupby("season")["cnt"]
        .sum()
        .reindex(season_order)
        .dropna()
    )
    if not season_cnt.empty:
        fig_t4, ax_t4 = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax_t4.bar(season_cnt.index, season_cnt.values,
                  color=sns.color_palette("Set3", len(season_cnt)))
        ax_t4.set_title("Total Penyewaan per Musim")
        ax_t4.set_xlabel("Musim")
        ax_t4.set_ylabel("Total Penyewaan")
        st.pyplot(fig_t4)
        plt.close(fig_t4)
    else:
        st.info("Tidak ada data musim untuk rentang yang dipilih.")

st.markdown("---")

# ── ARIMA Forecast (trained on full df, displayed for dff context) ──────────────
st.subheader("📉 Forecast 6 Bulan ke Depan (ARIMA)")

try:
    # Train on full historical daily data for best model accuracy
    arima_series = df["cnt"].asfreq("D").fillna(method="ffill")
    model  = ARIMA(arima_series, order=(5, 1, 0))
    result = model.fit()

    forecast_steps = 180  # ~6 months
    forecast       = result.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(
        start=arima_series.index[-1] + pd.Timedelta(days=1),
        periods=forecast_steps,
        freq="D",
    )
    forecast_series = pd.Series(forecast.values, index=forecast_index)

    fig_ar, ax_ar = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax_ar.plot(dff.index, dff["cnt"], label="Data Aktual (filter)", color="#4C72B0")
    ax_ar.plot(forecast_series.index, forecast_series.values,
               label="Forecast ARIMA", color="#E07B54", linestyle="--")
    ax_ar.set_title("Forecast Penyewaan Sepeda (ARIMA 6 Bulan)")
    ax_ar.set_xlabel("Tanggal")
    ax_ar.set_ylabel("Jumlah Penyewaan")
    ax_ar.legend()
    plt.setp(ax_ar.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig_ar)
    plt.close(fig_ar)

    forecast_df = forecast_series.reset_index()
    forecast_df.columns = ["Tanggal", "Prediksi Penyewaan"]
    forecast_df["Prediksi Penyewaan"] = forecast_df["Prediksi Penyewaan"].round(0).astype(int)
    st.dataframe(forecast_df, use_container_width=True)

except Exception as e:
    st.error(f"ARIMA gagal: {e}")

st.markdown("---")
st.caption(
    "Sumber data: [UCI Machine Learning Repository — Bike Sharing Dataset]"
    "(https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) | "
    "Dashboard oleh tim analisis data."
)
