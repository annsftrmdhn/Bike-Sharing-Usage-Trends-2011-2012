# Bike Sharing Data Analysis Dashboard

## Deskripsi Proyek
Proyek ini bertujuan untuk menganalisis data penyewaan sepeda selama tahun 2011–2012 menggunakan dataset dari UCI Bike Sharing Dataset. Analisis dilakukan untuk memahami tren penyewaan sepeda berdasarkan waktu (bulanan dan per jam), pengaruh musim dan kondisi cuaca, serta memprediksi tren penyewaan di masa depan.

Dashboard interaktif dibuat menggunakan Streamlit untuk memudahkan eksplorasi data dan visualisasi hasil analisis secara real-time.

## Link Dashboard
https://bike-sharing-usage-trends-2011-2012-nzdrvueqkaxvvumdrkskzm.streamlit.app/


## Dataset
- `day.csv` dan `hour.csv`: Dataset asli berisi data harian dan per jam.
- `day_cleaned.csv` dan `hour_cleaned.csv`: Dataset hasil pembersihan dan preprocessing.
- `main_data.csv` (opsional): Dataset gabungan hasil agregasi untuk analisis harian.

## Fitur Dashboard
- Menampilkan metrik utama seperti total penyewaan, rata-rata harian, dan hari dengan penyewaan tertinggi.
- Visualisasi tren penyewaan sepeda per bulan.
- Prediksi penyewaan sepeda untuk 6 bulan ke depan menggunakan model ARIMA.
- Analisis penyewaan berdasarkan musim dan kondisi cuaca.

## Cara Menjalankan
1. Pastikan Python sudah terinstall di komputer Anda.
2. Install dependencies dengan menjalankan:
``pip install -r requirements.txt``
3. Jalankan dashboard dengan perintah:
``streamlit run dashboard.py``
4. Buka browser dan akses URL yang muncul (biasanya http://localhost:8501).

---

Terima kasih telah menggunakan dashboard ini untuk analisis data penyewaan sepeda!
