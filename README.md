# 🚲 Bike Sharing Usage Trends Dashboard (2011–2012)

Dashboard interaktif untuk menganalisis tren penggunaan layanan bike sharing selama periode 2011–2012.

## 📊 Pertanyaan Bisnis

1. Bagaimana tren pertumbuhan total peminjaman sepeda setiap bulan selama periode Januari 2011 hingga Desember 2012, dan berapa rata-rata growth rate bulanannya?
2. Bagaimana pola rata-rata peminjaman sepeda per jam pada hari kerja dibandingkan hari libur selama periode 2011–2012, dan pada jam berapa puncak peminjaman tertinggi terjadi di masing-masing kondisi tersebut?
3. Musim apa yang memiliki rata-rata peminjaman sepeda harian tertinggi dan terendah selama periode 2011–2012, dan berapa selisih rata-rata peminjaman antara musim tersebut?

## 📁 Struktur Proyek

```
Bike-Sharing-Usage-Trends-2011-2012/
├── Dashboard/
│   ├── dashboard.py
│   └── main_data.csv
├── data/
│   ├── day.csv
│   └── hour.csv
├── notebook.ipynb
├── requirements.txt
└── README.md
```

## 🛠️ Setup Environment

### Menggunakan Anaconda

```bash
conda create --name bike-sharing python=3.10
conda activate bike-sharing
pip install -r requirements.txt
```

### Menggunakan Virtual Environment (venv)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## 📦 Install Library

Pastikan sudah berada di dalam environment yang aktif, lalu jalankan:

```bash
pip install -r requirements.txt
```

Isi `requirements.txt`:

```
streamlit
pandas
matplotlib
statsmodels
```

## ▶️ Menjalankan Dashboard Secara Lokal

```bash
cd Dashboard
streamlit run dashboard.py
```

Dashboard akan otomatis terbuka di browser pada alamat `http://localhost:8501`

## 🌐 Streamlit Cloud

Dashboard juga tersedia secara online di:
👉 **[Link Dashboard]([https://share.streamlit.io](https://bike-sharing-usage-trends-2011-2012-z6bgr9gcrmpyuifuzm6mou.streamlit.app/))**

## 📂 Sumber Data

Dataset yang digunakan adalah [Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset) yang berisi data penyewaan sepeda harian dan per jam selama tahun 2011–2012.
