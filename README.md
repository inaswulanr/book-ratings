# 📚 Book Insight AI — ADL Dashboard

Dashboard Streamlit interaktif untuk **Book-Crossing Dataset** dengan **Hybrid Recommendation System** (Content-Based + Item-Based Collaborative Filtering). Dibangun dari notebook `adl_dashboard_1.py` dengan UI modern bergaya *Anime Insight AI*.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Fitur Utama

| Halaman | Deskripsi |
|---|---|
| 🏠 **Overview** | Ringkasan KPI global (buku, pengguna, rating), pencarian cepat, dan insight otomatis |
| 📊 **Analytics** | Distribusi rating, top buku, genre penulis, heatmap korelasi, top publisher |
| 📖 **Book Explorer** | Jelajahi katalog buku dengan filter penulis, tahun, dan rating minimum |
| 👥 **User Analytics** | Distribusi usia, negara, top active users, gender/lokasi |
| 🎯 **Recommendations** | Rekomendasi hybrid per pengguna (Weighted Hybrid / Switching Hybrid) |
| 🧠 **AI Insights** | Tren, top penulis, analisis dominasi penerbit, dan kesimpulan strategis |
| ❤️ **Favorites** | Simpan buku favorit selama sesi berlangsung |
| 📖 **User Guide** | Panduan penggunaan lengkap |
| ⚙️ **Settings** | Atur α (bobot CF vs CB), tema, dan parameter dashboard |

---

## 🧠 Arsitektur Model Rekomendasi

Sistem mengimplementasikan **hybrid recommendation** sesuai Bab B di notebook ADL:

### 1. Content-Based Filtering (CB)
- **Item profile**: TF-IDF judul (1-gram + 2-gram) + one-hot penulis + one-hot penerbit
- **Bobot fitur**: `W_TITLE = 1.0`, `W_AUTHOR = 1.0`, `W_PUBLISHER = 0.3`
- **User profile**: bobot rating di atas rata-rata pengguna
- **Similarity**: cosine similarity antar-vektor item

### 2. Item-Based Collaborative Filtering (CF)
- **Similarity**: cosine pada pola "dirating bersama" (0/1)
- **KNN**: 20 tetangga terdekat per buku (menekan noise)
- **Matriks**: sparse user×book dari rating eksplisit 1–10

### 3. Weighted Hybrid (pengguna inti)