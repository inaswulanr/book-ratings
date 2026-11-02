# 🌿 Ruang Baca — Book Discovery & Analytics

> *Tempat menenangkan pikiran melalui harmoni data dan sastra.*

**Ruang Baca** adalah platform penemuan buku (*book discovery*) dan dashboard analitik interaktif berbasis **Streamlit**. Mengusung estetika editorial puitis **Tyrian Purple × Citron × Cream** (`#66023C` × `#CAD183` × `#FFFDD0`) dengan dukungan latar belakang kustom berlapis *glassmorphism*, aplikasi ini memadukan algoritma *Machine Learning* untuk menyajikan rekomendasi bacaan yang personal dan menenangkan.

---

## ✨ Fitur Utama

### 1. 🏠 Beranda & Pencarian Pintar
* **Instant Search:** Pencarian real-time berdasarkan judul buku atau nama penulis dengan tampilan kartu kover berbayang *offset* klasik.
* **Koleksi Terpopuler:** Kurasi otomatis buku-buku dengan apresiasi komunitas tertinggi.
* **Baca Sesuai Mood (*Mood-Based Filtering*):** Kurasi bacaan instan berdasarkan 4 suasana hati (*Menenangkan Pikiran*, *Petualangan Magis*, *Pengembangan Diri*, dan *Romansa Klasik*).

### 2. 📊 Estetika Data (Analitik Ekosistem)
* **Indikator KPI Utama:** Ringkasan metrik koleksi buku, total pembaca setia, jumlah ulasan, dan rata-rata kepuasan komunitas.
* **Tren Publikasi (*Time-Series*):** Grafik area interaktif yang memetakan pergerakan rilis literatur dari tahun 1970 hingga 2026.
* **Sebaran Apresiasi & Demografi Usia:** Distribusi skor rating eksplisit (1–10) dan profil usia pembaca.
* **Segmentasi Persona Pembaca (*K-Means Clustering*):** Pengelompokan otomatis pembaca menggunakan algoritma *Unsupervised Learning* (`StandardScaler` + `KMeans`) ke dalam 3 persona: *Kutu Buku Aktif*, *Kritikus Selektif*, dan *Pembaca Santai*.

### 3. 🎯 Kurator Personal (Hybrid AI Engine)
* **Hybrid Filtering:** Menggabungkan matriks **Collaborative Filtering** (pola selera pembaca serupa) dan **Content-Based Filtering** (kemiripan teks judul, penulis, dan penerbit via `TF-IDF` & `Cosine Similarity`).
* **Kontrol Karakter Rekomendasi:** *Slider* bobot $\alpha$ interaktif untuk mengatur keseimbangan antara kemiripan tema vs. selera komunitas, dilengkapi pilihan jumlah tampilan (3, 6, atau 12 buku).
* **Cold-Start Protocol:** Sistem pengaman otomatis yang mendeteksi buku langka/minim ulasan dan mengalihkannya ke daftar kurasi terbaik tanpa memicu *error*.
