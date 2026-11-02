# ==========================================================
# RUANG BACA - CALM & MINIMALIST DISCOVERY DASHBOARD
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import gdown

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Ruang Baca",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CSS KUSTOM =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Quicksand:wght@300;400;500;600;700&display=swap');

/* Hapus elemen default Streamlit */
html, body, [class*="css"] { font-family: 'Quicksand', sans-serif; color: #4A5553; }
.stApp { background-color: #F9FAF8; }
[data-testid="stSidebar"] { background-color: #9CB4A1; color: white; }
[data-testid="stSidebar"] * { color: white !important; font-family: 'Quicksand', sans-serif; }

/* Paksa Search Bar Menjadi Putih & Elegan (Mengatasi Bug Dark Mode) */
[data-testid="stTextInput"] div[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border: 1px solid #9CB4A1 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 10px rgba(156, 180, 161, 0.1) !important;
}
[data-testid="stTextInput"] input {
    color: #4A5553 !important;
    background-color: #FFFFFF !important;
    font-family: 'Quicksand', sans-serif !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #B2C9AB !important;
}

/* Modifikasi Expander untuk Sinopsis agar menyatu dengan kartu */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #F0F2EF !important;
    border-radius: 0 0 12px 12px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.02) !important;
    margin-top: -15px; /* Menarik expander mendekat ke kartu */
}
[data-testid="stExpander"] summary {
    color: #9CB4A1 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    justify-content: center !important;
}
[data-testid="stExpander"] p {
    color: #8FA998 !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
}

/* Header & Typography */
.top-header { background: linear-gradient(135deg, #B2C9AB, #E8D7D0); padding: 25px 30px; border-radius: 16px; color: #4A5553; display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; box-shadow: 0 4px 20px rgba(156, 180, 161, 0.15); }
.header-title { font-family: 'Lora', serif; font-size: 28px; font-weight: 600; margin: 0; letter-spacing: 0.5px; }
.header-subtitle { font-size: 15px; font-weight: 400; opacity: 0.85; margin-top: 4px;}

.section-title { font-family: 'Lora', serif; font-size: 26px; font-weight: 600; color: #4A5553; text-align: center; margin-top: 40px; margin-bottom: 25px; }
.section-title::after { content: ''; display: block; width: 40px; height: 2px; background-color: #D4A373; margin: 10px auto 0; border-radius: 2px; }

/* Book Card Tanpa Tombol (Tombol dipindah ke Expander bawaan Streamlit) */
.book-card { background: white; border-radius: 12px 12px 0 0; padding: 20px 15px; border: 1px solid #F0F2EF; border-bottom: none; text-align: center; height: 100%; position: relative; }
.book-cover-container { height: 180px; display: flex; align-items: center; justify-content: center; margin-bottom: 18px; }
.book-cover { max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 4px; box-shadow: 2px 4px 10px rgba(0,0,0,0.08); }
.badge-tag { position: absolute; top: 12px; left: 12px; background: #E8D7D0; color: #4A5553; font-size: 10px; font-weight: 600; padding: 3px 10px; border-radius: 20px; letter-spacing: 0.5px;}
.book-title { font-family: 'Lora', serif; font-weight: 600; font-size: 15px; color: #4A5553; margin-bottom: 6px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; line-height: 1.3;}
.book-author { color: #8FA998; font-size: 12.5px; font-weight: 500; margin-bottom: 15px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* KPI Cards */
.kpi-card { background: white; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.02); border: 1px solid #F0F2EF; }
.kpi-icon { font-size: 24px; margin-bottom: 10px; color: #9CB4A1; }
.kpi-title { font-size: 13px; color: #8FA998; font-weight: 600; margin-bottom: 5px; letter-spacing: 0.5px;}
.kpi-value { font-family: 'Lora', serif; font-size: 28px; font-weight: 600; color: #4A5553; }
</style>
""", unsafe_allow_html=True)

# ===== LOAD & CLEAN DATA =====
@st.cache_data
def load_and_clean_data():
    dataset_dir = "book_dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        folder_url = "https://drive.google.com/drive/folders/1ZZxB1NdkjbVBEykDLN7rXWj9L4P2J3Sp?usp=sharing"
        gdown.download_folder(folder_url, output=dataset_dir, quiet=False, use_cookies=False)

    path_books = os.path.join(dataset_dir, "BX-Books.csv")
    path_users = os.path.join(dataset_dir, "BX-Users.csv")
    path_ratings = os.path.join(dataset_dir, "BX-Book-Ratings.csv")

    books = pd.read_csv(path_books, sep=";", encoding="latin-1", dtype={"ISBN": str}, on_bad_lines="skip")
    users = pd.read_csv(path_users, sep=";", encoding="latin-1", on_bad_lines="skip")
    ratings = pd.read_csv(path_ratings, sep=";", encoding="latin-1", dtype={"ISBN": str}, on_bad_lines="skip")

    books.columns = books.columns.str.strip()
    books = books.drop_duplicates(subset=["ISBN"]).dropna(subset=["ISBN"])
    books["ISBN"] = books["ISBN"].astype(str).str.strip()
    books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce")
    books.loc[~books["Year-Of-Publication"].between(1000, 2026), "Year-Of-Publication"] = np.nan
    books["Image-URL-M"] = books["Image-URL-M"].astype(str).str.replace("http://", "https://", regex=False)

    users.columns = users.columns.str.strip()
    users["Age"] = pd.to_numeric(users["Age"], errors="coerce")
    users.loc[~users["Age"].between(5, 100), "Age"] = np.nan

    ratings.columns = ratings.columns.str.strip()
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    ratings = ratings.dropna(subset=["User-ID", "ISBN", "Book-Rating"])
    ratings = ratings[ratings["Book-Rating"].between(0, 10)]
    ratings = ratings.drop_duplicates(subset=["User-ID", "ISBN"])

    ratings = ratings[ratings["ISBN"].isin(books["ISBN"]) & ratings["User-ID"].isin(users["User-ID"])]
    ratings_explicit = ratings[ratings["Book-Rating"] > 0].copy()
    
    return books, users, ratings, ratings_explicit

with st.spinner("Menyiapkan koleksi perpustakaan Ruang Baca..."):
    books_df, users_df, ratings_df, ratings_explicit = load_and_clean_data()

# ===== HYBRID RECOMMENDATION ENGINE =====
@st.cache_data
def build_hybrid_matrices(ratings_exp, books):
    counts = ratings_exp["ISBN"].value_counts()
    popular_isbn = counts[counts >= 50].index
    
    filtered_ratings = ratings_exp[ratings_exp["ISBN"].isin(popular_isbn)]
    pivot = filtered_ratings.pivot_table(index="ISBN", columns="User-ID", values="Book-Rating").fillna(0)
    cf_sim = cosine_similarity(pivot)
    cf_sim_df = pd.DataFrame(cf_sim, index=pivot.index, columns=pivot.index)
    
    pop_books = books[books["ISBN"].isin(popular_isbn)].copy()
    pop_books = pop_books.set_index("ISBN").reindex(popular_isbn)
    pop_books["content_features"] = pop_books["Book-Title"].fillna("") + " " + pop_books["Book-Author"].fillna("") + " " + pop_books["Publisher"].fillna("")
    
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(pop_books["content_features"])
    cb_sim = cosine_similarity(tfidf_matrix)
    cb_sim_df = pd.DataFrame(cb_sim, index=popular_isbn, columns=popular_isbn)
    
    return cf_sim_df, cb_sim_df

with st.spinner("Memproses rekomendasi buku (Mungkin memakan waktu 1-2 menit)..."):
    cf_sim_df, cb_sim_df = build_hybrid_matrices(ratings_explicit, books_df)

# ===== SIDEBAR NAVIGATION =====
with st.sidebar:
    st.markdown('<div style="text-align:center; font-size:45px; margin-bottom:5px;">🌿</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center; margin-top:0; font-family:\'Lora\', serif; font-weight:600;">Ruang Baca</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; font-size:13px; opacity:0.8; margin-bottom:30px;">Tempat menenangkan pikiran</p>', unsafe_allow_html=True)
    
    page = st.radio("Navigasi", ["🏠 Beranda", "📊 Estetika Data", "🎯 Kurator Personal"])
    st.markdown('<div style="margin-top: 50px; font-size:11px; text-align:center; opacity:0.7;">Disusun dengan AI Hybrid<br>2026</div>', unsafe_allow_html=True)

# ===== HELPER FUNCTION UNTUK BOOK CARD =====
def render_book_card(row, col, tag="BACA", show_sim=False, sim_score=0):
    cover_url = row["Image-URL-M"]
    if pd.isna(cover_url) or len(cover_url) < 10:
        cover_url = f"https://covers.openlibrary.org/b/isbn/{row['ISBN']}-M.jpg"
    
    info_html = f'<div style="font-size:11px; color:#D4A373; margin-bottom:10px; font-weight:600;">Cocok {sim_score*100:.0f}%</div>' if show_sim else f'<div style="font-size:11px; color:#A9A9A9; margin-bottom:10px;">Terbitan {row["Year-Of-Publication"]}</div>'
    
    # 1. Render Gambar dan Teks (HTML)
    col.markdown(f"""
    <div class="book-card">
        <div class="badge-tag">{tag}</div>
        <div class="book-cover-container">
            <img src="{cover_url}" class="book-cover" onerror="this.src='https://via.placeholder.com/130x200?text=Kover+Tidak+Tersedia'">
        </div>
        <div class="book-title" title="{row['Book-Title']}">{row['Book-Title']}</div>
        <div class="book-author">Oleh {row['Book-Author']}</div>
        {info_html}
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Render Expander Sinopsis (Komponen bawaan Streamlit agar bisa diklik)
    with col.expander("📖 Baca Sinopsis"):
        st.write(f"**{row['Book-Title']}**")
        st.write("Sayangnya, dataset perpustakaan tidak menyertakan sinopsis untuk buku ini. Namun, buku ini adalah salah satu karya luar biasa yang patut masuk ke dalam daftar bacaanmu.")

# ==========================================
# HALAMAN 1: BERANDA
# ==========================================
if page == "🏠 Beranda":
    st.markdown("""
    <div class="top-header">
        <div>
            <div class="header-title">Halo, Selamat Datang di Ruang Baca.</div>
            <div class="header-subtitle">Temukan buku yang menenangkan pikiranmu hari ini.</div>
        </div>
        <div style="font-size:24px; color:#4A5553;">🕊️</div>
    </div>
    """, unsafe_allow_html=True)

    search_q = st.text_input("", placeholder="🔍 Ketik judul buku atau penulis favoritmu di sini...", label_visibility="collapsed")
    if search_q:
        res = books_df[books_df["Book-Title"].str.contains(search_q, case=False, na=False) | books_df["Book-Author"].str.contains(search_q, case=False, na=False)].head(5)
        if not res.empty:
            st.success(f"Menemukan {len(res)} buku yang mungkin kamu cari 🤍")
            cols = st.columns(len(res))
            for i, (_, row) in enumerate(res.iterrows()):
                render_book_card(row, cols[i], tag="PILIHAN")

    st.markdown('<div class="section-title">Koleksi Terpopuler Bulan Ini</div>', unsafe_allow_html=True)
    
    top_isbn = ratings_explicit["ISBN"].value_counts().head(6).index
    best_sellers = books_df[books_df["ISBN"].isin(top_isbn)].drop_duplicates(subset=["ISBN"])
    
    cols = st.columns(6)
    for i, (_, row) in enumerate(best_sellers.head(6).iterrows()):
        render_book_card(row, cols[i], tag="TERPOPULER")
            
    st.markdown('<br><br>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Baca Sesuai Mood-mu</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    moods = ["Menenangkan Pikiran", "Petualangan Magis", "Pengembangan Diri", "Romansa Klasik"]
    icons = ["🍵", "✨", "🌱", "💌"]
    for col, mood, icon in zip([m1, m2, m3, m4], moods, icons):
        col.markdown(f"""
        <div style="text-align:center; padding:20px; background:white; border-radius:12px; border:1px solid #F0F2EF; box-shadow:0 4px 15px rgba(0,0,0,0.02); cursor:pointer;">
            <div style="font-size:35px; margin-bottom:10px;">{icon}</div>
            <div style="font-size:14px; font-weight:600; color:#4A5553;">{mood}</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# HALAMAN 2: ESTETIKA DATA
# ==========================================
elif page == "📊 Estetika Data":
    st.markdown('<div class="section-title">Harmoni Ekosistem Buku</div>', unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-icon">📚</div><div class="kpi-title">Koleksi Buku</div><div class="kpi-value">{books_df["ISBN"].nunique():,}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-icon">👥</div><div class="kpi-title">Pembaca Setia</div><div class="kpi-value">{users_df["User-ID"].nunique():,}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-icon">💌</div><div class="kpi-title">Total Ulasan</div><div class="kpi-value">{len(ratings_df):,}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><div class="kpi-icon">⭐</div><div class="kpi-title">Kepuasan Rata-rata</div><div class="kpi-value">{ratings_explicit["Book-Rating"].mean():.1f} / 10</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("<h4 style='text-align:center; font-family:\"Lora\", serif; color:#4A5553;'>Sebaran Apresiasi Pembaca</h4>", unsafe_allow_html=True)
        dist = ratings_explicit["Book-Rating"].value_counts().sort_index().reset_index()
        dist.columns = ["Rating", "Jumlah"]
        
        # PEMEBARUAN: Warna Plotly selaras dengan Sage Green
        fig1 = px.bar(dist, x="Rating", y="Jumlah", color_discrete_sequence=["#9CB4A1"])
        fig1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(dtick=1, showgrid=False, title="Skor Bintang"), 
            yaxis=dict(showgrid=True, gridcolor="#F0F2EF", title="Jumlah Ulasan"), 
            font=dict(family="Quicksand", color="#4A5553")
        )
        fig1.update_traces(marker_line_width=0, opacity=0.9, hovertemplate="<b>Skor: %{x}</b><br>Jumlah: %{y}<extra></extra>")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_chart2:
        st.markdown("<h4 style='text-align:center; font-family:\"Lora\", serif; color:#4A5553;'>Demografi Usia Pembaca</h4>", unsafe_allow_html=True)
        
        # PEMEBARUAN: Warna Plotly selaras dengan Earthy Peach
        fig2 = px.histogram(users_df.dropna(subset=["Age"]), x="Age", nbins=25, color_discrete_sequence=["#D4A373"])
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
            xaxis=dict(showgrid=False, title="Usia"), 
            yaxis=dict(showgrid=True, gridcolor="#F0F2EF", title="Jumlah Pembaca"), 
            font=dict(family="Quicksand", color="#4A5553")
        )
        fig2.update_traces(marker_line_width=0, opacity=0.85, hovertemplate="<b>Rentang Usia: %{x}</b><br>Jumlah: %{y}<extra></extra>")
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# HALAMAN 3: AI KURATOR PERSONAL
# ==========================================
elif page == "🎯 Kurator Personal":
    st.markdown('<div class="section-title">Kurasi Spesial Untukmu</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#8FA998; margin-bottom:30px;'>Beritahu kami satu buku yang kamu suka, dan mesin kognitif kami akan meracik rekomendasi yang senada.</p>", unsafe_allow_html=True)
    
    valid_books = books_df[books_df["ISBN"].isin(cf_sim_df.index)]
    book_options = valid_books["Book-Title"].dropna().unique()
    
    selected_title = st.selectbox("Pilih buku referensimu:", sorted(book_options), placeholder="Ketik judul buku...")
    
    if selected_title:
        ref_isbn = valid_books[valid_books["Book-Title"] == selected_title]["ISBN"].values[0]
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight:600; color:#4A5553; font-family:\"Lora\", serif;'>Atur Karakter Rekomendasi (Hybrid AI)</p>", unsafe_allow_html=True)
        alpha = st.slider(
            "Geser ke kiri = Mirip Tema & Penulis (Content) | Geser ke kanan = Mirip Selera Pembaca Lain (Collaborative)",
            min_value=0.0, max_value=1.0, value=0.6, step=0.1, label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("<h3 style='font-family:\"Lora\", serif; color:#4A5553;'>✨ Bacaan yang Mungkin Mengisi Hatimu:</h3><br>", unsafe_allow_html=True)
        
        cf_scores = cf_sim_df[ref_isbn]
        cb_scores = cb_sim_df[ref_isbn]
        hybrid_scores = (alpha * cf_scores) + ((1 - alpha) * cb_scores)
        
        top_similar_isbns = hybrid_scores.sort_values(ascending=False).iloc[1:7].index
        reco_books = books_df[books_df["ISBN"].isin(top_similar_isbns)].drop_duplicates(subset=["ISBN"])
        
        cols = st.columns(6)
        for i, (isbn) in enumerate(top_similar_isbns):
            book_row = reco_books[reco_books["ISBN"] == isbn]
            if not book_row.empty:
                score = hybrid_scores[isbn]
                render_book_card(book_row.iloc[0], cols[i], tag="REKOMENDASI", show_sim=True, sim_score=score)
