# ==========================================================
# RUANG BACA - ADVANCED DISCOVERY & ANALYTICS DASHBOARD
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import gdown
import time

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

html, body, [class*="css"] { font-family: 'Quicksand', sans-serif; color: #4A5553; }
.stApp { background-color: #F9FAF8; }
[data-testid="stSidebar"] { background-color: #9CB4A1; color: white; border-right: none; }
[data-testid="stSidebar"] * { color: white !important; font-family: 'Quicksand', sans-serif; }

/* Pencarian */
[data-testid="stTextInput"] div[data-baseweb="input"] { background-color: #FFFFFF !important; border: 1px solid #9CB4A1 !important; border-radius: 8px !important; box-shadow: 0 2px 10px rgba(156, 180, 161, 0.1) !important; }
[data-testid="stTextInput"] input { color: #4A5553 !important; font-family: 'Quicksand', sans-serif !important; }
[data-testid="stTextInput"] input::placeholder { color: #B2C9AB !important; }

/* Slider Kustom */
div[data-baseweb="slider"] { padding-top: 10px; }
div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] { display: none !important; }

/* Expander */
[data-testid="stExpander"] { background-color: #FFFFFF !important; border: 1px solid #F0F2EF !important; border-radius: 0 0 12px 12px !important; box-shadow: 0 4px 15px rgba(0,0,0,0.02) !important; margin-top: -15px; }
[data-testid="stExpander"] summary { color: #9CB4A1 !important; font-weight: 600 !important; font-size: 13px !important; justify-content: center !important; }

/* Typography */
.top-header { background: linear-gradient(135deg, #B2C9AB, #E8D7D0); padding: 25px 30px; border-radius: 16px; color: #4A5553; display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; box-shadow: 0 4px 20px rgba(156, 180, 161, 0.15); }
.header-title { font-family: 'Lora', serif; font-size: 28px; font-weight: 600; margin: 0; letter-spacing: 0.5px; }
.header-subtitle { font-size: 15px; font-weight: 400; opacity: 0.85; margin-top: 4px;}
.section-title { font-family: 'Lora', serif; font-size: 26px; font-weight: 600; color: #4A5553; text-align: center; margin-top: 40px; margin-bottom: 25px; }
.section-title::after { content: ''; display: block; width: 40px; height: 2px; background-color: #D4A373; margin: 10px auto 0; border-radius: 2px; }

/* Cards */
.book-card { background: white; border-radius: 12px 12px 0 0; padding: 20px 15px; border: 1px solid #F0F2EF; border-bottom: none; text-align: center; height: 100%; position: relative; transition: 0.3s; }
.book-card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(156, 180, 161, 0.12); border-color: #9CB4A1; }
.book-cover-container { height: 180px; display: flex; align-items: center; justify-content: center; margin-bottom: 18px; }
.book-cover { max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 4px; box-shadow: 2px 4px 10px rgba(0,0,0,0.08); }
.badge-tag { position: absolute; top: 12px; left: 12px; background: #E8D7D0; color: #4A5553; font-size: 10px; font-weight: 700; padding: 4px 10px; border-radius: 20px; letter-spacing: 0.5px;}
.badge-sim { position: absolute; top: 12px; right: 12px; background: #D4A373; color: white; font-size: 10px; font-weight: 700; padding: 4px 8px; border-radius: 8px;}
.book-title { font-family: 'Lora', serif; font-weight: 600; font-size: 15px; color: #4A5553; margin-bottom: 6px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; line-height: 1.3;}
.book-author { color: #8FA998; font-size: 12.5px; font-weight: 500; margin-bottom: 15px; }

.kpi-card { background: white; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.02); border: 1px solid #F0F2EF; transition: 0.3s; }
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 15px rgba(156, 180, 161, 0.1); }
.kpi-icon { font-size: 26px; margin-bottom: 10px; color: #9CB4A1; }
.kpi-title { font-size: 13px; color: #8FA998; font-weight: 600; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;}
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

    books = pd.read_csv(os.path.join(dataset_dir, "BX-Books.csv"), sep=";", encoding="latin-1", dtype={"ISBN": str}, on_bad_lines="skip")
    users = pd.read_csv(os.path.join(dataset_dir, "BX-Users.csv"), sep=";", encoding="latin-1", on_bad_lines="skip")
    ratings = pd.read_csv(os.path.join(dataset_dir, "BX-Book-Ratings.csv"), sep=";", encoding="latin-1", dtype={"ISBN": str}, on_bad_lines="skip")

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
    ratings = ratings.drop_duplicates(subset=["User-ID", "ISBN"])

    ratings = ratings[ratings["ISBN"].isin(books["ISBN"]) & ratings["User-ID"].isin(users["User-ID"])]
    ratings_explicit = ratings[ratings["Book-Rating"] > 0].copy()
    
    return books, users, ratings, ratings_explicit

with st.spinner("Menyiapkan koleksi perpustakaan Ruang Baca..."):
    books_df, users_df, ratings_df, ratings_explicit = load_and_clean_data()

# ===== ML: HYBRID ENGINE & CLUSTERING =====
@st.cache_data
def train_machine_learning(ratings_exp, books, users):
    # 1. Algoritma Hybrid (Rekomendasi)
    counts = ratings_exp["ISBN"].value_counts()
    popular_isbn = counts[counts >= 40].index
    
    filtered_ratings = ratings_exp[ratings_exp["ISBN"].isin(popular_isbn)]
    pivot = filtered_ratings.pivot_table(index="ISBN", columns="User-ID", values="Book-Rating").fillna(0)
    cf_sim_df = pd.DataFrame(cosine_similarity(pivot), index=pivot.index, columns=pivot.index)
    
    pop_books = books[books["ISBN"].isin(popular_isbn)].set_index("ISBN")
    pop_books["content_features"] = pop_books["Book-Title"].fillna("") + " " + pop_books["Book-Author"].fillna("") + " " + pop_books["Publisher"].fillna("")
    cb_sim_df = pd.DataFrame(cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(pop_books["content_features"])), index=popular_isbn, columns=popular_isbn)
    
    # 2. Algoritma K-Means (Segmentasi Pembaca)
    user_stats = ratings_exp.groupby('User-ID').agg(Total_Baca=('Book-Rating', 'count'), Rata_Rata_Rating=('Book-Rating', 'mean')).reset_index()
    user_data = pd.merge(user_stats, users[['User-ID', 'Age']].dropna(), on='User-ID')
    
    if len(user_data) > 100:
        X = StandardScaler().fit_transform(user_data[['Total_Baca', 'Rata_Rata_Rating', 'Age']])
        user_data['Persona'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
        persona_map = {0: "Kutu Buku Aktif", 1: "Kritikus Selektif", 2: "Pembaca Santai"}
        user_data['Persona'] = user_data['Persona'].map(persona_map)
    else:
        user_data['Persona'] = "Pembaca Umum"
        
    return cf_sim_df, cb_sim_df, user_data

with st.spinner("Memproses mesin kecerdasan buatan (Mungkin memakan waktu sesaat)..."):
    cf_sim_df, cb_sim_df, user_clusters = train_machine_learning(ratings_explicit, books_df, users_df)

# ===== SIDEBAR NAVIGATION =====
with st.sidebar:
    st.markdown('<div style="text-align:center; font-size:45px; margin-bottom:5px;">🌿</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center; margin-top:0; font-family:\'Lora\', serif; font-weight:600;">Ruang Baca</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; font-size:13px; opacity:0.8; margin-bottom:30px;">Tempat menenangkan pikiran</p>', unsafe_allow_html=True)
    
    page = st.radio("Navigasi", ["🏠 Beranda", "📊 Estetika Data", "🎯 Kurator Personal"])
    
    if page == "🎯 Kurator Personal":
        st.markdown("<br><hr style='border: 0.5px solid rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:12px; font-weight:600;'>Jumlah Rekomendasi:</p>", unsafe_allow_html=True)
        n_reco = st.select_slider("", options=[3, 6, 12], value=6, label_visibility="collapsed")
        
    st.markdown('<div style="margin-top: 60px; font-size:11px; text-align:center; opacity:0.6;">Disusun dengan AI Hybrid<br>2026</div>', unsafe_allow_html=True)

# ===== HELPER FUNCTION UNTUK BOOK CARD =====
def render_book_card(row, col, tag="BACA", show_sim=False, sim_score=0):
    cover_url = row["Image-URL-M"] if pd.notna(row["Image-URL-M"]) and len(row["Image-URL-M"]) > 10 else f"https://covers.openlibrary.org/b/isbn/{row['ISBN']}-M.jpg"
    badge_sim = f'<div class="badge-sim">{sim_score*100:.0f}% Sesuai</div>' if show_sim else ''
    
    col.markdown(f"""
    <div class="book-card">
        <div class="badge-tag">{tag}</div>
        {badge_sim}
        <div class="book-cover-container">
            <img src="{cover_url}" class="book-cover" onerror="this.src='https://via.placeholder.com/130x200?text=Kover+Tidak+Tersedia'">
        </div>
        <div class="book-title" title="{row['Book-Title']}">{row['Book-Title']}</div>
        <div class="book-author">Oleh {row['Book-Author']}</div>
        <div style="font-size:11px; color:#A9A9A9; margin-bottom:5px;">Terbitan {row["Year-Of-Publication"]}</div>
    </div>
    """, unsafe_allow_html=True)
    
    with col.expander("📖 Baca Sinopsis"):
        st.write(f"**{row['Book-Title']}**")
        st.write("Buku ini adalah salah satu karya luar biasa yang direkomendasikan oleh komunitas pembaca kami. Sebuah literatur yang patut masuk ke dalam daftar bacaanmu akhir pekan ini.")

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
        <div style="font-size:26px; color:#4A5553;">🕊️</div>
    </div>
    """, unsafe_allow_html=True)

    search_q = st.text_input("", placeholder="🔍 Cari buku atau nama penulis di sini...", label_visibility="collapsed")
    if search_q:
        res = books_df[books_df["Book-Title"].str.contains(search_q, case=False, na=False) | books_df["Book-Author"].str.contains(search_q, case=False, na=False)].head(6)
        if not res.empty:
            st.success(f"Ditemukan {len(res)} buku yang mungkin sesuai 🤍")
            cols = st.columns(len(res))
            for i, (_, row) in enumerate(res.iterrows()):
                render_book_card(row, cols[i], tag="PILIHAN")
        else:
            st.warning("Buku tidak ditemukan di koleksi kami. Coba kata kunci lain.")

    st.markdown('<div class="section-title">Koleksi Terpopuler Bulan Ini</div>', unsafe_allow_html=True)
    top_isbn = ratings_explicit["ISBN"].value_counts().head(6).index
    best_sellers = books_df[books_df["ISBN"].isin(top_isbn)].drop_duplicates(subset=["ISBN"])
    
    cols = st.columns(6)
    for i, (_, row) in enumerate(best_sellers.iterrows()):
        render_book_card(row, cols[i], tag="TERPOPULER")

# ==========================================
# HALAMAN 2: ESTETIKA DATA (Analitik Lanjutan)
# ==========================================
elif page == "📊 Estetika Data":
    st.markdown('<div class="section-title">Harmoni Ekosistem Buku</div>', unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-icon">📚</div><div class="kpi-title">Koleksi Buku</div><div class="kpi-value">{books_df["ISBN"].nunique():,}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-icon">👥</div><div class="kpi-title">Pembaca Setia</div><div class="kpi-value">{users_df["User-ID"].nunique():,}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-icon">💌</div><div class="kpi-title">Total Ulasan</div><div class="kpi-value">{len(ratings_df):,}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><div class="kpi-icon">⭐</div><div class="kpi-title">Kepuasan Rata-rata</div><div class="kpi-value">{ratings_explicit["Book-Rating"].mean():.1f} / 10</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tren Publikasi (Time Series)
    st.markdown("<h4 style='text-align:center; font-family:\"Lora\", serif; color:#4A5553;'>Tren Tahun Publikasi Buku</h4>", unsafe_allow_html=True)
    yearly_pubs = books_df[(books_df['Year-Of-Publication'] >= 1970) & (books_df['Year-Of-Publication'] <= 2026)]['Year-Of-Publication'].value_counts().sort_index().reset_index()
    yearly_pubs.columns = ['Tahun', 'Jumlah Buku Dirilis']
    fig_line = px.area(yearly_pubs, x="Tahun", y="Jumlah Buku Dirilis", color_discrete_sequence=["#9CB4A1"])
    fig_line.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#F0F2EF"), font=dict(family="Quicksand", color="#4A5553"), height=300)
    fig_line.update_traces(fillcolor='rgba(156, 180, 161, 0.3)', line=dict(width=3))
    st.plotly_chart(fig_line, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<h4 style='text-align:center; font-family:\"Lora\", serif; color:#4A5553;'>Segmentasi Persona Pembaca (AI K-Means)</h4>", unsafe_allow_html=True)
        persona_counts = user_clusters['Persona'].value_counts().reset_index()
        fig_pie = px.pie(persona_counts, values='count', names='Persona', hole=0.5, color_discrete_sequence=["#9CB4A1", "#D4A373", "#E8D7D0"])
        fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Quicksand", color="#4A5553"), showlegend=True)
        fig_pie.update_traces(textinfo='percent+label', textposition='inside')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.markdown("<h4 style='text-align:center; font-family:\"Lora\", serif; color:#4A5553;'>Distribusi Usia Pembaca</h4>", unsafe_allow_html=True)
        fig_hist = px.histogram(users_df.dropna(subset=["Age"]), x="Age", nbins=25, color_discrete_sequence=["#E8D7D0"])
        fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False, title="Usia"), yaxis=dict(showgrid=True, gridcolor="#F0F2EF", title="Jumlah"), font=dict(family="Quicksand", color="#4A5553"))
        fig_hist.update_traces(marker_line_width=0, opacity=0.9)
        st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# HALAMAN 3: AI KURATOR (Cold-Start Protocol)
# ==========================================
elif page == "🎯 Kurator Personal":
    st.markdown('<div class="section-title">Kurasi Spesial Untukmu</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#8FA998; margin-bottom:30px;'>Beritahu kami satu buku yang kamu suka, dan mesin kognitif kami akan meracik rekomendasi yang senada.</p>", unsafe_allow_html=True)
    
    # Ambil SEMUA judul buku untuk dropdown, bukan cuma buku populer (Mendukung Cold Start)
    all_books = books_df["Book-Title"].dropna().unique()
    selected_title = st.selectbox("Pilih buku referensimu:", sorted(all_books), placeholder="Ketik judul buku...")
    
    if selected_title:
        ref_isbn = books_df[books_df["Book-Title"] == selected_title]["ISBN"].values[0]
        
        # Cek apakah buku memenuhi syarat masuk ke mesin AI (Minimal 40 ulasan)
        if ref_isbn not in cf_sim_df.index:
            # ---> COLD-START PROTOCOL TERPICU <---
            st.warning("🌿 Buku yang kamu pilih adalah permata tersembunyi! Ulasannya di komunitas kami masih sangat sedikit, sehingga AI belum bisa mengenali polanya dengan akurat. Sebagai gantinya, ini adalah mahakarya terbaik yang disukai semua orang:")
            
            top_isbn = ratings_explicit["ISBN"].value_counts().head(n_reco).index
            reco_books = books_df[books_df["ISBN"].isin(top_isbn)].drop_duplicates(subset=["ISBN"])
            
            cols = st.columns(6)
            for i, (_, row) in enumerate(reco_books.iterrows()):
                render_book_card(row, cols[i % 6], tag="BEST SELLER")
        else:
            # ---> NORMAL HYBRID PROTOCOL <---
            st.markdown("<br><p style='font-weight:600; color:#4A5553; font-size:14px; text-align:center;'>Atur Karakter Rekomendasi AI</p>", unsafe_allow_html=True)
            col_sl1, col_sl2, col_sl3 = st.columns([1, 2, 1])
            with col_sl2:
                alpha = st.slider(
                    "Geser ke kiri (Mirip Tema) ↔ Geser ke kanan (Mirip Selera Pembaca Lain)",
                    min_value=0.0, max_value=1.0, value=0.6, step=0.1, label_visibility="collapsed"
                )
            
            st.markdown("---")
            with st.status("Meracik rekomendasi untukmu...", expanded=True) as status:
                time.sleep(0.8) # Animasi natural UX
                cf_scores = cf_sim_df[ref_isbn]
                cb_scores = cb_sim_df[ref_isbn]
                hybrid_scores = (alpha * cf_scores) + ((1 - alpha) * cb_scores)
                
                top_similar_isbns = hybrid_scores.sort_values(ascending=False).iloc[1:n_reco+1].index
                reco_books = books_df[books_df["ISBN"].isin(top_similar_isbns)].drop_duplicates(subset=["ISBN"])
                status.update(label="Rekomendasi berhasil dimuat! ✨", state="complete", expanded=False)
            
            # Tampilkan Grid secara dinamis (max 6 per baris)
            for row_start in range(0, len(reco_books), 6):
                cols = st.columns(6)
                chunk = reco_books.iloc[row_start:row_start+6]
                for i, (_, row) in enumerate(chunk.iterrows()):
                    score = hybrid_scores[row['ISBN']]
                    render_book_card(row, cols[i], tag="REKOMENDASI", show_sim=True, sim_score=score)
