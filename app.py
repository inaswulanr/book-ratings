# ==========================================================
# BOOK STORE DASHBOARD - PERIPLUS STYLE (WITH CUSTOM BACKGROUND)
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
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CSS KUSTOM (Dengan Background Pastel & Efek Glassmorphism) =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Roboto', sans-serif; color: #333333; }

/* Background Website Menggunakan Gambar dari GitHub */
.stApp { 
    background-image: url("background.jpg"); /* Ganti dengan nama file gambarmu jika berbeda, misal background.png */
    background-size: cover;        
    background-repeat: no-repeat;  
    background-attachment: fixed;  
}

[data-testid="collapsedControl"] { display: none; }
[data-testid="stSidebar"] { display: none; }

/* Top Header Bar (Semi-transparan elegan) */
.periplus-header {
    background-color: rgba(178, 34, 34, 0.9);
    padding: 15px 30px;
    margin: -40px -40px 20px -40px;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
.header-logo { font-size: 28px; font-weight: 700; margin: 0; letter-spacing: 1px; }
.header-tagline { font-size: 12px; font-weight: 300; margin-top: -5px; }

/* Sub Navigasi */
.sub-nav {
    background-color: rgba(255, 255, 255, 0.9);
    margin: -20px -40px 30px -40px;
    padding: 10px 30px;
    border-bottom: 1px solid #E0E0E0;
    display: flex;
    gap: 20px;
    font-size: 14px;
    font-weight: 500;
    color: #333;
}

/* Section Title */
.section-title { font-size: 24px; font-weight: 700; color: #222; text-align: center; margin-top: 30px; margin-bottom: 25px; position: relative;}
.section-title::after { content: ''; display: block; width: 40px; height: 2px; background-color: #B22222; margin: 10px auto 0; }

/* Cards (Efek Kaca Putih Bersih di atas Background Garis) */
.book-card { 
    background: rgba(255, 255, 255, 0.92); 
    backdrop-filter: blur(5px);
    border-radius: 6px; 
    padding: 15px; 
    text-align: left; 
    height: 100%; 
    position: relative; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    border: 1px solid rgba(255,255,255,0.8);
    transition: 0.2s;
}
.book-card:hover { transform: translateY(-4px); box-shadow: 0 8px 25px rgba(0,0,0,0.12); }
.book-cover-container { height: 180px; display: flex; align-items: center; justify-content: center; margin-bottom: 15px; }
.book-cover { max-height: 100%; max-width: 100%; object-fit: contain; }

/* Badge & Typography */
.badge-delivery { background: #B22222; color: white; font-size: 10px; font-weight: 600; padding: 3px 8px; border-radius: 10px; display: inline-block; margin-bottom: 8px;}
.badge-sim { background: #D4A373; color: white; font-size: 10px; font-weight: 700; padding: 3px 6px; border-radius: 6px; float: right; }
.book-title { font-weight: 700; font-size: 14px; color: #222; margin-bottom: 4px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; line-height: 1.3;}
.book-author { color: #5C9EAD; font-size: 12px; margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* Expander Detail */
[data-testid="stExpander"] { background-color: rgba(255,255,255,0.85) !important; border: none !important; box-shadow: none !important; margin-top: 10px; border-radius: 6px !important; }
[data-testid="stExpander"] summary { color: #B22222 !important; font-weight: 500 !important; font-size: 12px !important; justify-content: center !important; }

/* KPI Cards */
.kpi-card { background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(5px); border-radius: 6px; padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid rgba(255,255,255,0.8); }
.kpi-icon { font-size: 24px; margin-bottom: 8px; color: #B22222; }
.kpi-title { font-size: 12px; color: #555; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;}
.kpi-value { font-size: 24px; font-weight: 700; color: #222; margin-top: 5px; }
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

with st.spinner("Memuat katalog buku..."):
    books_df, users_df, ratings_df, ratings_explicit = load_and_clean_data()

# ===== ML: HYBRID ENGINE & CLUSTERING =====
@st.cache_data
def train_machine_learning(ratings_exp, books, users):
    counts = ratings_exp["ISBN"].value_counts()
    popular_isbn = counts[counts >= 40].index
    
    filtered_ratings = ratings_exp[ratings_exp["ISBN"].isin(popular_isbn)]
    pivot = filtered_ratings.pivot_table(index="ISBN", columns="User-ID", values="Book-Rating").fillna(0)
    cf_sim_df = pd.DataFrame(cosine_similarity(pivot), index=pivot.index, columns=pivot.index)
    
    pop_books = books[books["ISBN"].isin(popular_isbn)].set_index("ISBN")
    pop_books["content_features"] = pop_books["Book-Title"].fillna("") + " " + pop_books["Book-Author"].fillna("") + " " + pop_books["Publisher"].fillna("")
    cb_sim_df = pd.DataFrame(cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(pop_books["content_features"])), index=popular_isbn, columns=popular_isbn)
    
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

with st.spinner("Memproses mesin kecerdasan buatan..."):
    cf_sim_df, cb_sim_df, user_clusters = train_machine_learning(ratings_explicit, books_df, users_df)


# ===== HEADER UI =====
st.markdown("""
<div class="periplus-header">
    <div>
        <div class="header-logo">BOOKSTORE.COM</div>
        <div class="header-tagline">Indonesia's largest book discovery platform</div>
    </div>
    <div style="display:flex; gap:20px; font-size:20px;">
        🔍 &nbsp; 🤍 &nbsp; 👤
    </div>
</div>
<div class="sub-nav">
    <span class="sub-nav-item">🏠 Beranda</span>
    <span class="sub-nav-item">📊 Estetika Data</span>
    <span class="sub-nav-item">🎯 Kurator Personal AI</span>
</div>
""", unsafe_allow_html=True)


# ===== HELPER FUNCTION UNTUK BOOK CARD =====
def render_book_card(row, col, status="normal", show_sim=False, sim_score=0):
    cover_url = row["Image-URL-M"] if pd.notna(row["Image-URL-M"]) and len(row["Image-URL-M"]) > 10 else f"https://covers.openlibrary.org/b/isbn/{row['ISBN']}-M.jpg"
    
    badge_html = ""
    if status == "fast":
        badge_html = '<div class="badge-delivery">Fast Delivery</div>'
    
    sim_badge = f'<div class="badge-sim">{sim_score*100:.0f}% Sesuai</div>' if show_sim else ''

    col.markdown(f"""
    <div class="book-card">
        {sim_badge}
        <div class="book-cover-container">
            <img src="{cover_url}" class="book-cover" onerror="this.src='https://via.placeholder.com/130x200?text=No+Cover'">
        </div>
        {badge_html}
        <div class="book-title" title="{row['Book-Title']}">{row['Book-Title']}</div>
        <div class="book-author">{row['Book-Author']}</div>
        <div style="font-size:11px; color:#666; margin-top:5px;">Terbitan {row['Year-Of-Publication']} • Paperback</div>
    </div>
    """, unsafe_allow_html=True)
    
    with col.expander("Lihat Detail & Sinopsis"):
        st.write(f"**ISBN:** {row['ISBN']}")
        st.write(f"**Penerbit:** {row['Publisher']}")
        st.write("Buku pilihan dalam koleksi perpustakaan ini menawarkan sudut pandang menarik yang memperkaya wawasan literasi Anda.")


# ===== MENU NAVIGASI UTAMA =====
menu_pilihan = st.radio("Pilih Menu:", ["Beranda Katalog", "Estetika Data & Analitik", "Kurator Personal AI"], horizontal=True, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------
# 1. BERANDA
# ------------------------------------------
if menu_pilihan == "Beranda Katalog":
    search_q = st.text_input("", placeholder="Search by title / author / ISBN here...", label_visibility="collapsed")
    if search_q:
        res = books_df[books_df["Book-Title"].str.contains(search_q, case=False, na=False) | books_df["Book-Author"].str.contains(search_q, case=False, na=False)].head(6)
        if not res.empty:
            st.markdown('<div class="section-title">Hasil Pencarian Buku</div>', unsafe_allow_html=True)
            cols = st.columns(6)
            for i, (_, row) in enumerate(res.iterrows()):
                render_book_card(row, cols[i])

    st.markdown('<div class="section-title">Editor\'s Pick</div>', unsafe_allow_html=True)
    top_isbn = ratings_explicit["ISBN"].value_counts().head(6).index
    best_sellers = books_df[books_df["ISBN"].isin(top_isbn)].drop_duplicates(subset=["ISBN"])
    
    cols = st.columns(6)
    statuses = ["fast", "normal", "normal", "fast", "normal", "normal"]
    for i, (_, row) in enumerate(best_sellers.iterrows()):
        render_book_card(row, cols[i], status=statuses[i])

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Rekomendasi Pintar Untukmu</div>', unsafe_allow_html=True)
    
    all_books = books_df["Book-Title"].dropna().unique()
    selected_title = st.selectbox("Pilih buku favorit Anda untuk mencari kemiripan:", sorted(all_books), placeholder="Ketik atau pilih judul...")
    
    if selected_title:
        ref_isbn = books_df[books_df["Book-Title"] == selected_title]["ISBN"].values[0]
        
        if ref_isbn not in cf_sim_df.index:
            st.info("Buku ini memiliki ulasan terbatas. Menampilkan kurasi buku unggulan kami:")
            new_books = books_df[books_df["Year-Of-Publication"] >= 2004].head(6)
            cols = st.columns(6)
            for i, (_, row) in enumerate(new_books.iterrows()):
                render_book_card(row, cols[i])
        else:
            cf_scores = cf_sim_df[ref_isbn]
            cb_scores = cb_sim_df[ref_isbn]
            hybrid_scores = (0.5 * cf_scores) + (0.5 * cb_scores)
            
            top_similar = hybrid_scores.sort_values(ascending=False).iloc[1:7].index
            reco_books = books_df[books_df["ISBN"].isin(top_similar)].drop_duplicates(subset=["ISBN"])
            
            cols = st.columns(6)
            for i, (_, row) in enumerate(reco_books.iterrows()):
                score = hybrid_scores[row['ISBN']]
                render_book_card(row, cols[i], show_sim=True, sim_score=score)

# ------------------------------------------
# 2. ESTETIKA DATA
# ------------------------------------------
elif menu_pilihan == "Estetika Data & Analitik":
    st.markdown('<div class="section-title">Harmoni Ekosistem Buku & Pembaca</div>', unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-icon">📚</div><div class="kpi-title">Koleksi Buku</div><div class="kpi-value">{books_df["ISBN"].nunique():,}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-icon">👥</div><div class="kpi-title">Pembaca Setia</div><div class="kpi-value">{users_df["User-ID"].nunique():,}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-icon">💌</div><div class="kpi-title">Total Ulasan</div><div class="kpi-value">{len(ratings_df):,}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><div class="kpi-icon">⭐</div><div class="kpi-title">Kepuasan Rata-rata</div><div class="kpi-value">{ratings_explicit["Book-Rating"].mean():.1f} / 10</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='text-align:center; font-family:\"Roboto\", sans-serif; color:#222;'>Tren Tahun Publikasi Buku</h4>", unsafe_allow_html=True)
    yearly_pubs = books_df[(books_df['Year-Of-Publication'] >= 1970) & (books_df['Year-Of-Publication'] <= 2026)]['Year-Of-Publication'].value_counts().sort_index().reset_index()
    yearly_pubs.columns = ['Tahun', 'Jumlah Buku Dirilis']
    fig_line = px.area(yearly_pubs, x="Tahun", y="Jumlah Buku Dirilis", color_discrete_sequence=["#B22222"])
    fig_line.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#E0E0E0"), font=dict(family="Roboto", color="#222"), height=300)
    fig_line.update_traces(fillcolor='rgba(178, 34, 34, 0.15)', line=dict(width=3))
    st.plotly_chart(fig_line, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<h4 style='text-align:center; font-family:\"Roboto\", sans-serif; color:#222;'>Segmentasi Persona Pembaca (AI K-Means)</h4>", unsafe_allow_html=True)
        persona_counts = user_clusters['Persona'].value_counts().reset_index()
        fig_pie = px.pie(persona_counts, values='count', names='Persona', hole=0.5, color_discrete_sequence=["#B22222", "#5C9EAD", "#D4A373"])
        fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Roboto", color="#222"), showlegend=True)
        fig_pie.update_traces(textinfo='percent+label', textposition='inside')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.markdown("<h4 style='text-align:center; font-family:\"Roboto\", sans-serif; color:#222;'>Distribusi Usia Pembaca</h4>", unsafe_allow_html=True)
        fig_hist = px.histogram(users_df.dropna(subset=["Age"]), x="Age", nbins=25, color_discrete_sequence=["#5C9EAD"])
        fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False, title="Usia"), yaxis=dict(showgrid=True, gridcolor="#E0E0E0", title="Jumlah"), font=dict(family="Roboto", color="#222"))
        fig_hist.update_traces(marker_line_width=0, opacity=0.9)
        st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------------------
# 3. KURATOR PERSONAL AI
# ------------------------------------------
elif menu_pilihan == "Kurator Personal AI":
    st.markdown('<div class="section-title">Kurasi Spesial Untukmu</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#555; margin-bottom:30px;'>Pilih buku kesukaan Anda, lalu atur keseimbangan kecerdasan buatan kami.</p>", unsafe_allow_html=True)
    
    all_books = books_df["Book-Title"].dropna().unique()
    selected_title = st.selectbox("Pilih buku referensimu:", sorted(all_books), placeholder="Ketik judul buku...")
    
    if selected_title:
        ref_isbn = books_df[books_df["Book-Title"] == selected_title]["ISBN"].values[0]
        
        if ref_isbn not in cf_sim_df.index:
            st.warning("Buku ini adalah koleksi langka dengan sedikit ulasan. Sistem mengalihkan ke daftar rekomendasi terbaik kami:")
            top_isbn = ratings_explicit["ISBN"].value_counts().head(6).index
            reco_books = books_df[books_df["ISBN"].isin(top_isbn)].drop_duplicates(subset=["ISBN"])
            
            cols = st.columns(6)
            for i, (_, row) in enumerate(reco_books.iterrows()):
                render_book_card(row, cols[i], tag="BEST SELLER")
        else:
            st.markdown("<br><p style='font-weight:600; color:#222; font-size:14px; text-align:center;'>Atur Karakter Rekomendasi (Hybrid AI)</p>", unsafe_allow_html=True)
            col_sl1, col_sl2, col_sl3 = st.columns([1, 2, 1])
            with col_sl2:
                alpha = st.slider(
                    "Geser ke kiri (Mirip Tema) ↔ Geser ke kanan (Mirip Selera Pembaca Lain)",
                    min_value=0.0, max_value=1.0, value=0.6, step=0.1, label_visibility="collapsed"
                )
            
            st.markdown("---")
            with st.status("Meracik rekomendasi untukmu...", expanded=True) as status:
                time.sleep(0.6)
                cf_scores = cf_sim_df[ref_isbn]
                cb_scores = cb_sim_df[ref_isbn]
                hybrid_scores = (alpha * cf_scores) + ((1 - alpha) * cb_scores)
                
                top_similar_isbns = hybrid_scores.sort_values(ascending=False).iloc[1:7].index
                reco_books = books_df[books_df["ISBN"].isin(top_similar_isbns)].drop_duplicates(subset=["ISBN"])
                status.update(label="Kurasi selesai! ✨", state="complete", expanded=False)
            
            cols = st.columns(6)
            for i, (_, row) in enumerate(reco_books.iterrows()):
                score = hybrid_scores[row['ISBN']]
                render_book_card(row, cols[i], show_sim=True, sim_score=score)
