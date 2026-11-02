# ==========================================================
# RUANG BACA — TYRIAN PURPLE × CITRON × CREAM THEME
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import gdown
import base64

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Ruang Baca",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== PALET WARNA =====
CITRON   = "#CAD183"
PURPLE   = "#66023C"
CREAM    = "#FFFDD0"
DARK_INK = "#3D0A22"   
SOFT     = "#F6F3C8"   

# ===== HELPER & ASSETS =====
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

# Deteksi otomatis background dari folder assets jika ada
BG_IMAGE = ""
for ext in ["jpg", "jpeg", "png", "webp"]:
    path = f"assets/background.{ext}"
    if os.path.exists(path):
        BG_IMAGE = get_base64_image(path)
        break

bg_css = f"linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)), url('data:image/jpeg;base64,{BG_IMAGE}')" if BG_IMAGE else CREAM

# ===== CSS =====
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,800;0,900;1,400;1,600&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

html, body, [class*="css"] {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    color: {DARK_INK};
    font-size: 16px;
}}
[data-testid="stAppViewContainer"] *,
[data-testid="stSidebar"] *,
.stApp * {{
    font-family: 'Cormorant Garamond', Georgia, serif !important;
}}

h1, h2, h3, h4, h5,
.page-title, .section-title, .header-title,
.kpi-value, .book-title, .mood-label {{
    font-family: 'Playfair Display', Georgia, serif !important;
}}

.stApp {{
    background: {bg_css};
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {{
    background: transparent !important;
}}

/* SIDEBAR */
[data-testid="stSidebar"] {{
    background-color: {PURPLE} !important;
    border-right: 1px solid rgba(202, 209, 131, 0.25);
}}
[data-testid="stSidebar"] * {{
    color: {CREAM} !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    border-radius: 8px;
    padding: 8px 12px !important;
    font-size: 17px;
}}

/* SEARCH & INPUT */
[data-testid="stTextInput"] div[data-baseweb="input"] {{
    background-color: rgba(255, 253, 208, 0.9) !important;
    border: 2px solid {PURPLE} !important;
    border-radius: 10px !important;
}}
[data-testid="stTextInput"] input {{
    color: {DARK_INK} !important;
    background-color: transparent !important;
    font-size: 16px !important;
}}

/* BUTTONS */
[data-testid="stButton"] button {{
    background-color: {PURPLE} !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
}}
[data-testid="stButton"] button p,
[data-testid="stButton"] button span {{
    color: {CREAM} !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
}}

/* HEADER & CARDS */
.top-header {{
    background: {PURPLE};
    padding: 32px 38px;
    border-radius: 18px;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 28px;
    box-shadow: 0 8px 0 {CITRON};
}}
.header-title {{
    font-family: 'Playfair Display', serif !important;
    font-size: 36px; font-weight: 700; margin: 0; color: {CREAM};
}}
.header-title em {{ color: {CITRON}; font-style: italic; }}
.header-subtitle {{
    font-size: 18px; opacity: 0.9; margin-top: 6px; color: {CREAM}; font-style: italic;
}}
.section-title {{
    font-family: 'Playfair Display', serif !important;
    font-size: 32px; font-weight: 600; color: {PURPLE}; text-align: center;
    margin-top: 44px; margin-bottom: 8px; font-style: italic;
}}
.section-sub {{
    text-align: center; color: {PURPLE}; opacity: 0.6;
    font-size: 14px; letter-spacing: 3px; margin-bottom: 28px; text-transform: uppercase; font-style: italic;
}}
.book-card {{
    background: rgba(255, 253, 208, 0.95);
    border-radius: 12px 12px 0 0; padding: 20px 15px;
    border: 2px solid {PURPLE}; border-bottom: none;
    text-align: center; height: 100%; position: relative;
}}
.book-cover-container {{ height: 190px; display: flex; align-items: center; justify-content: center; margin-bottom: 18px; }}
.book-cover {{ max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 4px; box-shadow: 4px 4px 0 {PURPLE}; }}
.badge-tag {{
    position: absolute; top: 12px; left: 12px; background: {CITRON}; color: {PURPLE};
    font-size: 10px; font-weight: 800; padding: 4px 11px; border-radius: 20px;
}}
.book-title {{ font-weight: 700; font-size: 16px; color: {PURPLE}; margin-bottom: 6px; }}
.book-author {{ color: {DARK_INK}; opacity: 0.7; font-size: 14px; font-style: italic; }}
.kpi-card, .mood-card {{
    background: rgba(255, 253, 208, 0.95); border-radius: 14px; padding: 24px 20px;
    text-align: center; border: 2px solid {PURPLE}; box-shadow: 5px 5px 0 {PURPLE};
}}
.info-box {{
    background: rgba(255, 253, 208, 0.95); border-left: 6px solid {CITRON};
    padding: 18px 24px; border-radius: 8px; margin-bottom: 28px; font-style: italic; color: {PURPLE};
    border: 1px solid {PURPLE};
}}
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

# ===== HYBRID ENGINE =====
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
    pop_books["content_features"] = (pop_books["Book-Title"].fillna("") + " "
                                     + pop_books["Book-Author"].fillna("") + " "
                                     + pop_books["Publisher"].fillna(""))
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(pop_books["content_features"])
    cb_sim = cosine_similarity(tfidf_matrix)
    cb_sim_df = pd.DataFrame(cb_sim, index=popular_isbn, columns=popular_isbn)
    return cf_sim_df, cb_sim_df

with st.spinner("Memproses rekomendasi buku..."):
    cf_sim_df, cb_sim_df = build_hybrid_matrices(ratings_explicit, books_df)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown('<div style="text-align:center; font-size:48px; margin-bottom:6px;">🌿</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="text-align:center; margin:0; font-family:\'Playfair Display\', serif; font-weight:700; font-style:italic; font-size:34px; color:#FFFDD0;">Ruang Baca</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; font-size:11px; letter-spacing:3px; opacity:0.7; margin-top:6px; margin-bottom:28px; color:#CAD183;">TEMPAT MENENANGKAN PIKIRAN</p>', unsafe_allow_html=True)
    
    page = st.radio("Navigasi", ["🏠 Beranda", "📊 Estetika Data", "🎯 Kurator Personal"])

# ===== BOOK CARD =====
def render_book_card(row, col, tag="BACA", show_sim=False, sim_score=0):
    cover_url = row.get("Image-URL-M", "")
    if pd.isna(cover_url) or len(cover_url) < 10:
        cover_url = f"https://covers.openlibrary.org/b/isbn/{row['ISBN']}-M.jpg"
    
    year_val = row.get("Year-Of-Publication")
    year_str = str(int(float(year_val))) if pd.notna(year_val) else "—"

    info_html = f'<div style="background:#CAD183; color:#66023C; padding:3px 12px; border-radius:999px; font-size:11px; font-weight:700;">Cocok {sim_score*100:.0f}%</div>' if show_sim else f'<div style="font-size:10.5px; opacity:0.55; letter-spacing:1.5px; font-weight:700; margin-bottom:10px;">TERBIT {year_str}</div>'

    col.markdown(f"""
    <div class="book-card">
        <div class="badge-tag">{tag}</div>
        <div class="book-cover-container">
            <img src="{cover_url}" class="book-cover" onerror="this.src='https://via.placeholder.com/130x200/FFFDD0/66023C?text=No+Cover'">
        </div>
        <div class="book-title" title="{row['Book-Title']}">{row['Book-Title']}</div>
        <div class="book-author">oleh {row['Book-Author']}</div>
        {info_html}
    </div>
    """, unsafe_allow_html=True)

    with col.expander("🔎 LIHAT DETAIL"):
        st.markdown(f"**{row['Book-Title']}**")
        st.markdown(f"- **Penulis:** {row['Book-Author']}")
        st.markdown(f"- **Penerbit:** {row.get('Publisher', '—')}")
        st.markdown(f"- **Tahun:** {year_str}")

# ==========================================
# HALAMAN 1: BERANDA
# ==========================================
if page == "🏠 Beranda":
    st.markdown(f"""
    <div class="top-header">
        <div>
            <div class="header-title">Selamat datang di <em>Ruang Baca</em>.</div>
            <div class="header-subtitle">Temukan buku yang menenangkan pikiranmu hari ini.</div>
        </div>
        <div style="background:#CAD183; color:#66023C; padding:7px 16px; border-radius:999px; font-size:11px; font-weight:700; letter-spacing:2px;">EST. 2026</div>
    </div>
    """, unsafe_allow_html=True)

    search_q = st.text_input("", placeholder="🔍 Cari judul atau penulis favoritmu...", label_visibility="collapsed")
    if search_q:
        res = books_df[books_df["Book-Title"].str.contains(search_q, case=False, na=False) | books_df["Book-Author"].str.contains(search_q, case=False, na=False)].head(5)
        if not res.empty:
            st.success(f"Menemukan {len(res)} buku yang mungkin kamu cari 🤍")
            cols = st.columns(len(res))
            for i, (_, row) in enumerate(res.iterrows()):
                render_book_card(row, cols[i], tag="PILIHAN")

    st.markdown('<div class="section-title">Koleksi Terpopuler</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">bulan ini</div>', unsafe_allow_html=True)

    top_isbn = ratings_explicit["ISBN"].value_counts().head(6).index
    best_sellers = books_df[books_df["ISBN"].isin(top_isbn)].drop_duplicates(subset=["ISBN"])
    cols = st.columns(6)
    for i, (_, row) in enumerate(best_sellers.head(6).iterrows()):
        render_book_card(row, cols[i], tag="TERPOPULER")

# ==========================================
# HALAMAN 2: ESTETIKA DATA
# ==========================================
elif page == "📊 Estetika Data":
    st.markdown('<div class="section-title">Harmoni Ekosistem Buku</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">data dalam harmoni</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-icon">📚</div><div class="kpi-title">Koleksi Buku</div><div class="kpi-value">{books_df["ISBN"].nunique():,}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-icon">👥</div><div class="kpi-title">Pembaca Setia</div><div class="kpi-value">{users_df["User-ID"].nunique():,}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-icon">💌</div><div class="kpi-title">Total Ulasan</div><div class="kpi-value">{len(ratings_df):,}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><div class="kpi-icon">⭐</div><div class="kpi-title">Kepuasan Rata-rata</div><div class="kpi-value">{ratings_explicit["Book-Rating"].mean():.1f}<small> / 10</small></div></div>', unsafe_allow_html=True)

# ==========================================
# HALAMAN 3: KURATOR PERSONAL
# ==========================================
elif page == "🎯 Kurator Personal":
    st.markdown('<div class="section-title">Kurasi Spesial Untukmu</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">pilih buku, dapatkan racikan</div>', unsafe_allow_html=True)

    valid_books = books_df[books_df["ISBN"].isin(cf_sim_df.index)]
    book_options = valid_books["Book-Title"].dropna().unique()

    selected_title = st.selectbox("Pilih buku referensimu:", sorted(book_options), index=None, placeholder="Ketik judul buku...")

    if selected_title:
        ref_isbn = valid_books[valid_books["Book-Title"] == selected_title]["ISBN"].values[0]
        alpha = st.slider("Geser kiri = mirip tema | Geser kanan = mirip pembaca lain", 0.0, 1.0, 0.6, 0.1)
        
        cf_scores = cf_sim_df[ref_isbn]
        cb_scores = cb_sim_df[ref_isbn]
        hybrid_scores = (alpha * cf_scores) + ((1 - alpha) * cb_scores)

        top_similar_isbns = hybrid_scores.sort_values(ascending=False).iloc[1:7].index
        reco_books = books_df[books_df["ISBN"].isin(top_similar_isbns)].drop_duplicates(subset=["ISBN"])

        cols = st.columns(6)
        for i, isbn in enumerate(top_similar_isbns):
            book_row = reco_books[reco_books["ISBN"] == isbn]
            if not book_row.empty:
                render_book_card(book_row.iloc[0], cols[i], tag="REKOMENDASI", show_sim=True, sim_score=hybrid_scores[isbn])
