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

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Ruang Baca",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== PALET WARNA =====
CITRON  = "#CAD183"
PURPLE  = "#66023C"
CREAM   = "#FFFDD0"
DARK_INK = "#3D0A22"   # varian lebih gelap untuk teks di cream
SOFT     = "#F6F3C8"   # cream lebih pekat untuk card

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

/* Judul & Angka besar pakai Playfair */
h1, h2, h3, h4, h5,
.page-title, .section-title, .header-title,
.kpi-value, .book-title, .mood-label,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5 {{
    font-family: 'Playfair Display', Georgia, serif !important;
}}

/* ===== BACKGROUND: CREAM ===== */
.stApp {{
    background: {CREAM};
}}
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {{
    background: transparent !important;
}}

/* ===== SIDEBAR: TYRIAN PURPLE ===== */
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
    transition: 0.2s;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 17px;
    letter-spacing: 0.3px;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background: rgba(202, 209, 131, 0.12);
}}

/* ===== SEARCH BAR ===== */
[data-testid="stTextInput"] div[data-baseweb="input"] {{
    background-color: {CREAM} !important;
    border: 2px solid {PURPLE} !important;
    border-radius: 10px !important;
    box-shadow: 0 3px 0 {PURPLE} !important;
}}
[data-testid="stTextInput"] input {{
    color: {DARK_INK} !important;
    background-color: {CREAM} !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 16px !important;
}}
[data-testid="stTextInput"] input::placeholder {{
    color: {PURPLE} !important;
    opacity: 0.5;
    font-family: 'Cormorant Garamond', serif !important;
}}

/* ===== SELECTBOX / SLIDER ===== */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background-color: {CREAM} !important;
    border: 2px solid {PURPLE} !important;
    border-radius: 10px !important;
}}
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] div {{
    color: {DARK_INK} !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 16px !important;
}}
[data-baseweb="popover"] li {{
    color: {DARK_INK} !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 15px !important;
}}

[data-testid="stSlider"] [role="slider"] {{
    background-color: {PURPLE} !important;
}}
[data-testid="stSlider"] > div > div > div > div {{
    background: {PURPLE} !important;
}}

/* Tombol mood */
[data-testid="stButton"] button {{
    background-color: {PURPLE} !important;
    border: none !important;
    border-radius: 8px !important;
    transition: 0.2s;
    padding: 8px 14px !important;
}}
[data-testid="stButton"] button p,
[data-testid="stButton"] button div,
[data-testid="stButton"] button span {{
    color: {CREAM} !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 1.5px !important;
}}
[data-testid="stButton"] button:hover {{
    background-color: {DARK_INK} !important;
    transform: translateY(-2px);
}}

/* ===== EXPANDER ===== */
[data-testid="stExpander"] {{
    background-color: {CREAM} !important;
    border: 2px solid {PURPLE} !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    margin-top: -18px;
}}
[data-testid="stExpander"] summary {{
    justify-content: center !important;
    padding: 8px 0 !important;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {{
    color: {PURPLE} !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 1.2px !important;
}}
[data-testid="stExpander"] p,
[data-testid="stExpander"] div,
[data-testid="stExpander"] li,
[data-testid="stExpander"] span {{
    color: {DARK_INK} !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}}
[data-testid="stExpander"] strong {{
    font-family: 'Playfair Display', serif !important;
    color: {PURPLE} !important;
}}

/* ===== HEADER ===== */
.top-header {{
    background: {PURPLE};
    padding: 32px 38px;
    border-radius: 18px;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 28px;
    box-shadow: 0 8px 0 {CITRON};
    position: relative;
}}
.header-title {{
    font-family: 'Playfair Display', serif !important;
    font-size: 36px; font-weight: 700; margin: 0;
    letter-spacing: 0.3px; line-height: 1.1;
    color: {CREAM};
}}
.header-title em {{ color: {CITRON}; font-style: italic; font-weight: 600; }}
.header-subtitle {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 18px; font-weight: 400;
    opacity: 0.9; margin-top: 6px;
    color: {CREAM};
    font-style: italic;
    letter-spacing: 0.3px;
}}
.header-badge {{
    background: {CITRON}; color: {PURPLE};
    font-family: 'Playfair Display', serif !important;
    font-size: 11px; font-weight: 700;
    padding: 7px 16px; border-radius: 999px;
    letter-spacing: 2px;
}}

/* ===== SECTION TITLE ===== */
.section-title {{
    font-family: 'Playfair Display', serif !important;
    font-size: 32px; font-weight: 600;
    color: {PURPLE};
    text-align: center;
    margin-top: 44px; margin-bottom: 8px;
    font-style: italic;
    letter-spacing: 0.3px;
}}
.section-sub {{
    font-family: 'Cormorant Garamond', serif !important;
    text-align: center;
    color: {PURPLE}; opacity: 0.6;
    font-size: 14px; letter-spacing: 3px;
    margin-bottom: 28px; text-transform: uppercase;
    font-style: italic;
}}
.section-title::after {{
    content: ''; display: block;
    width: 60px; height: 3px;
    background-color: {CITRON};
    margin: 12px auto 0; border-radius: 2px;
}}

/* ===== BOOK CARD ===== */
.book-card {{
    background: {CREAM};
    border-radius: 12px 12px 0 0;
    padding: 20px 15px;
    border: 2px solid {PURPLE};
    border-bottom: none;
    text-align: center;
    height: 100%;
    position: relative;
}}
.book-cover-container {{
    height: 190px; display: flex;
    align-items: center; justify-content: center;
    margin-bottom: 18px;
}}
.book-cover {{
    max-height: 100%; max-width: 100%;
    object-fit: contain; border-radius: 4px;
    box-shadow: 4px 4px 0 {PURPLE};
    transition: 0.25s;
}}
.book-card:hover .book-cover {{
    transform: translate(-2px, -2px);
    box-shadow: 6px 6px 0 {CITRON}, 6px 6px 0 1px {PURPLE};
}}
.badge-tag {{
    position: absolute; top: 12px; left: 12px;
    background: {CITRON}; color: {PURPLE};
    font-family: 'Playfair Display', serif !important;
    font-size: 10px; font-weight: 800;
    padding: 4px 11px; border-radius: 20px;
    letter-spacing: 1.2px;
}}
.book-title {{
    font-family: 'Playfair Display', serif !important;
    font-weight: 700; font-size: 16px;
    color: {PURPLE};
    margin-bottom: 6px;
    display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden;
    line-height: 1.25;
}}
.book-author {{
    font-family: 'Cormorant Garamond', serif !important;
    color: {DARK_INK}; opacity: 0.7;
    font-size: 14px; font-weight: 500;
    margin-bottom: 6px;
    font-style: italic;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.book-year {{
    font-family: 'Playfair Display', serif !important;
    color: {PURPLE}; opacity: 0.55;
    font-size: 10.5px; font-weight: 700;
    margin-bottom: 10px; letter-spacing: 1.5px;
}}
.sim-pill {{
    display: inline-block;
    background: {CITRON}; color: {PURPLE};
    font-family: 'Playfair Display', serif !important;
    padding: 3px 12px; border-radius: 999px;
    font-size: 11px; font-weight: 700;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
}}

/* ===== KPI ===== */
.kpi-card {{
    background: {CREAM};
    border-radius: 14px;
    padding: 24px 20px;
    text-align: center;
    border: 2px solid {PURPLE};
    box-shadow: 5px 5px 0 {PURPLE};
    transition: 0.25s;
}}
.kpi-card:hover {{
    transform: translate(-2px, -2px);
    box-shadow: 7px 7px 0 {CITRON}, 7px 7px 0 2px {PURPLE};
}}
.kpi-icon {{ font-size: 26px; margin-bottom: 12px; }}
.kpi-title {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 12px; color: {PURPLE};
    font-weight: 700;
    margin-bottom: 8px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}}
.kpi-value {{
    font-family: 'Playfair Display', serif !important;
    font-size: 34px; font-weight: 700;
    color: {PURPLE};
    line-height: 1;
}}
.kpi-value small {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 16px; color: {PURPLE};
    opacity: 0.6;
    font-style: italic;
}}

/* ===== MOOD CARD ===== */
.mood-card {{
    background: {CREAM};
    border-radius: 14px;
    padding: 26px 16px;
    border: 2px solid {PURPLE};
    box-shadow: 5px 5px 0 {PURPLE};
    text-align: center;
    cursor: pointer;
    transition: all 0.25s ease;
    height: 100%;
}}
.mood-card:hover {{
    transform: translate(-2px, -2px);
    box-shadow: 7px 7px 0 {CITRON}, 7px 7px 0 2px {PURPLE};
}}
.mood-icon {{ font-size: 38px; margin-bottom: 12px; }}
.mood-label {{
    font-family: 'Playfair Display', serif !important;
    font-size: 17px; font-weight: 700;
    color: {PURPLE};
    font-style: italic;
    line-height: 1.2;
}}
.mood-card.active {{
    background: {CITRON};
}}

/* ===== INFO BOX ===== */
.info-box {{
    background: {CREAM};
    border-left: 6px solid {CITRON};
    padding: 18px 24px;
    border-radius: 8px;
    margin-bottom: 28px;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 16px;
    line-height: 1.7;
    font-style: italic;
    color: {PURPLE};
}}

/* Streamlit alert (success) */
[data-testid="stAlert"] {{
    background: {CITRON} !important;
    border-radius: 8px !important;
}}
[data-testid="stAlert"] * {{
    color: {PURPLE} !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 10px; }}
::-webkit-scrollbar-track {{ background: {CREAM}; }}
::-webkit-scrollbar-thumb {{ background: {PURPLE}; border-radius: 5px; }}
</style>
""", unsafe_allow_html=True)

# ===== HELPERS =====
def format_year(y):
    try:
        if pd.isna(y): return "—"
        return str(int(float(y)))
    except (ValueError, TypeError):
        return "—"

def get_cover(row):
    url = row.get("Image-URL-M", "")
    if pd.isna(url) or not isinstance(url, str) or len(url) < 10:
        return f"https://covers.openlibrary.org/b/isbn/{row['ISBN']}-M.jpg"
    return url

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
    st.markdown(
        '<h1 style="text-align:center; margin:0; font-family:\'Playfair Display\', serif; '
        'font-weight:700; font-style:italic; font-size:34px; color:#FFFDD0;">Ruang Baca</h1>',
        unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center; font-size:11px; letter-spacing:3px; '
        'opacity:0.7; margin-top:6px; margin-bottom:28px; color:#CAD183;">TEMPAT MENENANGKAN PIKIRAN</p>',
        unsafe_allow_html=True)

    page = st.radio("Navigasi", ["🏠 Beranda", "📊 Estetika Data", "🎯 Kurator Personal"])

    st.markdown(
        '<div style="margin-top: 60px; font-size:10.5px; text-align:center; '
        'opacity:0.6; letter-spacing:2px; color:#CAD183;">AI HYBRID · 2026</div>',
        unsafe_allow_html=True)

# ===== BOOK CARD =====
def render_book_card(row, col, tag="BACA", show_sim=False, sim_score=0):
    cover_url = get_cover(row)
    year_str = format_year(row.get("Year-Of-Publication"))

    if show_sim:
        info_html = f'<div class="sim-pill">Cocok {sim_score*100:.0f}%</div>'
    else:
        info_html = f'<div class="book-year">TERBIT {year_str}</div>'

    col.markdown(f"""
    <div class="book-card">
        <div class="badge-tag">{tag}</div>
        <div class="book-cover-container">
            <img src="{cover_url}" class="book-cover"
                 onerror="this.src='https://via.placeholder.com/130x200/FFFDD0/66023C?text=No+Cover'">
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
        rv = ratings_explicit[ratings_explicit["ISBN"] == row["ISBN"]]["Book-Rating"]
        if len(rv):
            st.markdown(f"- **Rating komunitas:** ⭐ {rv.mean():.1f}/10 ({len(rv)} ulasan)")
        else:
            st.markdown("- **Rating komunitas:** —")

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
        <div class="header-badge">EST. 2026</div>
    </div>
    """, unsafe_allow_html=True)

    search_q = st.text_input("", placeholder="🔍 Cari judul atau penulis favoritmu...",
                             label_visibility="collapsed")
    if search_q:
        res = books_df[
            books_df["Book-Title"].str.contains(search_q, case=False, na=False) |
            books_df["Book-Author"].str.contains(search_q, case=False, na=False)
        ].head(5)
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

    st.markdown('<br><br>', unsafe_allow_html=True)

    # ===== MOOD SECTION =====
    st.markdown('<div class="section-title">Baca Sesuai Mood</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">pilih suasana hatimu</div>', unsafe_allow_html=True)

    mood_map = {
        "🍵  Menenangkan Pikiran": ["Spiritual", "Philosophy", "Self-Help", "Inspirational",
                                    "Poetry", "Meditation", "Peace", "Zen"],
        "✨  Petualangan Magis": ["Fantasy", "Magic", "Adventure", "Epic", "Wizard",
                                   "Quest", "Dragon", "Myth"],
        "🌱  Pengembangan Diri": ["Self", "Business", "Success", "Psychology", "Motivation",
                                   "Productivity", "Leadership", "Personal"],
        "💌  Romansa Klasik": ["Romance", "Love", "Passion", "Heart", "Historical",
                               "Victorian", "Regency", "Bride"],
    }
    mood_icons = {"🍵  Menenangkan Pikiran": "🍵", "✨  Petualangan Magis": "✨",
                  "🌱  Pengembangan Diri": "🌱", "💌  Romansa Klasik": "💌"}
    mood_labels = {k: k.split("  ", 1)[1] for k in mood_map}

    mood_cols = st.columns(4)
    selected_mood = st.session_state.get("selected_mood", None)
    for i, mood in enumerate(mood_map.keys()):
        is_active = (selected_mood == mood)
        active_cls = " active" if is_active else ""
        with mood_cols[i]:
            st.markdown(f"""
            <div class="mood-card{active_cls}">
                <div class="mood-icon">{mood_icons[mood]}</div>
                <div class="mood-label">{mood_labels[mood]}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("✓ TERPILIH" if is_active else "PILIH",
                         key=f"mood_{i}", use_container_width=True):
                st.session_state["selected_mood"] = None if is_active else mood
                st.rerun()

    if st.session_state.get("selected_mood"):
        mood = st.session_state["selected_mood"]
        keywords = mood_map[mood]
        pattern = "|".join(keywords)
        mood_books = books_df[
            books_df["Book-Title"].str.contains(pattern, case=False, na=False) |
            books_df["Book-Author"].str.contains(pattern, case=False, na=False) |
            books_df["Publisher"].str.contains(pattern, case=False, na=False)
        ].drop_duplicates(subset=["ISBN"]).head(6)

        st.markdown(f'<div class="section-title">Rekomendasi untuk "{mood_labels[mood]}"</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-sub">berdasarkan mood pilihanmu</div>',
                    unsafe_allow_html=True)

        if mood_books.empty:
            top_ids = (ratings_explicit.groupby("ISBN")["Book-Rating"].mean()
                       .sort_values(ascending=False).head(6).index)
            mood_books = books_df[books_df["ISBN"].isin(top_ids)].drop_duplicates(subset=["ISBN"])
            st.markdown('<div class="info-box">Belum ada buku spesifik untuk mood ini. Berikut pilihan rating tertinggi dari komunitas.</div>', unsafe_allow_html=True)

        cols = st.columns(6)
        for i, (_, row) in enumerate(mood_books.head(6).iterrows()):
            render_book_card(row, cols[i], tag="MOOD")

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

    st.markdown("<br><br>", unsafe_allow_html=True)
    col_chart1, col_chart2 = st.columns(2)

    PLOT_LAYOUT = dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=PURPLE),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    with col_chart1:
        st.markdown(
            f"<h3 style='text-align:center; font-family:\"Playfair Display\", serif; "
            f"color:{PURPLE}; font-style:italic; margin-bottom:4px;'>Sebaran Apresiasi</h3>"
            f"<p style='text-align:center; color:{PURPLE}; opacity:0.5; font-size:11px; "
            f"letter-spacing:2px; margin-bottom:18px;'>SKOR BINTANG</p>",
            unsafe_allow_html=True)
        dist = ratings_explicit["Book-Rating"].value_counts().sort_index().reset_index()
        dist.columns = ["Rating", "Jumlah"]
        fig1 = px.bar(dist, x="Rating", y="Jumlah", color_discrete_sequence=[PURPLE])
        fig1.update_layout(**PLOT_LAYOUT, height=380,
                           xaxis=dict(dtick=1, showgrid=False, title=""),
                           yaxis=dict(showgrid=True, gridcolor="rgba(102,2,60,0.08)", title=""))
        fig1.update_traces(marker_line_width=0, opacity=0.95)
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.markdown(
            f"<h3 style='text-align:center; font-family:\"Playfair Display\", serif; "
            f"color:{PURPLE}; font-style:italic; margin-bottom:4px;'>Demografi Usia</h3>"
            f"<p style='text-align:center; color:{PURPLE}; opacity:0.5; font-size:11px; "
            f"letter-spacing:2px; margin-bottom:18px;'>PEMBACA</p>",
            unsafe_allow_html=True)
        fig2 = px.histogram(users_df.dropna(subset=["Age"]), x="Age", nbins=25,
                            color_discrete_sequence=[CITRON])
        fig2.update_layout(**PLOT_LAYOUT, height=380,
                           xaxis=dict(showgrid=False, title=""),
                           yaxis=dict(showgrid=True, gridcolor="rgba(102,2,60,0.08)", title=""))
        fig2.update_traces(marker_line_width=1, marker_line_color=PURPLE, opacity=0.95)
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# HALAMAN 3: KURATOR PERSONAL
# ==========================================
elif page == "🎯 Kurator Personal":
    st.markdown('<div class="section-title">Kurasi Spesial Untukmu</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">pilih buku, dapatkan racikan</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="info-box">Beritahu kami satu buku yang kamu sukai — mesin kognitif kami akan meracik rekomendasi yang senada.</div>', unsafe_allow_html=True)

    valid_books = books_df[books_df["ISBN"].isin(cf_sim_df.index)]
    book_options = valid_books["Book-Title"].dropna().unique()

    selected_title = st.selectbox("Pilih buku referensimu:", sorted(book_options),
                                  index=None, placeholder="Ketik judul buku...")

    if selected_title:
        ref_isbn = valid_books[valid_books["Book-Title"] == selected_title]["ISBN"].values[0]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-weight:700; color:{PURPLE}; font-family:\"Playfair Display\", serif; "
            f"font-style:italic; font-size:18px;'>Atur Karakter Rekomendasi</p>",
            unsafe_allow_html=True)
        alpha = st.slider(
            "Geser kiri = mirip tema & penulis | Geser kanan = mirip selera pembaca lain",
            min_value=0.0, max_value=1.0, value=0.6, step=0.1, label_visibility="collapsed"
        )

        st.markdown("<hr style='border:none; border-top:2px solid rgba(102,2,60,0.15); margin:28px 0;'>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<h3 style='font-family:\"Playfair Display\", serif; color:{PURPLE}; "
            f"font-style:italic; font-weight:600;'>✨ Bacaan yang Mungkin Mengisi Hatimu</h3>",
            unsafe_allow_html=True)

        cf_scores = cf_sim_df[ref_isbn]
        cb_scores = cb_sim_df[ref_isbn]
        hybrid_scores = (alpha * cf_scores) + ((1 - alpha) * cb_scores)

        top_similar_isbns = hybrid_scores.sort_values(ascending=False).iloc[1:7].index
        reco_books = books_df[books_df["ISBN"].isin(top_similar_isbns)].drop_duplicates(subset=["ISBN"])

        cols = st.columns(6)
        for i, isbn in enumerate(top_similar_isbns):
            book_row = reco_books[reco_books["ISBN"] == isbn]
            if not book_row.empty:
                score = hybrid_scores[isbn]
                render_book_card(book_row.iloc[0], cols[i], tag="REKOMENDASI",
                                 show_sim=True, sim_score=score)
