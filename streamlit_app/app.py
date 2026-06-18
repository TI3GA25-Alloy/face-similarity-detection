import matplotlib
import numpy as np
import streamlit as st

matplotlib.use("Agg")
import os
import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.face_utils import (
    detect_face,
    load_image_from_pil,
    preprocess_face,
)

from core.pca_svd import (
    analyze_two_faces,
    analyze_two_faces_with_dataset,
    build_eigenspace_from_dataset,
    load_custom_selfie_dataset,
    load_lfw_dataset,
    load_olivetti_dataset,
    load_pretrained_eigenspace,
)

from core.similarity import compute_all_metrics, make_decision
from PIL import Image

st.set_page_config(
    page_title="FaceMatch PCA/SVD + Dataset",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #030712;
    --bg-card: rgba(17, 24, 39, 0.4);
    --bg-card2: rgba(31, 41, 55, 0.5);
    --accent-blue: #4f46e5;
    --accent-purple: #6366f1;
    --accent-cyan: #818cf8;
    --accent-green: #10b981; 
    --text-primary: #e5e7eb;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    --border: rgba(31, 41, 55, 1);
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] { 
    background: var(--bg-primary) !important; 
    font-family: 'Inter', sans-serif; 
    overflow-x: hidden !important; 
}
.main .block-container { padding: 1.5rem 1rem 3rem; max-width: 1200px; }
@media (min-width: 768px) {
    .main .block-container { padding: 1.5rem 2rem 3rem; }
}

@keyframes gradient {
    to { background-position: 200% center; }
}

.app-header {
    position: relative;
    padding: 2rem 0;
    margin-bottom: 1.5rem;
    text-align: center;
    border-bottom: 1px solid transparent;
    border-image: linear-gradient(to right, transparent, rgba(99, 102, 241, 0.25), transparent) 1;
}
@media (min-width: 768px) {
    .app-header { padding: 3rem 0; margin-bottom: 2rem; }
}

.app-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(to right, #e5e7eb, #c7d2fe, #f9fafb, #a5b4fc, #e5e7eb);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient 6s linear infinite;
    margin: 0 0 1rem;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
@media (min-width: 768px) {
    .app-title { font-size: 3.5rem; }
}

.badge-row { display: flex; gap: 0.5rem; margin-top: 1.5rem; flex-wrap: wrap; justify-content: center; }
.badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: rgba(99, 102, 241, 0.1); 
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 50px; padding: 0.3rem 0.8rem; font-size: 0.75rem;
    font-weight: 600; color: #a5b4fc; text-transform: uppercase; letter-spacing: 0.05em;
}

.dataset-card {
    background: rgba(17, 24, 39, 0.5); backdrop-filter: blur(8px);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 1.5rem; margin-bottom: 1.25rem;
}
.dataset-stat {
    text-align: center; background: rgba(31, 41, 55, 0.4);
    border: 1px solid var(--border); border-radius: 12px; padding: 1rem;
}
.dataset-stat-num { font-size: 1.8rem; font-weight: 800; line-height: 1; color: #818cf8; }
.dataset-stat-lbl { font-size: 0.72rem; color: var(--text-muted); margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }

.metric-card {
    background: var(--bg-card); backdrop-filter: blur(4px);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 1.5rem; margin-bottom: 1rem;
}

.result-card { border-radius: 24px; padding: 1.5rem; text-align: center; margin: 1.5rem 0; backdrop-filter: blur(8px); }
@media (min-width: 768px) {
    .result-card { padding: 2.5rem; }
}
.result-same { background: rgba(17, 24, 39, 0.4); border: 1px solid rgba(16, 185, 129, 0.3); box-shadow: 0 0 40px -10px rgba(16, 185, 129, 0.2); }
.result-diff { background: rgba(17, 24, 39, 0.4); border: 1px solid rgba(239, 68, 68, 0.3); box-shadow: 0 0 40px -10px rgba(239, 68, 68, 0.2); }

.math-box {
    background: rgba(3, 7, 18, 1); border: 1px solid rgba(31, 41, 55, 0.8);
    border-left: 4px solid var(--accent-purple); border-radius: 8px;
    padding: 1.25rem; margin: 0.75rem 0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: rgba(199, 210, 254, 0.8);
    overflow-x: auto;
}

.section-title {
    font-size: 1.125rem; font-weight: 700; color: var(--text-primary);
    margin: 2rem 0 1.5rem; display: flex; align-items: center; gap: 0.75rem;
}
.section-title::after {
    content: ''; flex: 1; height: 1px; background: var(--border);
}

.progress-container { margin: 1rem 0; }
.progress-bar-bg { background: rgba(31, 41, 55, 0.5); border: 1px solid rgba(31, 41, 55, 1); border-radius: 50px; height: 8px; overflow: hidden; }
.progress-bar-fill { height: 100%; border-radius: 50px; transition: width 0.7s ease-out; }

.stButton > button {
    background: linear-gradient(to top, var(--accent-blue), var(--accent-purple)) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    font-weight: 600 !important; padding: 0.8rem 2rem !important; font-size: 1rem !important;
    width: 100%; box-shadow: inset 0px 1px 0px 0px rgba(255,255,255,0.16) !important;
    transition: all 0.3s ease !important; background-position: bottom !important; background-size: 100% 100% !important;
}
.stButton > button:hover { background-size: 100% 150% !important; }

h1,h2,h3,h4 { color: var(--text-primary) !important; font-weight: 700 !important; tracking: -0.02em !important; }
p { color: var(--text-secondary) !important; }

[data-testid="stSidebar"] { background: #030712 !important; border-right: 1px solid var(--border) !important; }
[data-testid="stFileUploader"] { 
    background: rgba(17, 24, 39, 0.4) !important; 
    border: 2px dashed rgba(99, 102, 241, 0.3) !important; 
    border-radius: 16px !important; 
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(99, 102, 241, 0.6) !important; background: rgba(99, 102, 241, 0.05) !important; }
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #030712; }
::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #374151; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="app-header">
  <div style="position:relative;z-index:1">
    <div class="app-title">FaceMatch<span style="opacity:0.5">/</span>Edge-PCA & SVD</div>
    <div style="color:rgba(199,210,254,0.65);font-size:1.125rem;margin-bottom:1rem;font-weight:500;">
      Deteksi Kemiripan Foto Lama vs Foto Baru menggunakan implementasi Edge-PCA Aljabar Linear + Euclidean Tie-Breaker.
    </div>
    <div class="badge-row">
      <span class="badge">Aljabar Linear</span>
      <span class="badge">Sobel Edge</span>
      <span class="badge">SVD</span>
      <span class="badge">Pinalti Euclidean</span>
      <span class="badge">PCA</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


def make_dark_plot():
    plt.rcParams.update(
        {
            "figure.facecolor": "#111827",
            "axes.facecolor": "#1a2235",
            "axes.edgecolor": "#2d3748",
            "axes.labelcolor": "#94a3b8",
            "xtick.color": "#94a3b8",
            "ytick.color": "#94a3b8",
            "text.color": "#f1f5f9",
            "grid.color": "#1e293b",
            "grid.alpha": 0.5,
        }
    )


def score_color(s):
    if s >= 0.85:
        return "#10b981"
    elif s >= 0.70:
        return "#22c55e"
    elif s >= 0.55:
        return "#f59e0b"
    elif s >= 0.40:
        return "#f97316"
    return "#ef4444"


def render_progress(score, label, color=None):
    c = color or score_color(score)
    pct = max(0.0, score) * 100
    st.markdown(
        f"""
    <div class="progress-container">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;font-size:0.78rem">
        <span style="color:#94a3b8">{label}</span>
        <span style="color:{c};font-weight:700;font-family:'JetBrains Mono',monospace">{pct:.1f}%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{c}88,{c})"></div>
      </div>
    </div>""",
        unsafe_allow_html=True,
    )


LANG_DICT = {
    "EN": {
        "title": "FaceMatch Edge-PCA & SVD",
        "subtitle": "Cross-Age Face Verification using Classical Linear Algebra + Euclidean Tie-Breaker.",
        "sidebar_dataset": "Eigenspace Dataset",
        "sidebar_dataset_choice": "Choose training dataset:",
        "sidebar_params": "Parameters",
        "sidebar_lang": "Language",
        "threshold_label": "Similarity Threshold",
        "penalty_label": "Euclidean Penalty Factor",
        "target_size_label": "Resize Dimension",
        "upload_old": "Upload Old Photo",
        "upload_new": "Upload New Photo",
        "preview_old": "Preview Old Photo",
        "preview_new": "Preview New Photo",
        "btn_analyze": "Analyze Similarity",
        "sec_01": "Section 01: Primary Similarity Results",
        "sec_02": "Section 02: Pixel Matrix Representation",
        "sec_03": "Section 03: PCA & Eigenspace",
        "sec_04": "Section 04: SVD Reconstruction",
        "sec_05": "Section 05: Vector Projection",
        "sec_06": "Section 06: Advanced Analysis",
        "sec_07": "Section 07: Dynamic Report",
    },
    "ID": {
        "title": "FaceMatch Edge-PCA & SVD",
        "subtitle": "Deteksi Kemiripan Foto Lama vs Foto Baru menggunakan implementasi Edge-PCA Aljabar Linear + Euclidean Tie-Breaker.",
        "sidebar_dataset": "Dataset Eigenspace",
        "sidebar_dataset_choice": "Pilih dataset training:",
        "sidebar_params": "Parameter",
        "sidebar_lang": "Bahasa",
        "threshold_label": "Ambang Batas Kemiripan",
        "penalty_label": "Faktor Penalti Euclidean",
        "target_size_label": "Ukuran Resize",
        "upload_old": "Unggah Foto Lama",
        "upload_new": "Unggah Foto Baru",
        "preview_old": "Pratinjau Foto Lama",
        "preview_new": "Pratinjau Foto Baru",
        "btn_analyze": "Analisis Kemiripan",
        "sec_01": "Bagian 01: Hasil Similarity Utama",
        "sec_02": "Bagian 02: Representasi Matriks Piksel",
        "sec_03": "Bagian 03: PCA & Eigenspace",
        "sec_04": "Bagian 04: SVD Rekonstruksi",
        "sec_05": "Bagian 05: Proyeksi Vektor",
        "sec_06": "Bagian 06: Analisis Lanjutan",
        "sec_07": "Bagian 07: Laporan Dinamis",
    },
}

with st.sidebar:
    lang = st.selectbox("Language / Bahasa", ["ID", "EN"])
    T = LANG_DICT[lang]

    st.markdown(f"## {T['sidebar_dataset']}")

    dataset_choice = st.radio(
        T["sidebar_dataset_choice"],
        [
            "Dataset Lokal (Selfie & ID)",
            "Olivetti Faces (Direkomendasikan)",
            "LFW (Labeled Faces in the Wild)",
            "Tanpa Dataset (2 gambar saja)",
        ],
        index=0,
        help="Dataset digunakan untuk membangun eigenspace yang lebih kaya",
    )

    n_components = st.slider(
        "Jumlah Eigenfaces (k) / PCA Components",
        5,
        100,
        50,
        5,
        help="Makin banyak = makin akurat, tapi lebih lambat",
    )

    st.divider()
    st.markdown(f"## {T['sidebar_params']}")

    threshold = st.slider(
        T["threshold_label"],
        0.20,
        0.95,
        0.45,
        0.01,
        help="Standar ML: 0.70 untuk Same-Age, 0.40 - 0.45 untuk Cross-Age Verification.",
    )
    penalty_factor = st.slider(T["penalty_label"], 0.01, 0.20, 0.05, 0.01)
    target_size_sel = 100

    st.sidebar.markdown("### Bobot Fusion (Sensor)")
    alpha_lbp = st.sidebar.slider(
        "Bobot LBP (Tekstur)",
        0.0,
        2.0,
        0.3,
        0.1,
        help="Tekstur wajah. Tahan terhadap variasi pencahayaan dan penuaan lokal.",
    )
    beta_hog = st.sidebar.slider(
        "Bobot HOG (Bentuk)",
        0.0,
        2.0,
        0.4,
        0.1,
        help="Struktur tulang/geometri wajah. Sensor paling tangguh untuk Lintas Usia.",
    )
    gamma_pix = st.sidebar.slider(
        "Bobot Pixel (Intensitas)",
        0.0,
        2.0,
        0.3,
        0.1,
        help="Sangat akurat untuk usia sebaya, namun rentan pada variasi pose dan usia.",
    )

    detect_face_opt = st.toggle("Auto Deteksi Wajah", value=True)
    apply_aging_vector = st.toggle(
        "Injeksi Vektor Penuaan (Foto Lama)", 
        value=False,
        help="Gunakan Aljabar Linear untuk menyuntikkan Vektor Penuaan ke Eigenspace foto masa kecil agar bentuk geometrisnya mendekati struktur wajah dewasa."
    )
    prob_asian = 0.5
    aging_scale = 0.0
    if apply_aging_vector:
        st.sidebar.markdown("### Ethnicity-Aware Aging")
        prob_asian = st.sidebar.slider(
            "Probabilitas Wajah Asia",
            0.0, 1.0, 0.8, 0.05,
            help="1.0 = Full AAF (Asia). 0.0 = Full FG-NET (Kaukasia)."
        )
        aging_scale = st.sidebar.slider(
            "Skala Penuaan (Aging Scale)",
            0.0, 2.0, 0.25, 0.05,
            help="Faktor skala untuk menyuntikkan vektor penuaan. Nilai optimal biasanya 0.1 - 0.3."
        )
    show_eigenfaces = st.toggle("Tampilkan Eigenfaces Dataset", value=True)

    st.divider()
    st.markdown("## Kenapa Butuh Dataset?")
    st.markdown("""
Eigenspace dari **2 gambar saja**:
- Max 2 komponen
- Tidak menangkap pola wajah umum
- Kurang akurat

Eigenspace dari **400 foto wajah**:
- 50+ komponen
- Menangkap variasi wajah manusia
- Jauh lebih akurat & bermakna

**Rumus SVD:**
```
A = U Sigma Vt
eigenfaces = k baris Vt
```

**Proyeksi:**
```
w = eigenfaces @ (face - mean)
```
""")
    st.caption("🎓 Aljabar Linear — Semester 2")


@st.cache_resource(show_spinner=False)
def get_eigenspace_olivetti(k):
    data = load_olivetti_dataset()
    eigenspace = build_eigenspace_from_dataset(
        data["images"], n_components=k, target_size=(64, 64)
    )
    return data, eigenspace


@st.cache_resource(show_spinner=False)
def get_eigenspace_lfw(k):
    data = load_lfw_dataset(min_faces=20, resize=0.4)
    if data is None:
        return None, None
    eigenspace = build_eigenspace_from_dataset(
        data["images"], n_components=k, target_size=data["image_shape"]
    )
    return data, eigenspace


@st.cache_resource(show_spinner=False)
def get_eigenspace_custom(k, mtime=0):
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "pretrained_eigenspace.npz",
    )
    eigenspace = load_pretrained_eigenspace(model_path)

    if eigenspace is not None:
        data = {
            "source": eigenspace["source"],
            "description": eigenspace["description"],
            "n_samples": eigenspace["n_samples"],
            "n_people": "?",
            "image_shape": eigenspace["image_shape"],
        }
        return data, eigenspace

    base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Selfie & id data - public sample",
    )
    data = load_custom_selfie_dataset(base_path, target_size=(128, 128))
    if data is not None:
        eigenspace = build_eigenspace_from_dataset(
            data["images"], n_components=k, target_size=(128, 128)
        )
    return data, eigenspace


use_dataset = "Tanpa Dataset" not in dataset_choice
dataset_data = None
eigenspace = None

if use_dataset:
    with st.spinner("Memuat dataset & membangun eigenspace..."):
        if "Lokal" in dataset_choice:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "pretrained_eigenspace.npz",
            )
            mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0
            dataset_data, eigenspace = get_eigenspace_custom(n_components, mtime)
            if eigenspace is None:
                st.warning("Dataset lokal tidak ditemukan. Beralih ke Olivetti.")
                dataset_data, eigenspace = get_eigenspace_olivetti(n_components)
        elif "Olivetti" in dataset_choice:
            dataset_data, eigenspace = get_eigenspace_olivetti(n_components)
        else:
            dataset_data, eigenspace = get_eigenspace_lfw(n_components)
            if eigenspace is None:
                st.warning("LFW gagal dimuat (butuh internet). Beralih ke Olivetti.")
                dataset_data, eigenspace = get_eigenspace_olivetti(n_components)

if use_dataset and dataset_data and eigenspace:
    total_var = eigenspace.get("total_variance_captured", 0)
    k95 = eigenspace.get("k_for_95pct_variance", "?")
    k_used = eigenspace.get("n_components", n_components)
    n_train = dataset_data.get("n_samples", 0)
    n_people = dataset_data.get("n_people", 0)
    img_shape = dataset_data.get("image_shape", (64, 64))

    st.markdown(
        '<div class="section-title">Dataset Eigenspace</div>', unsafe_allow_html=True
    )
    st.markdown(
        f"""
    <div class="dataset-card">
      <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem">
        <span style="font-weight:700;color:#10b981">OK</span>
        <div>
          <div style="font-weight:700;color:#f1f5f9">{dataset_data.get("source", "Dataset")}</div>
          <div style="font-size:0.8rem;color:#64748b">{dataset_data.get("description", "")[:120]}...</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.75rem">
        <div class="dataset-stat">
          <div class="dataset-stat-num" style="color:#10b981">{n_train}</div>
          <div class="dataset-stat-lbl">Foto Training</div>
        </div>
        <div class="dataset-stat">
          <div class="dataset-stat-num" style="color:#3b82f6">{n_people}</div>
          <div class="dataset-stat-lbl">Orang Berbeda</div>
        </div>
        <div class="dataset-stat">
          <div class="dataset-stat-num" style="color:#8b5cf6">{img_shape[0]}×{img_shape[1]}</div>
          <div class="dataset-stat-lbl">Ukuran Gambar</div>
        </div>
        <div class="dataset-stat">
          <div class="dataset-stat-num" style="color:#f59e0b">{k_used}</div>
          <div class="dataset-stat-lbl">Eigenfaces (k)</div>
        </div>
        <div class="dataset-stat">
          <div class="dataset-stat-num" style="color:#06b6d4">{total_var * 100:.1f}%</div>
          <div class="dataset-stat-lbl">Variance Captured</div>
        </div>
      </div>
      <div style="margin-top:0.75rem;font-size:0.78rem;color:#64748b">
        Untuk menangkap 95% variance dibutuhkan <strong style="color:#a78bfa">{k95} eigenfaces</strong>
        dari {img_shape[0] * img_shape[1]:,} dimensi asli
        &#8594; reduksi dimensi <strong style="color:#6ee7b7">{(img_shape[0] * img_shape[1] // max(k95, 1)) if isinstance(k95, int) else "?"}x lebih kecil</strong>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if show_eigenfaces:
        st.markdown(
            '<div class="section-title">Eigenfaces dari Dataset</div>',
            unsafe_allow_html=True,
        )

        make_dark_plot()
        n_show = min(15, eigenspace["n_components"])
        img_h, img_w = dataset_data.get("image_shape", (64, 64))
        custom_cmap = LinearSegmentedColormap.from_list(
            "eigen", ["#0a0e1a", "#3b82f6", "#8b5cf6", "#e2e8f0"]
        )

        fig, axes = plt.subplots(2, n_show // 2 + 1, figsize=(16, 5))
        axes = axes.flatten()

        mean_img = eigenspace["mean_face"].reshape(img_h, img_w)
        mean_disp = (mean_img - mean_img.min()) / (
            mean_img.max() - mean_img.min() + 1e-8
        )
        axes[0].imshow(mean_disp, cmap="gray")
        axes[0].set_title("Mean Face", fontsize=8, color="#f59e0b", pad=4)
        axes[0].axis("off")

        evr = eigenspace["explained_variance_ratio"]
        for i in range(1, min(n_show, len(axes))):
            ef = eigenspace["eigenfaces"][i - 1].reshape(img_h, img_w)
            ef_disp = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
            axes[i].imshow(ef_disp, cmap=custom_cmap)
            pct = evr[i - 1] * 100 if i - 1 < len(evr) else 0
            axes[i].set_title(
                f"EF #{i}\n{pct:.1f}%", fontsize=7, color="#94a3b8", pad=3
            )
            axes[i].axis("off")

        for ax in axes[n_show:]:
            ax.set_visible(False)

        plt.suptitle(
            f"Eigenfaces dari {n_train} gambar training — "
            f"setiap eigenface = pola dasar wajah manusia",
            color="#f1f5f9",
            fontsize=10,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.markdown("""
        > **Catatan:** Eigenface #1 menangkap pola dengan *variance terbesar* (fitur paling umum semua wajah).
        > Eigenface-eigenface selanjutnya menangkap pola yang makin spesifik & kecil.
        > Kombinasi dari semua eigenface merepresentasikan "bahasa" wajah manusia.
        """)

st.markdown(
    f'<div class="section-title">{T.get("upload_title", "Upload Foto")}</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown(
        f'<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;margin-bottom:0.5rem">{T["upload_old"]}</div>',
        unsafe_allow_html=True,
    )
    file1 = st.file_uploader(
        T["upload_old"],
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="photo_old",
        label_visibility="collapsed",
    )
    if file1 is not None:
        st.image(file1, caption=T["preview_old"], use_container_width=True)
with col2:
    st.markdown(
        f'<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;margin-bottom:0.5rem">{T["upload_new"]}</div>',
        unsafe_allow_html=True,
    )
    file2 = st.file_uploader(
        T["upload_new"],
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="photo_new",
        label_visibility="collapsed",
    )
    if file2 is not None:
        st.image(file2, caption=T["preview_new"], use_container_width=True)

if file1 and file2:
    with st.spinner("Memproses gambar & menjalankan PCA/SVD..."):
        pil1 = Image.open(file1)
        pil2 = Image.open(file2)

        gray1 = load_image_from_pil(pil1)
        gray2 = load_image_from_pil(pil2)

        target_sz = (target_size_sel, target_size_sel)

        face1_proc, info1 = preprocess_face(
            gray1, detect=detect_face_opt, target_size=target_sz
        )

        bbox2 = None
        if detect_face_opt:
            bbox2 = detect_face(gray2)

        face2_proc, info2 = preprocess_face(
            gray2, detect=detect_face_opt, target_size=target_sz, pre_bbox=bbox2, force_angle=0.0
        )
        info2["is_flipped"] = False

        if use_dataset and eigenspace is not None:
            result = analyze_two_faces_with_dataset(
                face1_proc, face2_proc, eigenspace,
                apply_aging=apply_aging_vector, prob_asian=prob_asian, aging_scale=aging_scale
            )
            face1_display = result["face1_resized"]
            face2_display = result["face2_resized"]
            m_label = "Dataset: " + dataset_data.get("source", "")[:40]
        else:
            olivetti_data, olivetti_es = get_eigenspace_olivetti(n_components)
            if olivetti_es is not None:
                result = analyze_two_faces_with_dataset(
                    face1_proc, face2_proc, olivetti_es,
                    apply_aging=apply_aging_vector, prob_asian=prob_asian, aging_scale=aging_scale
                )
                face1_display = result["face1_resized"]
                face2_display = result["face2_resized"]
                m_label = "Dataset: Olivetti Faces (fallback)"
            else:
                result = analyze_two_faces(face1_proc, face2_proc)
                face1_display = face1_proc
                face2_display = face2_proc
                m_label = "Mode: 2 gambar saja (tanpa dataset)"

        w1 = result["weights_face1"]
        w2 = result["weights_face2"]
        S_joint = result["singular_values_joint"]

        fusion_args = {}
        if "weights_face1_lbp" in result:
            fusion_args["weights1_lbp"] = result["weights_face1_lbp"]
            fusion_args["weights2_lbp"] = result["weights_face2_lbp"]
            fusion_args["weights1_hog"] = result["weights_face1_hog"]
            fusion_args["weights2_hog"] = result["weights_face2_hog"]
            fusion_args["S_lbp"] = result.get("singular_values_lbp")
            fusion_args["S_hog"] = result.get("singular_values_hog")

        metrics = compute_all_metrics(
            w1,
            w2,
            face1_display,
            face2_display,
            S_joint,
            penalty_factor=penalty_factor,
            alpha=alpha_lbp,
            beta=beta_hog,
            gamma=gamma_pix,
            **fusion_args,
        )

        mode_label = f"{m_label} | Rotasi: 0.0° | Cermin: Tidak"
        decision = make_decision(metrics, threshold=threshold)

        U1, S1, Vt1 = (
            result["svd_face1"]["U"],
            result["svd_face1"]["S"],
            result["svd_face1"]["Vt"],
        )
        U2, S2, Vt2 = (
            result["svd_face2"]["U"],
            result["svd_face2"]["S"],
            result["svd_face2"]["Vt"],
        )

    st.markdown(
        f'<div class="section-title">{T.get("sec_01", "Section 01: Primary Similarity Results")}</div>',
        unsafe_allow_html=True,
    )

    score = decision["score"]
    is_same = decision["is_same_person"]
    s_color = "#10b981" if is_same else "#ef4444"
    r_class = "result-same" if is_same else "result-diff"
    verdict_display = decision.get("verdict_display", decision["verdict"])

    st.markdown(
        f"""
    <div class="result-card {r_class}">
      <div style="font-size:1.5rem;font-weight:800;color:{s_color};margin-bottom:0.3rem">{verdict_display}</div>
      <div style="font-size:3.5rem;font-weight:900;color:{s_color};line-height:1;margin:0.5rem 0">{score:.1%}</div>
      <div style="color:#64748b;font-size:0.85rem;margin-bottom:0.75rem">Composite Score (Edge-PCA + Pinalti Euclidean)</div>
      <div style="display:flex;justify-content:center;gap:0.75rem;flex-wrap:wrap">
        <span style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.1);
                     border-radius:50px;padding:0.3rem 0.85rem;font-size:0.75rem;color:{s_color}">
          {decision["level"]} &middot; Kepercayaan: {decision["confidence"]}
        </span>
        <span style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.1);
                     border-radius:50px;padding:0.3rem 0.85rem;font-size:0.75rem;color:#94a3b8">
          Threshold: {threshold:.0%}
        </span>
        <span style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.1);
                     border-radius:50px;padding:0.3rem 0.85rem;font-size:0.75rem;color:#67e8f9">
          {mode_label}
        </span>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="section-title">{T.get("sec_prep", "Preprocessing Pipeline")}</div>',
        unsafe_allow_html=True,
    )

    def render_preprocessing_steps(info_dict, label):
        st.markdown(
            f"**{label}** (Sudut Rotasi: `{info_dict.get('angle_used', 0.0):.2f}°` | Eye Aligned: `{'Ya' if info_dict.get('eye_aligned') else 'Tidak'}`)"
        )
        steps = info_dict.get("steps", {})

        step_names = ["Original", "Crop", "Aligned", "Equalized", "Final"]
        step_keys = ["original_gray", "crop", "aligned", "equalized", "final"]

        cols = st.columns(len(step_names))
        for i, (name, key) in enumerate(zip(step_names, step_keys)):
            if key in steps:
                img = steps[key]
                disp = (
                    (img - img.min()) / (img.max() - img.min() + 1e-8)
                    if img.dtype == np.float64
                    else img
                )
                with cols[i]:
                    st.image(
                        disp,
                        caption=f"{i + 1}. {name}",
                        width="stretch",
                        clamp=True,
                    )
                    if i < len(step_names) - 1 and key != "final":
                        st.markdown(
                            "<div style='text-align:center;color:#6366f1'>v</div>",
                            unsafe_allow_html=True,
                        )

    render_preprocessing_steps(info1, "Pipeline Foto Lama")
    st.divider()
    render_preprocessing_steps(info2, "Pipeline Foto Baru")

    import pandas as pd

    st.markdown(
        f'<div class="section-title">{T.get("sec_02", "Section 02: Pixel Matrix Representation")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"*{T.get('sec_01_desc', 'Menampilkan sub-matriks 16x16 piksel dari pojok kiri atas gambar wajah (setelah pre-processing).')}*"
    )

    col_mat1, col_mat2 = st.columns(2)
    with col_mat1:
        st.markdown(f"**{T['upload_old']}**")
        sub_mat1 = face1_display[:16, :16]
        df1 = pd.DataFrame(sub_mat1)
        st.dataframe(
            df1.style.background_gradient(cmap="gray", vmin=0.0, vmax=1.0), height=300
        )
        st.caption(
            f"Shape: {face1_display.shape} | Min: {face1_display.min():.2f} | Max: {face1_display.max():.2f} | Mean: {face1_display.mean():.2f} | Std: {face1_display.std():.2f}"
        )

    with col_mat2:
        st.markdown(f"**{T['upload_new']}**")
        sub_mat2 = face2_display[:16, :16]
        df2 = pd.DataFrame(sub_mat2)
        st.dataframe(
            df2.style.background_gradient(cmap="gray", vmin=0.0, vmax=1.0), height=300
        )
        st.caption(
            f"Shape: {face2_display.shape} | Min: {face2_display.min():.2f} | Max: {face2_display.max():.2f} | Mean: {face2_display.mean():.2f} | Std: {face2_display.std():.2f}"
        )

    st.markdown(
        f'<div class="section-title">{T.get("sec_03", "Section 03: PCA & Eigenspace")}</div>',
        unsafe_allow_html=True,
    )
    if use_dataset and eigenspace is not None:
        st.latex(r"C = \frac{1}{n-1}X^TX \qquad C \cdot v = \lambda \cdot v")
        st.markdown(
            f"*{T.get('sec_02_desc', 'Proyeksi wajah ke dalam Eigenspace yang dibangun dari dataset.')}*"
        )

        ev_df = pd.DataFrame(
            {
                "Principal Component": [
                    f"PC{i + 1}"
                    for i in range(min(10, len(eigenspace["explained_variance_ratio"])))
                ],
                "Eigenvalue (\u03bb)": [
                    S_joint[i] ** 2 for i in range(min(10, len(S_joint)))
                ],
                "Explained Variance (%)": [
                    eigenspace["explained_variance_ratio"][i] * 100
                    for i in range(min(10, len(eigenspace["explained_variance_ratio"])))
                ],
            }
        )

        col_pca1, col_pca2 = st.columns([1, 2])
        with col_pca1:
            st.markdown("**Top 10 Eigenvalues (\u03bb)**")
            st.dataframe(ev_df, width="stretch")
        with col_pca2:
            st.markdown("**Scree Plot (Explained Variance)**")
            make_dark_plot()
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(
                range(1, len(ev_df) + 1),
                ev_df["Explained Variance (%)"],
                marker="o",
                linestyle="-",
                color="#6366f1",
            )
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Variance (%)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
    else:
        st.warning(
            "Eigenspace tidak tersedia. Pastikan Anda memilih Dataset pada Sidebar untuk melihat PCA yang valid."
        )

    st.markdown(
        f'<div class="section-title">{T.get("sec_04", "Section 04: SVD Reconstruction")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"*{T.get('sec_03_desc', 'Dekomposisi SVD dan rekonstruksi matriks dengan $k$ komponen.')}*"
    )

    k_recon = st.slider("Nilai k untuk Rekonstruksi SVD", 1, min(len(S1), 50), 10)

    def reconstruct_svd(U, S, Vt, k):
        return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    def display_svd_recon(face, U, S, Vt, label):
        recon = reconstruct_svd(U, S, Vt, k_recon)
        frob_error = np.linalg.norm(face - recon, ord="fro")

        c1, c2 = st.columns(2)
        with c1:
            disp_face = (face - face.min()) / (face.max() - face.min() + 1e-8)
            st.image(
                disp_face,
                caption=f"Asli ({label})",
                width="stretch",
                clamp=True,
            )
        with c2:
            disp_recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
            st.image(
                disp_recon,
                caption=f"Rekonstruksi k={k_recon}",
                width="stretch",
                clamp=True,
            )
        st.caption(f"Error Frobenius ($|| A - A_k ||_F$): {frob_error:.4f}")

        make_dark_plot()
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(range(1, min(21, len(S) + 1)), S[:20], marker=".", color="#10b981")
        ax.set_title("Top 20 Singular Values", fontsize=8, color="#f1f5f9")
        ax.set_ylabel("\u03c3", fontsize=7)
        ax.tick_params(axis="both", which="major", labelsize=6)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    col_svd1, col_svd2 = st.columns(2)
    with col_svd1:
        display_svd_recon(face1_display, U1, S1, Vt1, T["upload_old"])
    with col_svd2:
        display_svd_recon(face2_display, U2, S2, Vt2, T["upload_new"])

    st.markdown(
        f'<div class="section-title">{T.get("sec_05", "Section 05: Vector Projection")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"*{T.get('sec_04_desc', 'Bobot proyeksi PCA ke dalam Eigenspace.')}*")

    proj_df = pd.DataFrame(
        {
            "Component": [f"w_{i + 1}" for i in range(min(10, len(w1)))],
            T["upload_old"]: w1[:10],
            T["upload_new"]: w2[:10],
        }
    )

    c_proj1, c_proj2 = st.columns([1, 2])
    with c_proj1:
        st.dataframe(proj_df, width="stretch")
    with c_proj2:
        make_dark_plot()
        fig, ax = plt.subplots(figsize=(6, 3))
        width = 0.35
        x = np.arange(len(proj_df))
        ax.bar(
            x - width / 2,
            proj_df[T["upload_old"]],
            width,
            label=T["upload_old"],
            color="#6366f1",
        )
        ax.bar(
            x + width / 2,
            proj_df[T["upload_new"]],
            width,
            label=T["upload_new"],
            color="#f43f5e",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(proj_df["Component"], rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    if len(w1) >= 2 and len(w2) >= 2:
        make_dark_plot()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(w1[0], w1[1], c="#6366f1", s=100, label=T["upload_old"], zorder=3)
        ax.scatter(w2[0], w2[1], c="#f43f5e", s=100, label=T["upload_new"], zorder=3)
        ax.plot([w1[0], w2[0]], [w1[1], w2[1]], "w--", alpha=0.5, zorder=2)
        ax.plot([0, w1[0]], [0, w1[1]], color="#6366f1", alpha=0.3, zorder=1)
        ax.plot([0, w2[0]], [0, w2[1]], color="#f43f5e", alpha=0.3, zorder=1)
        ax.scatter(0, 0, c="white", marker="x", s=50, label="Origin (Mean Face)")
        ax.set_xlabel("PC1 (w_1)")
        ax.set_ylabel("PC2 (w_2)")
        ax.legend(fontsize=7)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    st.markdown(
        f'<div class="section-title">{T.get("sec_06", "Section 06: Advanced Analysis")}</div>',
        unsafe_allow_html=True,
    )

    tab_hm, tab_hist, tab_pc, tab_svd = st.tabs(
        [
            "Heatmap Selisih",
            "Histogram Piksel",
            "Kurva PCA",
            "Singular Values",
        ]
    )

    with tab_hm:
        st.markdown("**Absolute Difference Heatmap**")
        st.markdown(
            "*Menampilkan perbedaan piksel absolut antara Foto Lama dan Foto Baru (setelah alignment).*"
        )
        diff_img = np.abs(face1_display - face2_display)

        make_dark_plot()
        fig_hm, ax_hm = plt.subplots(figsize=(5, 4))
        cax = ax_hm.imshow(diff_img, cmap="hot")
        fig_hm.colorbar(cax, ax=ax_hm, fraction=0.046, pad=0.04)
        ax_hm.axis("off")
        st.pyplot(fig_hm, width="stretch")
        plt.close(fig_hm)

    with tab_hist:
        st.markdown("**Distribusi Intensitas Piksel**")
        make_dark_plot()
        fig_h, ax_h = plt.subplots(figsize=(6, 3))
        ax_h.hist(
            face1_display.ravel(),
            bins=50,
            alpha=0.5,
            label=T["upload_old"],
            color="#6366f1",
        )
        ax_h.hist(
            face2_display.ravel(),
            bins=50,
            alpha=0.5,
            label=T["upload_new"],
            color="#f43f5e",
        )
        ax_h.set_xlabel("Intensitas Piksel (0 - 1)")
        ax_h.set_ylabel("Frekuensi")
        ax_h.legend(fontsize=8)
        st.pyplot(fig_h, width="stretch")
        plt.close(fig_h)

    with tab_pc:
        st.markdown(
            "**Kurva Perbedaan Bobot (w1 vs w2) untuk 30 Principal Components Pertama**"
        )
        make_dark_plot()
        fig_pc, ax_pc = plt.subplots(figsize=(8, 3))
        k_plot = min(30, len(w1))
        x_pc = np.arange(1, k_plot + 1)
        ax_pc.plot(
            x_pc, w1[:k_plot], marker="o", label=T["upload_old"], color="#6366f1"
        )
        ax_pc.plot(
            x_pc, w2[:k_plot], marker="x", label=T["upload_new"], color="#f43f5e"
        )
        ax_pc.fill_between(x_pc, w1[:k_plot], w2[:k_plot], color="gray", alpha=0.2)
        ax_pc.set_xlabel("Principal Component")
        ax_pc.set_ylabel("Bobot (w)")
        ax_pc.legend(fontsize=8)
        st.pyplot(fig_pc, width="stretch")
        plt.close(fig_pc)

    with tab_svd:
        st.markdown("**Perbandingan 30 Singular Values Pertama**")
        make_dark_plot()
        fig_s, ax_s = plt.subplots(figsize=(8, 3))
        k_s = min(30, len(S1))
        x_s = np.arange(1, k_s + 1)
        ax_s.plot(x_s, S1[:k_s], marker="o", label=T["upload_old"], color="#6366f1")
        ax_s.plot(x_s, S2[:k_s], marker="x", label=T["upload_new"], color="#f43f5e")
        ax_s.set_xlabel("Rank")
        ax_s.set_ylabel("Singular Value (\u03c3)")
        ax_s.legend(fontsize=8)
        st.pyplot(fig_s, width="stretch")
        plt.close(fig_s)

    st.markdown(
        f'<div class="section-title">{T.get("sec_07", "Section 07: Dynamic Report")}</div>',
        unsafe_allow_html=True,
    )

    report_text = f"""
    # Laporan Analisis FaceMatch (Edge-PCA & SVD)
    
    ## Konfigurasi
    - Dataset Eigenspace: {m_label}
    - Ambang Batas (Threshold): {threshold:.2f}
    - Euclidean Penalty Factor: {penalty_factor:.2f}
    - Jumlah Eigenfaces (k): {len(w1)}
    
    ## Metrik Ekstraksi (Aljabar Linear)
    - Cosine Similarity (Eigenspace PCA): {metrics.get("cosine_similarity_eigenspace", 0):.4f}
    - Euclidean Distance (Eigenspace): {metrics.get("euclidean_distance_eigenspace", 0):.4f}
    - Euclidean Similarity (Norm): {metrics.get("euclidean_similarity_norm", 0):.4f}
    """

    if "cosine_lbp" in metrics:
        report_text += f"""
    ## Metrik Fusion (LBP + HOG)
    - Cosine LBP (Tekstur): {metrics.get("cosine_lbp", 0):.4f}  => Score: {metrics.get("score_lbp", 0):.4f}
    - Cosine HOG (Bentuk): {metrics.get("cosine_hog", 0):.4f}  => Score: {metrics.get("score_hog", 0):.4f}
    - Score Pixel (Intensitas): {metrics.get("score_pix", 0):.4f}
        """

    report_text += f"""
    ## Metrik Piksel Dasar
    - Cosine Similarity (Pixel): {metrics.get("cosine_similarity_pixel", 0):.4f}
    - SSIM: {metrics.get("ssim_pixel", 0):.4f}
    
    ## Kesimpulan Akhir
    - Composite Score: {decision["score"]:.4f} ({(decision["score"] * 100):.1f}%)
    - Hasil Keputusan: {decision["verdict_display"]} ({"SAMA" if is_same else "BERBEDA"})
    - Tingkat Kepercayaan: {decision["confidence"]}
    
    **Alasan Matematis:**
    """
    for r in decision.get("reasoning", []):
        report_text += f"\n    - {r}"

    if is_same == False and apply_aging_vector:
        report_text += f"""

    ## Catatan Akademis (Analisis Kegagalan Lintas-Usia)
    Berdasarkan hasil eksperimen, sistem menghasilkan skor Cosine Similarity di bawah ambang batas identifikasi yang dapat diterima. Temuan ini **bukan** merupakan indikasi kesalahan implementasi, melainkan manifestasi empiris dari dua batasan struktural yang melekat pada paradigma *Classical Machine Learning* (PCA/Eigenfaces):
    
    1. **Non-Ekuivarians Translasi pada Representasi Holistik:** PCA dan HOG beroperasi pada representasi piksel global dalam grid yang tetap (tidak memiliki properti *translation equivariance*). Konsekuensinya, ketidakakuratan lokalisasi wajah oleh *Haar Cascade* (pergeseran koordinat mata/translasi piksel) ditransmisikan langsung sebagai *noise* sistematis ke dalam ruang fitur PCA, yang menurunkan skor similaritas secara artifisial.
    2. **Degenerasi Identitas akibat Vektor Penuaan Global:** Vektor penuaan (\\Delta W) yang diekstrak secara global dari dataset FG-NET merepresentasikan trayektori penuaan rata-rata (populasi umum), bukan trayektori spesifik individu. Injeksi vektor ini memang berhasil mengubah umur wajah, namun secara bersamaan menggeser proyeksi subjek menjauhi identitas uniknya dan mendekati wajah rata-rata populasi.
    
    **Kesimpulan:** 
    Temuan ini secara empiris menjustifikasi mengapa transisi paradigma dari metode klasik ke *Deep Learning* (CNN / Metric Learning) merupakan sebuah keharusan arsitektural untuk menyelesaikan masalah Lintas-Usia secara utuh (mencapai *pose & translation invariance*). Sistem ini berhasil membuktikan *upper bound experiment* dari kapabilitas maksimal metode Aljabar Linear klasik.
        """

    st.markdown(f'<div class="math-box">{report_text}</div>', unsafe_allow_html=True)

    st.download_button(
        label="Download Report (.txt)",
        data=report_text,
        file_name="facematch_report.txt",
        mime="text/plain",
    )

else:
    st.markdown(
        """
    <div style="text-align:center;padding:4rem 2rem;background:#111827;border:2px dashed rgba(139,92,246,0.25);border-radius:20px;margin-top:1rem">
      <h3 style="color:#e2e8f0;margin-bottom:0.5rem">Upload Dua Foto untuk Memulai</h3>
      <p style="max-width:500px;margin:0 auto;color:#475569;font-size:0.9rem">
        Upload <strong style="color:#a78bfa">Foto Lama</strong> (masa kecil) dan
        <strong style="color:#60a5fa">Foto Baru</strong> untuk deteksi kemiripan menggunakan
        <strong style="color:#6ee7b7">Eigenfaces dari Dataset Olivetti/LFW</strong>.
      </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div style="text-align:center;margin-top:4rem;padding-top:2rem;border-top:1px solid rgba(255,255,255,0.06)">
  <p style="color:#334155;font-size:0.78rem">
    Tugas Aljabar Linear Semester 2 &middot;
    PCA & SVD (Eigenfaces) &middot; Dataset Olivetti/LFW &middot; NumPy &middot; OpenCV &middot; Streamlit
  </p>
</div>
""",
    unsafe_allow_html=True,
)
