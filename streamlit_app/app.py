import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pca_svd import (
    svd_decompose,
    compute_eigenfaces,
    project_to_eigenspace,
    reconstruct_from_eigenspace,
    load_olivetti_dataset,
    load_lfw_dataset,
    load_custom_selfie_dataset,
    load_pretrained_eigenspace,
    build_eigenspace_from_dataset,
    analyze_two_faces_with_dataset,
    analyze_two_faces,
    get_singular_values_info,
)
from core.face_utils import (
    load_image_from_pil, preprocess_face, detect_face, draw_face_box,
)
from core.similarity import compute_all_metrics, make_decision
from PIL import Image

st.set_page_config(
    page_title="FaceMatch PCA/SVD + Dataset",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
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

html, body, .stApp { background: var(--bg-primary) !important; font-family: 'Inter', sans-serif; }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1200px; }

@keyframes gradient {
    to { background-position: 200% center; }
}

.app-header {
    position: relative;
    padding: 3rem 0;
    margin-bottom: 2rem;
    text-align: center;
    border-bottom: 1px solid transparent;
    border-image: linear-gradient(to right, transparent, rgba(99, 102, 241, 0.25), transparent) 1;
}

.app-title {
    font-size: 3.5rem;
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

.result-card { border-radius: 24px; padding: 2.5rem; text-align: center; margin: 1.5rem 0; backdrop-filter: blur(8px); }
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
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <div style="position:relative;z-index:1">
    <div class="app-title">FaceMatch <span style="opacity:0.5">/</span> Edge-PCA & SVD</div>
    <div style="color:rgba(199,210,254,0.65);font-size:1.125rem;margin-bottom:1rem;font-weight:500;">
      Deteksi Kemiripan Foto Lama vs Foto Baru menggunakan implementasi Edge-PCA Aljabar Linear + Euclidean Tie-Breaker.
    </div>
    <div class="badge-row">
      <span class="badge">📐 Aljabar Linear</span>
      <span class="badge">🧮 Sobel Edge</span>
      <span class="badge">🔢 SVD</span>
      <span class="badge">⚖️ Pinalti Euclidean</span>
      <span class="badge">📊 PCA</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

def make_dark_plot():
    plt.rcParams.update({
        "figure.facecolor": "#111827", "axes.facecolor": "#1a2235",
        "axes.edgecolor": "#2d3748", "axes.labelcolor": "#94a3b8",
        "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
        "text.color": "#f1f5f9", "grid.color": "#1e293b", "grid.alpha": 0.5,
    })


def score_color(s):
    if s >= 0.85: return "#10b981"
    elif s >= 0.70: return "#22c55e"
    elif s >= 0.55: return "#f59e0b"
    elif s >= 0.40: return "#f97316"
    return "#ef4444"


def render_progress(score, label, color=None):
    c = color or score_color(score)
    pct = max(0.0, score) * 100
    st.markdown(f"""
    <div class="progress-container">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;font-size:0.78rem">
        <span style="color:#94a3b8">{label}</span>
        <span style="color:{c};font-weight:700;font-family:'JetBrains Mono',monospace">{pct:.1f}%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{c}88,{c})"></div>
      </div>
    </div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📦 Dataset Eigenspace")

    dataset_choice = st.radio(
        "Pilih dataset training:",
        ["Dataset Lokal (Selfie & ID)", "Olivetti Faces (Direkomendasikan)", "LFW (Labeled Faces in the Wild)", "Tanpa Dataset (2 gambar saja)"],
        index=0,
        help="Dataset digunakan untuk membangun eigenspace yang lebih kaya"
    )

    n_components = st.slider("Jumlah Eigenfaces (k)", 10, 100, 50, 5,
        help="Makin banyak = makin akurat, tapi lebih lambat")

    st.divider()
    st.markdown("## ⚙️ Parameter")

    threshold = st.slider("Ambang Batas Kemiripan", 0.40, 0.95, 0.68, 0.01)
    detect_face_opt = st.toggle("Auto Deteksi Wajah", value=True)
    show_eigenfaces = st.toggle("Tampilkan Eigenfaces Dataset", value=True)
    show_math = st.toggle("Penjelasan Matematis", value=True)
    show_svd = st.toggle("Grafik Singular Values", value=True)

    st.divider()
    st.markdown("## 📚 Kenapa Butuh Dataset?")
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
    eigenspace = build_eigenspace_from_dataset(data["images"], n_components=k, target_size=(64, 64))
    return data, eigenspace


@st.cache_resource(show_spinner=False)
def get_eigenspace_lfw(k):
    data = load_lfw_dataset(min_faces=20, resize=0.4)
    if data is None:
        return None, None
    eigenspace = build_eigenspace_from_dataset(data["images"], n_components=k,
    target_size=data["image_shape"])
    return data, eigenspace

@st.cache_resource(show_spinner=False)
def get_eigenspace_custom(k):
    # Coba muat model pre-trained (Colab)
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_eigenspace.npz")
    eigenspace = load_pretrained_eigenspace(model_path)
    
    if eigenspace is not None:
        data = {
            "source": eigenspace["source"],
            "description": eigenspace["description"],
            "n_samples": eigenspace["n_samples"],
            "n_people": "?",
            "image_shape": eigenspace["image_shape"]
        }
        return data, eigenspace

    # Fallback ke training lokal jika model tidak ada (namun dataset lokal masih ada)
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Selfie & id data - public sample")
    data = load_custom_selfie_dataset(base_path, target_size=(128, 128))
    if data is None:
        return None, None
    eigenspace = build_eigenspace_from_dataset(data["images"], n_components=k, target_size=(128, 128))
    return data, eigenspace

use_dataset = "Tanpa Dataset" not in dataset_choice
dataset_data = None
eigenspace   = None

if use_dataset:
    with st.spinner("Memuat dataset & membangun eigenspace..."):
        if "Lokal" in dataset_choice:
            dataset_data, eigenspace = get_eigenspace_custom(n_components)
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

    st.markdown('<div class="section-title">📦 Dataset Eigenspace</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="dataset-card">
      <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem">
        <span style="font-size:1.5rem">✅</span>
        <div>
          <div style="font-weight:700;color:#f1f5f9">{dataset_data.get("source","Dataset")}</div>
          <div style="font-size:0.8rem;color:#64748b">{dataset_data.get("description","")[:120]}...</div>
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
          <div class="dataset-stat-num" style="color:#06b6d4">{total_var*100:.1f}%</div>
          <div class="dataset-stat-lbl">Variance Captured</div>
        </div>
      </div>
      <div style="margin-top:0.75rem;font-size:0.78rem;color:#64748b">
        Untuk menangkap 95% variance dibutuhkan <strong style="color:#a78bfa">{k95} eigenfaces</strong>
        dari {img_shape[0]*img_shape[1]:,} dimensi asli
        &#8594; reduksi dimensi <strong style="color:#6ee7b7">{(img_shape[0]*img_shape[1]//max(k95,1)) if isinstance(k95, int) else '?'}x lebih kecil</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if show_eigenfaces:
        st.markdown('<div class="section-title">👁️ Eigenfaces dari Dataset</div>', unsafe_allow_html=True)

        make_dark_plot()
        n_show = min(15, eigenspace["n_components"])
        img_h, img_w = dataset_data.get("image_shape", (64, 64))
        custom_cmap = LinearSegmentedColormap.from_list(
            "eigen", ["#0a0e1a", "#3b82f6", "#8b5cf6", "#e2e8f0"])

        fig, axes = plt.subplots(2, n_show // 2 + 1, figsize=(16, 5))
        axes = axes.flatten()

        mean_img = eigenspace["mean_face"].reshape(img_h, img_w)
        mean_disp = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min() + 1e-8)
        axes[0].imshow(mean_disp, cmap="gray")
        axes[0].set_title("Mean Face", fontsize=8, color="#f59e0b", pad=4)
        axes[0].axis("off")

        evr = eigenspace["explained_variance_ratio"]
        for i in range(1, min(n_show, len(axes))):
            ef = eigenspace["eigenfaces"][i-1].reshape(img_h, img_w)
            ef_disp = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
            axes[i].imshow(ef_disp, cmap=custom_cmap)
            pct = evr[i-1] * 100 if i-1 < len(evr) else 0
            axes[i].set_title(f"EF #{i}\n{pct:.1f}%", fontsize=7, color="#94a3b8", pad=3)
            axes[i].axis("off")

        for ax in axes[n_show:]:
            ax.set_visible(False)

        plt.suptitle(
            f"Eigenfaces dari {n_train} gambar training — "
            f"setiap eigenface = pola dasar wajah manusia",
            color="#f1f5f9", fontsize=10, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""
        > 💡 **Catatan:** Eigenface #1 menangkap pola dengan *variance terbesar* (fitur paling umum semua wajah).
        > Eigenface-eigenface selanjutnya menangkap pola yang makin spesifik & kecil.
        > Kombinasi dari semua eigenface merepresentasikan "bahasa" wajah manusia.
        """)

st.markdown('<div class="section-title">📸 Upload Foto</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown('<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;margin-bottom:0.5rem">📷 Foto Lama (Masa Kecil)</div>', unsafe_allow_html=True)
    file1 = st.file_uploader("Upload foto pertama", type=["jpg","jpeg","png","bmp","webp"],
                              key="photo_old", label_visibility="collapsed")
with col2:
    st.markdown('<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;margin-bottom:0.5rem">📱 Foto Baru (Saat Ini)</div>', unsafe_allow_html=True)
    file2 = st.file_uploader("Upload foto kedua", type=["jpg","jpeg","png","bmp","webp"],
                              key="photo_new", label_visibility="collapsed")

if file1 and file2:
    with st.spinner("Memproses gambar & menjalankan PCA/SVD..."):
        pil1 = Image.open(file1)
        pil2 = Image.open(file2)

        gray1 = load_image_from_pil(pil1)
        gray2 = load_image_from_pil(pil2)

        face1_proc, info1 = preprocess_face(gray1, detect=detect_face_opt)
        # Rotation Search on Photo 2 for Pose Invariance
        best_cos = -1.0
        best_metrics = None
        best_decision = None
        best_face2_proc = None
        best_info2 = None
        best_result = None

        # To avoid re-detecting bbox multiple times, detect once:
        bbox2 = None
        if detect_face_opt:
            bbox2 = detect_face(gray2)

        for angle in [0.0, -10.0, 10.0, -5.0, 5.0]:
            f2_proc, i2 = preprocess_face(gray2, detect=detect_face_opt, angle=angle, pre_bbox=bbox2)
            
            if use_dataset and eigenspace is not None:
                res = analyze_two_faces_with_dataset(face1_proc, f2_proc, eigenspace)
                f1_disp = res["face1_resized"]
                f2_disp = res["face2_resized"]
                target_sz = eigenspace.get("target_size", (64, 64))
                m_label = "Dataset: " + dataset_data.get("source", "")[:40]
            else:
                res = analyze_two_faces(face1_proc, f2_proc)
                f1_disp = face1_proc
                f2_disp = f2_proc
                target_sz = (128, 128)
                m_label = "Mode: 2 gambar saja (tanpa dataset)"
                
            w1 = res["weights_face1"]
            w2 = res["weights_face2"]
            mets = compute_all_metrics(w1, w2, f1_disp, f2_disp)
            
            if mets["cosine_similarity_eigenspace"] > best_cos:
                best_cos = mets["cosine_similarity_eigenspace"]
                best_metrics = mets
                best_result = res
                best_face2_proc = f2_proc
                best_info2 = i2
                mode_label = f"{m_label} | Rotasi: {angle}°"
                
        # Commit the best rotation results
        face2_proc = best_face2_proc
        info2 = best_info2
        result = best_result
        metrics = best_metrics
        decision = make_decision(metrics, threshold=threshold)
        
        face1_display = result["face1_resized"] if use_dataset else face1_proc
        face2_display = result["face2_resized"] if use_dataset else face2_proc
        
        w1 = result["weights_face1"]
        w2 = result["weights_face2"]
        U1, S1, Vt1 = result["svd_face1"]["U"], result["svd_face1"]["S"], result["svd_face1"]["Vt"]
        U2, S2, Vt2 = result["svd_face2"]["U"], result["svd_face2"]["S"], result["svd_face2"]["Vt"]

    st.markdown('<div class="section-title">🖼️ Pratinjau Foto</div>', unsafe_allow_html=True)

    pc1, pc2 = st.columns(2, gap="large")
    with pc1:
        st.markdown("**📷 Foto Lama**")
        disp1 = np.array(pil1.convert("RGB"))
        if info1.get("bbox"):
            disp1 = draw_face_box(disp1, info1["bbox"])
        st.image(disp1, use_container_width=True)
        st.caption("Wajah terdeteksi" if info1["face_detected"] else "Gambar penuh (wajah tidak terdeteksi)")

    with pc2:
        st.markdown("**📱 Foto Baru**")
        disp2 = np.array(pil2.convert("RGB"))
        if info2.get("bbox"):
            disp2 = draw_face_box(disp2, info2["bbox"])
        st.image(disp2, use_container_width=True)
        st.caption("Wajah terdeteksi" if info2["face_detected"] else "Gambar penuh (wajah tidak terdeteksi)")

    st.markdown('<div class="section-title">🎯 Hasil Analisis</div>', unsafe_allow_html=True)

    score    = decision["score"]
    is_same  = decision["is_same_person"]
    s_color  = "#10b981" if is_same else "#ef4444"
    r_class  = "result-same" if is_same else "result-diff"
    verdict_display = decision.get("verdict_display", decision["verdict"])

    st.markdown(f"""
    <div class="result-card {r_class}">
      <div style="font-size:3rem;margin-bottom:0.5rem">{'&#x2705;' if is_same else '&#x274c;'}</div>
      <div style="font-size:1.5rem;font-weight:800;color:{s_color};margin-bottom:0.3rem">{verdict_display}</div>
      <div style="font-size:3.5rem;font-weight:900;color:{s_color};line-height:1;margin:0.5rem 0">{score:.1%}</div>
      <div style="color:#64748b;font-size:0.85rem;margin-bottom:0.75rem">Composite Score (Edge-PCA + Pinalti Euclidean)</div>
      <div style="display:flex;justify-content:center;gap:0.75rem;flex-wrap:wrap">
        <span style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.1);
                     border-radius:50px;padding:0.3rem 0.85rem;font-size:0.75rem;color:{s_color}">
          {decision['level']} &middot; Kepercayaan: {decision['confidence']}
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
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Metrik Kemiripan</div>', unsafe_allow_html=True)

    mc1, mc2 = st.columns(2, gap="large")
    m = metrics

    with mc1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Eigenspace (PCA/SVD)**")
        cos_e = m["cosine_similarity_eigenspace"]
        render_progress(max(0, cos_e), f"Cosine Similarity Eigenspace ({len(w1)}D)")
        render_progress(m["euclidean_similarity_norm"], "Euclidean Similarity (Normalized)")
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:0.5rem 0;font-size:0.8rem;border-top:1px solid rgba(255,255,255,0.06);margin-top:0.5rem">
          <span style="color:#94a3b8">Euclidean Distance</span>
          <span style="font-family:'JetBrains Mono',monospace;color:#f1f5f9">{m['euclidean_distance_eigenspace']:.4f}</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:0.5rem 0;font-size:0.8rem;border-top:1px solid rgba(255,255,255,0.06)">
          <span style="color:#94a3b8">Dimensi Eigenspace</span>
          <span style="font-family:'JetBrains Mono',monospace;color:#a78bfa">{len(w1)} komponen</span>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with mc2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Pixel & Composite**")
        render_progress(m["ssim_pixel"], "SSIM (Structural Similarity)")
        render_progress(max(0, m["cosine_similarity_pixel"]), "Cosine Similarity (Pixel)")
        render_progress(score, "Composite Score (Final)", score_color(score))
        st.markdown('</div>', unsafe_allow_html=True)

    if show_math:
        st.markdown('<div class="section-title">📐 Penjelasan Matematis</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🔢 Pipeline Lengkap", "📊 Eigenvalue & Variance", "📏 Similarity Score"])

        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                dataset_size_info = dataset_data.get("n_samples", 2) if use_dataset else 2
                shape_info = dataset_data.get("image_shape", (64,64)) if use_dataset else (128, 128)
                st.markdown("#### Langkah 1: Bangun Eigenspace")
                st.markdown(f"""
                <div class="math-box"># Matriks dataset A ({dataset_size_info} gambar x {shape_info[0]*shape_info[1]} piksel):
A shape = ({dataset_size_info}, {shape_info[0]*shape_info[1]})

# Centrasi:
mean_face = mean(A, axis=0)   # shape ({shape_info[0]*shape_info[1]},)
A_centered = A - mean_face    # shape ({dataset_size_info}, {shape_info[0]*shape_info[1]})

# SVD: A_centered = U Sigma Vt
U  shape = ({dataset_size_info}, {min(dataset_size_info, shape_info[0]*shape_info[1])})
S  shape = ({min(dataset_size_info, shape_info[0]*shape_info[1])},)
Vt shape = ({min(dataset_size_info, shape_info[0]*shape_info[1])}, {shape_info[0]*shape_info[1]})

# Ambil k={n_components} eigenfaces = k baris pertama Vt:
eigenfaces shape = ({n_components}, {shape_info[0]*shape_info[1]})</div>
                """, unsafe_allow_html=True)

            with col_b:
                st.markdown("#### Langkah 2: Proyeksi Foto Baru")
                k_disp = min(3, len(w1))
                st.markdown(f"""
                <div class="math-box"># Untuk setiap foto yang diupload:
face shape = ({shape_info[0]*shape_info[1]},)

# Centrasi dengan mean dari dataset:
face_centered = face - mean_face

# Proyeksi ke eigenspace:
w = eigenfaces @ face_centered
w shape = ({len(w1)},)   # dari {shape_info[0]*shape_info[1]} -> {len(w1)} dimensi!

# Foto Lama: w1 = [{', '.join(f'{v:.3f}' for v in w1[:k_disp])}...]
# Foto Baru: w2 = [{', '.join(f'{v:.3f}' for v in w2[:k_disp])}...]

# Lalu bandingkan w1 vs w2 dengan cosine similarity</div>
                """, unsafe_allow_html=True)

        with tab2:
            if use_dataset and eigenspace is not None:
                evr = eigenspace["explained_variance_ratio"]
                eigenvalues = eigenspace["eigenvalues"]
                cumvar = np.cumsum(evr) * 100

                col_x, col_y = st.columns(2)
                with col_x:
                    make_dark_plot()
                    fig_ev, ax_ev = plt.subplots(figsize=(6, 3.5))
                    x_range = np.arange(1, len(eigenvalues) + 1)
                    ax_ev.bar(x_range, eigenvalues, color="#8b5cf6", alpha=0.85)
                    ax_ev.set_facecolor("#1a2235")
                    ax_ev.set_title("Eigenvalues (λ)", color="#f1f5f9", fontweight="bold", fontsize=10)
                    ax_ev.set_xlabel("Komponen ke-i")
                    ax_ev.set_ylabel("Eigenvalue (λᵢ)")
                    ax_ev.grid(True, alpha=0.3)
                    for sp in ["top","right"]: ax_ev.spines[sp].set_visible(False)
                    for sp in ["bottom","left"]: ax_ev.spines[sp].set_color("#2d3748")
                    plt.tight_layout()
                    st.pyplot(fig_ev, use_container_width=True)
                    plt.close(fig_ev)

                with col_y:
                    fig_cv, ax_cv = plt.subplots(figsize=(6, 3.5))
                    ax_cv.plot(x_range, cumvar, "o-", color="#10b981", linewidth=2, markersize=4)
                    ax_cv.axhline(y=95, color="#ef4444", linestyle="--", linewidth=1.5, label="95%")
                    ax_cv.fill_between(x_range, cumvar, alpha=0.15, color="#10b981")
                    ax_cv.set_facecolor("#1a2235")
                    ax_cv.set_title("Cumulative Variance (%)", color="#f1f5f9", fontweight="bold", fontsize=10)
                    ax_cv.set_xlabel("Jumlah Komponen k")
                    ax_cv.set_ylabel("Variance Explained (%)")
                    ax_cv.set_ylim(0, 105)
                    ax_cv.legend(framealpha=0.3, fontsize=9)
                    ax_cv.grid(True, alpha=0.3)
                    for sp in ["top","right"]: ax_cv.spines[sp].set_visible(False)
                    for sp in ["bottom","left"]: ax_cv.spines[sp].set_color("#2d3748")
                    plt.tight_layout()
                    st.pyplot(fig_cv, use_container_width=True)
                    plt.close(fig_cv)

                st.markdown(f"""
                <div class="math-box"># Eigenvalue terbesar (5 pertama):
{chr(10).join(f'lambda_{i+1} = {eigenvalues[i]:.4f}   ({evr[i]*100:.2f}% variance)' for i in range(min(5, len(eigenvalues))))}

# Cumulative variance dengan k={n_components} eigenfaces:
sum(lambda_1..lambda_{n_components}) = {evr.sum()*100:.2f}% variance terkandung

# Interpretasi: {n_components} eigenfaces meringkas {shape_info[0]*shape_info[1]:,} dimensi gambar
# menjadi representasi {evr.sum()*100:.1f}% akurat hanya dengan {n_components} angka!</div>
                """, unsafe_allow_html=True)

        with tab3:
            cos_val = m["cosine_similarity_eigenspace"]
            euc_val = m["euclidean_distance_eigenspace"]
            k_disp = min(5, len(w1))
            st.markdown(f"""
            <div class="math-box"># Cosine Similarity di Eigenspace {len(w1)}D:
w1 = [{', '.join(f'{v:.3f}' for v in w1[:k_disp])}, ...]   (Foto Lama)
w2 = [{', '.join(f'{v:.3f}' for v in w2[:k_disp])}, ...]   (Foto Baru)

dot(w1, w2)  = {float(np.dot(w1, w2)):.4f}
norm(w1)     = {float(np.linalg.norm(w1)):.4f}
norm(w2)     = {float(np.linalg.norm(w2)):.4f}

cos(theta)   = {float(np.dot(w1,w2)):.4f} / ({float(np.linalg.norm(w1)):.4f} x {float(np.linalg.norm(w2)):.4f})
             = {cos_val:.4f}  ->  {cos_val*100:.2f}%

Euclidean distance:
d = norm(w1 - w2) = {euc_val:.4f}

Composite Score (Edge-PCA + Tie-Breaker):
= max(0, cos_val) x penalty_factor
= {score:.4f}  ->  {score*100:.2f}%

Threshold = {threshold:.0%}  ->  {'SAMA' if is_same else 'BERBEDA'}</div>
            """, unsafe_allow_html=True)
            for r in decision.get("reasoning", []):
                st.markdown(f"- {r}")

    if show_svd:
        st.markdown('<div class="section-title">📈 SVD Singular Values per Foto</div>', unsafe_allow_html=True)

        make_dark_plot()
        k_show = min(30, len(S1), len(S2))
        x_r = np.arange(1, k_show + 1)

        fig_sv, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        fig_sv.patch.set_facecolor("#111827")

        for ax, S, lbl, c in [(ax1, S1, "Foto Lama", "#3b82f6"), (ax2, S2, "Foto Baru", "#8b5cf6")]:
            ax.plot(x_r, S[:k_show], "o-", color=c, lw=2, ms=4)
            ax.fill_between(x_r, S[:k_show], alpha=0.15, color=c)
            ax.set_facecolor("#1a2235")
            ax.set_title(f"Singular Values — {lbl}", color="#f1f5f9", fontweight="bold", fontsize=10)
            ax.set_xlabel("Rank")
            ax.set_ylabel("Nilai Singular (sigma)")
            ax.grid(True, alpha=0.3)
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            for sp in ["bottom","left"]: ax.spines[sp].set_color("#2d3748")

        plt.tight_layout()
        st.pyplot(fig_sv, use_container_width=True)
        plt.close(fig_sv)

    st.markdown('<div class="section-title">🖼️ Rekonstruksi dari Eigenspace</div>', unsafe_allow_html=True)

    make_dark_plot()
    r1_img = result["reconstructed_face1"]
    r2_img = result["reconstructed_face2"]

    fig_rec, axes = plt.subplots(2, 3, figsize=(11, 7))
    fig_rec.patch.set_facecolor("#111827")

    imgs_top = [face1_display, r1_img, np.abs(face1_display - np.clip(r1_img, 0, 1))]
    imgs_bot = [face2_display, r2_img, np.abs(face2_display - np.clip(r2_img, 0, 1))]
    titles_top = ["Foto Lama\n(asli preprocessed)", f"Rekonstruksi\n(k={len(w1)} eigenfaces)", "Error\n|asli - rekonstruksi|"]
    titles_bot = ["Foto Baru\n(asli preprocessed)", f"Rekonstruksi\n(k={len(w2)} eigenfaces)", "Error\n|asli - rekonstruksi|"]

    for col_i, (img, title) in enumerate(zip(imgs_top, titles_top)):
        ax = axes[0, col_i]
        cmap = "hot" if col_i == 2 else "gray"
        ax.imshow(np.clip(img, 0, 1), cmap=cmap)
        ax.set_title(title, fontsize=8, color="#94a3b8", pad=4)
        ax.axis("off")

    for col_i, (img, title) in enumerate(zip(imgs_bot, titles_bot)):
        ax = axes[1, col_i]
        cmap = "hot" if col_i == 2 else "gray"
        ax.imshow(np.clip(img, 0, 1), cmap=cmap)
        ax.set_title(title, fontsize=8, color="#94a3b8", pad=4)
        ax.axis("off")

    plt.suptitle(
        f"Rekonstruksi Wajah dari Eigenspace — {len(w1)} eigenfaces dari {mode_label}",
        color="#f1f5f9", fontsize=10, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    st.pyplot(fig_rec, use_container_width=True)
    plt.close(fig_rec)

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;background:#111827;border:2px dashed rgba(139,92,246,0.25);border-radius:20px;margin-top:1rem">
      <div style="font-size:3rem;margin-bottom:1rem">🔬</div>
      <h3 style="color:#e2e8f0;margin-bottom:0.5rem">Upload Dua Foto untuk Memulai</h3>
      <p style="max-width:500px;margin:0 auto;color:#475569;font-size:0.9rem">
        Upload <strong style="color:#a78bfa">Foto Lama</strong> (masa kecil) dan
        <strong style="color:#60a5fa">Foto Baru</strong> untuk deteksi kemiripan menggunakan
        <strong style="color:#6ee7b7">Eigenfaces dari Dataset Olivetti/LFW</strong>.
      </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;margin-top:4rem;padding-top:2rem;border-top:1px solid rgba(255,255,255,0.06)">
  <p style="color:#334155;font-size:0.78rem">
    Tugas Aljabar Linear Semester 2 &middot;
    PCA & SVD (Eigenfaces) &middot; Dataset Olivetti/LFW &middot; NumPy &middot; OpenCV &middot; Streamlit
  </p>
</div>
""", unsafe_allow_html=True)
