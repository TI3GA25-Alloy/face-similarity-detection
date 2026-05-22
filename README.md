(ver)

# 🔬 FaceMatch — Deteksi Kemiripan Foto Menggunakan PCA & SVD

> **Tugas Mata Kuliah Aljabar Linear — Semester 2**  
> Implementasi *Eigenfaces* untuk mendeteksi kemiripan antara foto lama (masa kecil) dan foto baru (saat ini).

---

## 📐 Konsep Aljabar Linear yang Diimplementasikan

| Konsep | Penerapan |
|--------|-----------|
| **SVD**: `A = U Σ Vᵀ` | Dekomposisi matriks gambar ke komponen utama |
| **Eigenvalue (λ)** | Mengukur seberapa penting setiap eigenface |
| **Eigenvector (v)** | Arah komponen utama (pola dasar wajah) |
| **Matriks Kovarians** | `C = (1/n) AᵀA` — basis untuk PCA |
| **Proyeksi Vektor** | `w = eigenfaces @ (face - mean_face)` |
| **Cosine Similarity** | `cos(θ) = (a·b)/(‖a‖‖b‖)` |
| **Euclidean Distance** | `d = ‖a - b‖₂` |

---

## 🏗️ Arsitektur Folder

```
detection-similarity-between-old-photos-and-new-photos/
│
├── streamlit_app/              ← 🐍 Versi Streamlit (lokal)
│   ├── app.py                  ← Entry point utama
│   ├── core/
│   │   ├── pca_svd.py          ← Logika PCA & SVD (NumPy murni)
│   │   ├── face_utils.py       ← Preprocessing OpenCV
│   │   └── similarity.py       ← Cosine similarity, SSIM, composite score
│   └── requirements.txt
│
├── vercel_app/                 ← ▲ Versi Vercel (Next.js)
│   ├── pages/
│   │   ├── _app.tsx
│   │   └── index.tsx           ← Halaman utama dengan visualisasi
│   ├── api/
│   │   └── analyze.py          ← Python serverless function
│   ├── styles/globals.css      ← Design system premium dark
│   ├── package.json
│   ├── requirements.txt
│   └── vercel.json
│
└── README.md
```

---

## 🚀 Cara Menjalankan

### 🐍 Versi Streamlit (Lokal)

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Buka browser: `http://localhost:8501`

### ▲ Versi Vercel (Next.js)

**Development lokal:**
```bash
cd vercel_app
npm install
npm run dev
```
Buka browser: `http://localhost:3000`

**Deploy ke Vercel:**
```bash
npm install -g vercel
vercel --prod
```

---

## 🔬 Cara Kerja Pipeline

```
Foto Lama  ──┐
              ├──► Grayscale → Haar Cascade → Crop → Resize 128×128 → Normalize
Foto Baru  ──┘
                    │
                    ▼
             Stack jadi matriks (2 × 16384)
                    │
                    ▼
             SVD: A = U Σ Vᵀ  ←── Eigenfaces = baris Vᵀ
                    │
                    ▼
             Proyeksi: w = eigenfaces @ (face - mean_face)
                    │
            ┌───────┴───────┐
            ▼               ▼
       w₁ (foto lama)  w₂ (foto baru)
            │               │
            └───────┬───────┘
                    ▼
         Cosine Similarity: cos(θ) = (w₁·w₂)/(‖w₁‖‖w₂‖)
         Euclidean Distance: d = ‖w₁ - w₂‖₂
         SSIM, Composite Score
                    │
                    ▼
         score ≥ threshold → ✅ Orang yang Sama
         score < threshold → ❌ Orang yang Berbeda
```

---

## 📦 Dependencies

### Python (Streamlit & API)
| Library | Peran |
|---------|-------|
| `numpy` | SVD (`np.linalg.svd`), operasi matriks |
| `opencv-python` | Deteksi wajah (Haar Cascade), preprocessing |
| `Pillow` | Load & konversi gambar |
| `matplotlib` | Visualisasi eigenfaces & grafik |
| `streamlit` | UI web interaktif |

### JavaScript (Vercel)
| Library | Peran |
|---------|-------|
| `next` | Framework React SSR |
| `recharts` | Grafik singular values & variance |
| `framer-motion` | Animasi transisi |
| `react-dropzone` | Drag & drop upload |

---

## ❓ Mengapa Tidak Perlu Dataset?

PCA & SVD bukan model machine learning — ini adalah **operasi matematika murni**.  
- ❌ Tidak perlu data training ribuan foto  
- ❌ Tidak perlu GPU  
- ✅ Cukup upload 2 foto, matematika yang bekerja

Berbeda dengan deep learning (FaceNet, ArcFace) yang butuh jutaan foto untuk training.

---

## 📊 Interpretasi Skor

| Composite Score | Level | Interpretasi |
|----------------|-------|--------------|
| ≥ 95% | Identik | Hampir pasti foto yang sama |
| ≥ 85% | Sangat Mirip | Kemungkinan besar orang yang sama |
| ≥ 70% | Mirip | Bisa orang yang sama (threshold default) |
| ≥ 55% | Kurang Mirip | Tidak pasti |
| < 55% | Tidak Mirip | Kemungkinan besar orang berbeda |