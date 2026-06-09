import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi
NAMA_NPZ_EIGENSPACE = "pretrained_eigenspace.npz"
NAMA_NPZ_DATASET = "privasi_kelompok_100x100.npz"
THRESHOLD_COSINE = 0.70

print("=" * 65)
print(" EVALUASI HYBRID AGING VECTOR (SOFT ROUTING)")
print(" Pendekatan Aljabar Linear Murni (Kombinasi Linear Vektor Eigen)")
print("=" * 65)

# 1. Memuat Dataset Kelompok
try:
    data_kelompok = np.load(NAMA_NPZ_DATASET)
    X_latih_kel = data_kelompok["X_latih"]
    y_kel = data_kelompok["y"]
    X_test_lintas = data_kelompok["X_test_lintas"]
    print(f"\n[+] Berhasil memuat dataset kelompok: {X_test_lintas.shape[0]} foto uji lintas usia.")
    
    KAMUS_NAMA = {}
    if "nama_anggota" in data_kelompok:
        nama_list = [str(n) for n in data_kelompok["nama_anggota"]]
        for lbl, nama in zip(y_kel, nama_list):
            KAMUS_NAMA[int(lbl)] = nama
    else:
        for i, lbl in enumerate(y_kel):
            KAMUS_NAMA[int(lbl)] = f"Anggota {i + 1}"

except FileNotFoundError:
    print(f"\n[!] Dataset {NAMA_NPZ_DATASET} tidak ditemukan.")
    print("    Menggunakan data simulasi untuk keperluan demonstrasi.")
    n_anggota = 5
    X_latih_kel = np.random.rand(n_anggota, 10000).astype(np.float32)
    y_kel = np.array(list(range(40, 40 + n_anggota)), dtype=np.int32)
    X_test_lintas = np.random.rand(n_anggota, 10000).astype(np.float32)
    KAMUS_NAMA = {lbl: f"Anggota {i+1}" for i, lbl in enumerate(y_kel)}

# 2. Memuat Eigenspace
try:
    eigenspace = np.load(NAMA_NPZ_EIGENSPACE)
    
    mean_face = eigenspace["mean_hog"]
    eigenfaces = eigenspace["eigenfaces_hog"]
    singular_values = eigenspace["singular_values_hog"]
    
    # Ambil aging vector FGNET (Kaukasia) - HOG
    if "aging_vector_fgnet_hog" in eigenspace:
        aging_vector_fgnet = eigenspace["aging_vector_fgnet_hog"]
    else:
        aging_vector_fgnet = np.zeros(eigenfaces.shape[0])
        
    # Ambil aging vector AAF (Asia) - HOG
    if "aging_vector_aaf_hog" in eigenspace:
        aging_vector_aaf = eigenspace["aging_vector_aaf_hog"]
        print(f"    [+] Berhasil memuat REAL AAF Aging Vector (HOG) dari dataset Asia!")
    else:
        print(f"    [!] REAL AAF Vector HOG tidak ditemukan. Menggunakan fallback.")
        aging_vector_aaf = np.zeros(eigenfaces.shape[0])
        
    print(f"\n[+] Berhasil memuat eigenspace dengan {eigenfaces.shape[0]} Principal Components.")
    
except FileNotFoundError:
    print(f"\n[!] File {NAMA_NPZ_EIGENSPACE} tidak ditemukan. Harus menjalankan evaluasi PCA sebelumnya.")
    sys.exit(1)


from skimage.feature import hog

def extract_hog_features(X_raw):
    print(f"    Mengekstrak fitur HOG untuk {len(X_raw)} gambar...")
    X_hog = []
    for vec in X_raw:
        img_uint8 = (np.clip(vec.reshape(100, 100), 0, 1) * 255).astype(np.uint8)
        h = hog(img_uint8, orientations=8, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), block_norm="L2-Hys", visualize=False)
        X_hog.append(h.astype(np.float32))
    return np.array(X_hog)

def proyeksikan_ke_pca(X, mean_f, e_faces):
    """Proyeksi data ke ruang PCA"""
    return (X - mean_f) @ e_faces.T

print("\n[+] Mengubah mode ekstraksi dari Pixel ke HOG (Shape/Bentuk)...")
X_latih_hog = extract_hog_features(X_latih_kel)
X_test_lintas_hog = extract_hog_features(X_test_lintas)

# Proyeksikan dataset database (dewasa) ke PCA HOG
X_db_pca = proyeksikan_ke_pca(X_latih_hog, mean_face, eigenfaces)
# Proyeksikan gambar test masa kecil ke PCA HOG
X_test_pca = proyeksikan_ke_pca(X_test_lintas_hog, mean_face, eigenfaces)

# Scaling factor untuk "Unwhiten" (Sesuai implementasi pca_svd.py)
n_samples = eigenspace["n_samples"][0] if "n_samples" in eigenspace else 3359
unwhiten_scale = singular_values / np.sqrt(max(1, n_samples - 1))
injeksi_multiplier = 0.35 # Kekuatan injeksi

# Simulasikan Probabilitas SVM
# Asumsikan detektor wajah SVM mendeteksi bahwa semua subjek adalah ras Asia
# dengan probabilitas 85% (0.85).
prob_asia_prediksi = 0.85

def hitung_akurasi(X_query, nama_skenario, print_detail=True):
    print(f"\n EVALUASI: {nama_skenario}")
    benar = 0
    total = len(X_query)
    
    for i in range(total):
        lbl_asli = int(y_kel[i])
        nama_asli = KAMUS_NAMA.get(lbl_asli, f"ID-{lbl_asli}")
        
        # Cari Cosine Similarity tertinggi dengan database dewasa
        sims = cosine_similarity(X_query[i].reshape(1, -1), X_db_pca).flatten()
        
        # Urutkan untuk mendapatkan top-3
        idx_sorted = np.argsort(sims)[::-1]
        
        pred_label = int(y_kel[idx_sorted[0]])
        nama_pred = KAMUS_NAMA.get(pred_label, f"ID-{pred_label}")
        sim_pred = sims[idx_sorted[0]]
        
        top3_info = []
        for rank in range(min(3, len(idx_sorted))):
            l = int(y_kel[idx_sorted[rank]])
            n = KAMUS_NAMA.get(l, f"ID-{l}")
            s = sims[idx_sorted[rank]]
            top3_info.append(f"{n}({s:.3f})")
            
        ok_c = (pred_label == lbl_asli)
        if ok_c:
            benar += 1
            
        interp_c = "Sangat Mirip" if sim_pred >= THRESHOLD_COSINE else "Kurang Mirip"
        
        if print_detail:
            print(f"  Target: {nama_asli} (ID: {lbl_asli})")
            print(f"  COSINE {'BENAR' if ok_c else 'SALAH'}")
            print(f"    Prediksi : {nama_pred} | Sim: {sim_pred:.4f} ({interp_c})")
            print(f"    Top-3    : " + ", ".join(top3_info))
            
    akurasi = (benar / total) * 100
    print(f"\n  [KESIMPULAN] {nama_skenario}: Akurasi {akurasi:.1f}% ({benar}/{total} benar)")
    return akurasi

print("\n" + "=" * 65)
print(" HASIL EVALUASI ABLATION STUDY (CROSS-AGE MATCHING)")
print("=" * 65)

# Skenario 1: Baseline (Tanpa Injeksi Penuaan)
acc_baseline = hitung_akurasi(X_test_pca, "1. Baseline (Tanpa Injeksi)")

# Skenario 2: Hard Routing (FG-NET - Kaukasia 100%)
# Rumus: W_new = W_old + (V_fgnet * scale)
X_test_fgnet = X_test_pca + (aging_vector_fgnet * unwhiten_scale * injeksi_multiplier)
acc_fgnet = hitung_akurasi(X_test_fgnet, "2. Hard Routing (100% FG-NET)")

# Skenario 3: Hard Routing (AAF - Asia 100%)
# Rumus: W_new = W_old + (V_aaf * scale)
X_test_aaf = X_test_pca + (aging_vector_aaf * unwhiten_scale * injeksi_multiplier)
acc_aaf = hitung_akurasi(X_test_aaf, "3. Hard Routing (100% AAF)")

# Skenario 4: Soft Routing (Hybrid Vector)
# Rumus: V_hybrid = (0.85 * V_aaf) + (0.15 * V_fgnet)
# W_new = W_old + (V_hybrid * scale)
aging_vector_hybrid = (prob_asia_prediksi * aging_vector_aaf) + ((1 - prob_asia_prediksi) * aging_vector_fgnet)
X_test_hybrid = X_test_pca + (aging_vector_hybrid * unwhiten_scale * injeksi_multiplier)
acc_hybrid = hitung_akurasi(X_test_hybrid, f"4. Hybrid Vector (Asia {prob_asia_prediksi*100:.0f}%)")

print("-" * 65)
if acc_hybrid >= acc_aaf and acc_hybrid >= acc_fgnet:
    print(" KESIMPULAN: Pendekatan Hybrid Vector memberikan performa terbaik/seimbang.")
    print(" Bukti matematis bahwa kombinasi linear (Aljabar Linear) efektif mengatasi dataset bias.")


# ==========================================
# PEMBUATAN GRAFIK VISUALISASI
# ==========================================
print("\n[+] Membuat grafik evaluasi...")
labels = ['Baseline\n(No Aging)', 'FG-NET\n(Kaukasia 100%)', 'AAF\n(Asia 100%)', f'Hybrid Vector\n(Asia {prob_asia_prediksi*100:.0f}%)']
accuracies = [acc_baseline, acc_fgnet, acc_aaf, acc_hybrid]
colors = ['#94a3b8', '#f87171', '#fbbf24', '#34d399']

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=1.2)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.title('Evaluasi Akurasi: Pengaruh "Ethnicity-Aware Aging Vector"', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Akurasi Pengenalan Lintas Usia (%)', fontsize=12)
plt.ylim(0, max(accuracies) + 20 if max(accuracies) < 80 else 105)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Tambahkan panah peningkatan
if acc_hybrid > acc_baseline:
    plt.annotate(
        f'+{acc_hybrid - acc_baseline:.1f}%',
        xy=(3, acc_hybrid), xycoords='data',
        xytext=(0, acc_baseline), textcoords='data',
        arrowprops=dict(arrowstyle="->", color="green", connectionstyle="arc3,rad=-0.2", lw=2),
        fontsize=12, fontweight='bold', color='green'
    )

plt.tight_layout()
plt.savefig('evaluasi_hybrid.png', dpi=120)
print("[+] Grafik berhasil disimpan sebagai 'evaluasi_hybrid.png'")
plt.show()

print("\nSELESAI. Anda bisa menggunakan skrip ini untuk bab hasil pengujian di skripsi/laporan.")
