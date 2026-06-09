import sys
import warnings
import os
import random
import tarfile
import urllib.request
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")


KAMUS_NAMA_OVERRIDE = {}  
KAMUS_NAMA = {} 
NAMA_NPZ = "privasi_kelompok_100x100.npz"

N_KOMPONEN_PCA = 100

THRESHOLD_EUCLIDEAN = 15.0
THRESHOLD_COSINE = 0.70

print("  Konfigurasi sistem (v2 - improved accuracy)")
print(f"   PCA komponen    : {N_KOMPONEN_PCA}")
print(f"   Threshold Eucl  : <= {THRESHOLD_EUCLIDEAN}")
print(f"   Threshold Cosine: >= {THRESHOLD_COSINE}")


print("\n" + "=" * 65)
print(" STEP 1: Memuat Olivetti Faces...")

olivetti = fetch_olivetti_faces()
y_olivetti = olivetti.target
X_olivetti = np.array(
    [cv2.resize(img, (100, 100)).flatten() for img in olivetti.images], dtype=np.float32
)

print(
    f"   Olivetti: {X_olivetti.shape[0]} foto dari {len(np.unique(y_olivetti))} orang"
)

print("\n STEP 1.5: Memuat LFW (Labeled Faces in the Wild)...")
print("   (Dataset selebriti untuk memperkaya ruang wajah PCA)")
try:
    lfw = fetch_lfw_people(min_faces_per_person=40, resize=None, color=False)
    offset_label = np.max(y_olivetti) + 1
    y_lfw = lfw.target + offset_label

    X_lfw = np.array(
        [cv2.resize(img, (100, 100)).flatten() for img in lfw.images], dtype=np.float32
    )
    print(f"   LFW     : {X_lfw.shape[0]} foto dari {len(np.unique(y_lfw))} orang")
except Exception as e:
    print(f"    Gagal memuat LFW: {e}. Melanjutkan tanpa LFW.")
    X_lfw = np.empty((0, 10000), dtype=np.float32)
    y_lfw = np.array([], dtype=np.int32)


print("\n STEP 1.7: Memuat FG-NET Aging Database...")
print("   (Dataset Lintas Usia asli - mendownload dari internet...)")
import os
import urllib.request
import zipfile

X_fgnet = []
y_fgnet = []
age_fgnet = []

try:
    if not os.path.exists("FGNET.zip"):
        print("   ⏳ Mendownload FGNET.zip (~20MB)...")
        urllib.request.urlretrieve(
            "http://yanweifu.github.io/FG_NET_data/FGNET.zip", "FGNET.zip"
        )

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    print("   ⏳ Memproses foto FG-NET langsung dari .zip (In-Memory)...")
    offset_fgnet = np.max(y_lfw) + 1 if len(y_lfw) > 0 else np.max(y_olivetti) + 1

    if not os.path.exists("lbfmodel.yaml"):
        print("   ⏳ Downloading lbfmodel.yaml untuk alignment...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml", "lbfmodel.yaml")
    
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("lbfmodel.yaml")

    with zipfile.ZipFile("FGNET.zip", "r") as archive:
        image_files = [f for f in archive.namelist() if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for f in image_files:
            try:
                basename = os.path.basename(f)
                sub_id = int(basename[:3]) + offset_fgnet
                
                age_str = ''.join([c for c in basename[3:7] if c.isdigit()])
                age = int(age_str) if age_str else -1
                
                file_data = archive.read(f)
                img_array = np.frombuffer(file_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                faces = face_cascade.detectMultiScale(img, 1.1, 5, minSize=(30, 30))
                if len(faces) > 0:
                    bbox = faces[0]
                    
                    # Langsung fit (model sudah diload di luar loop)
                    ok, landmarks = facemark.fit(img, np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]]))
                    
                    if ok and len(landmarks) > 0:
                        pts = landmarks[0][0]
                        left_eye = np.mean(pts[36:42], axis=0)
                        right_eye = np.mean(pts[42:48], axis=0)
                        dy = right_eye[1] - left_eye[1]
                        dx = right_eye[0] - left_eye[0]
                        angle = np.degrees(np.arctan2(dy, dx))
                        dist = np.sqrt(dx**2 + dy**2)
                        
                        target_size = (100, 100)
                        desired_dist = target_size[0] * 0.40
                        scale = desired_dist / max(dist, 1.0)
                        
                        eye_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
                        M = cv2.getRotationMatrix2D(eye_center, angle, scale)
                        
                        t_x = target_size[0] * 0.50
                        t_y = target_size[1] * 0.35
                        M[0, 2] += (t_x - eye_center[0])
                        M[1, 2] += (t_y - eye_center[1])
                        
                        img_resized = cv2.warpAffine(img, M, target_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    else:
                        x, y, w, h = bbox
                        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
                        x1 = max(0, x - pad_x)
                        y1 = max(0, y - pad_y)
                        x2 = min(img.shape[1], x + w + pad_x)
                        y2 = min(img.shape[0], y + h + pad_y)
                        img = img[y1:y2, x1:x2]
                        img_resized = cv2.resize(img, (100, 100))
                else:
                    img_resized = cv2.resize(img, (100, 100))
                X_fgnet.append((img_resized.astype(np.float32) / 255.0).flatten())
                y_fgnet.append(sub_id)
                age_fgnet.append(age)
            except:
                pass

    X_fgnet = np.array(X_fgnet, dtype=np.float32)
    y_fgnet = np.array(y_fgnet, dtype=np.int32)
    print(f"    FG-NET : {X_fgnet.shape[0]} foto dari {len(np.unique(y_fgnet))} orang")
except Exception as e:
    print(f"    Gagal memuat FG-NET: {e}")
    X_fgnet = np.empty((0, 10000), dtype=np.float32)
    y_fgnet = np.array([], dtype=np.int32)


print("\n STEP 1.8: Memuat AAF (All-Age-Faces) Database...")
print("   (Dataset Lintas Usia Asia - mendownload dari internet jika belum ada...)")

X_aaf = []
y_aaf = []
age_aaf = []

try:
    if not os.path.exists("AAF.zip"):
        print("   ⏳ PENTING: Mendownload AAF.zip (~1.25 GB)... Proses ini mungkin memakan waktu lama!")
        # Dropbox direct link
        url_aaf = "https://www.dropbox.com/s/a0lj1ddd54ns8qy/All-Age-Faces%20Dataset.zip?dl=1"
        urllib.request.urlretrieve(url_aaf, "AAF.zip")

    print("   ⏳ Memproses foto AAF langsung dari .zip (In-Memory)...")
    offset_aaf = np.max(y_fgnet) + 1 if len(y_fgnet) > 0 else (np.max(y_lfw) + 1 if len(y_lfw) > 0 else np.max(y_olivetti) + 1)
    
    with zipfile.ZipFile("AAF.zip", "r") as archive:
        image_files = [f for f in archive.namelist() if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        # Batasi ke 2000 foto saja agar proses training tidak kehabisan RAM/terlalu lama (opsional)
        if len(image_files) > 2000:
            print(f"      Ditemukan {len(image_files)} gambar. Menggunakan 2000 sampel acak untuk training.")
            random.seed(42)
            image_files = random.sample(image_files, 2000)
            
        for f in image_files:
            try:
                basename = os.path.basename(f)
                # Format: 13322A80.jpg -> ID: 13322, Umur: 80
                parts = basename.split('A')
                if len(parts) == 2:
                    sub_id = int(parts[0]) + offset_aaf
                    age_str = parts[1].split('.')[0]
                    age = int(age_str)
                else:
                    continue
                
                file_data = archive.read(f)
                img_array = np.frombuffer(file_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Untuk AAF, kita skip landmark detection berat agar cepat, cukup resize
                # Karena AAF sudah memiliki versi 'aligned', asumsikan gambar cukup terpusat.
                img_resized = cv2.resize(img, (100, 100))
                X_aaf.append((img_resized.astype(np.float32) / 255.0).flatten())
                y_aaf.append(sub_id)
                age_aaf.append(age)
            except:
                pass

    X_aaf = np.array(X_aaf, dtype=np.float32)
    y_aaf = np.array(y_aaf, dtype=np.int32)
    print(f"    AAF    : {X_aaf.shape[0]} foto dari {len(np.unique(y_aaf))} orang")
except Exception as e:
    print(f"    Gagal memuat AAF: {e}")
    X_aaf = np.empty((0, 10000), dtype=np.float32)
    y_aaf = np.array([], dtype=np.int32)


print("\n" + "=" * 65)
print(" STEP 2: Memuat dataset kelompok dari .npz...")

try:
    data = np.load(NAMA_NPZ)
    X_latih_kel = data["X_latih"]
    X_test_sama = data["X_test_sama"]
    X_test_lintas = data["X_test_lintas"]
    y_kel = data["y"]
    n_anggota = len(y_kel)

    if "nama_anggota" in data:
        nama_list = [str(n) for n in data["nama_anggota"]]
        for lbl, nama in zip(y_kel, nama_list):
            KAMUS_NAMA[int(lbl)] = nama
    else:
        for i, lbl in enumerate(y_kel):
            KAMUS_NAMA[int(lbl)] = f"Anggota {i + 1}"
    KAMUS_NAMA.update(KAMUS_NAMA_OVERRIDE)

    print(f"   {n_anggota} anggota | Nama: {list(KAMUS_NAMA.values())}")
    print(f"   X_latih shape       : {X_latih_kel.shape}")
    print(f"   X_test_sama shape   : {X_test_sama.shape}")
    print(f"   X_test_lintas shape : {X_test_lintas.shape}")
    MODE_SIMULASI = False

except FileNotFoundError:
    print(f"     File '{NAMA_NPZ}' tidak ditemukan! MODE SIMULASI aktif.")
    n_anggota = 5
    X_latih_kel = np.random.rand(n_anggota, 10000).astype(np.float32)
    y_kel = np.array(list(range(40, 40 + n_anggota)), dtype=np.int32)
    X_test_sama = np.random.rand(n_anggota, 10000).astype(np.float32)
    X_test_lintas = np.random.rand(n_anggota, 10000).astype(np.float32)
    for i, lbl in enumerate(y_kel):
        KAMUS_NAMA[int(lbl)] = f"Anggota {i + 1}"
    MODE_SIMULASI = True


print("\n" + "=" * 65)
print(" STEP 3: Augmentasi data training kelompok...")


def augment_5x(vec: np.ndarray) -> list:
    """Buat 5 variasi dari 1 vektor foto (100x100 flatten)."""
    img = vec.reshape(100, 100)
    return [
        vec.copy(),  # asli
        np.fliplr(img).flatten(),  # flip H
        np.power(img, 1.0 / 1.15).astype(np.float32).flatten(),  # lebih terang
        np.power(img, 1.15).astype(np.float32).flatten(),  # lebih gelap
        np.clip(img + np.random.normal(0, 0.02, img.shape), 0, 1)
        .astype(np.float32)
        .flatten(),  # noise ringan
    ]


X_kel_aug, y_kel_aug = [], []
for i, (vec, lbl) in enumerate(zip(X_latih_kel, y_kel)):
    for v in augment_5x(vec):
        X_kel_aug.append(v)
        y_kel_aug.append(int(lbl))

X_kel_aug = np.array(X_kel_aug, dtype=np.float32)
y_kel_aug = np.array(y_kel_aug, dtype=np.int32)

print(f"   Kelompok sebelum aug : {X_latih_kel.shape[0]} foto")
print(f"   Kelompok setelah aug : {X_kel_aug.shape[0]} foto (5x lipat)")


print("\n" + "=" * 65)
print(f" STEP 4: Gabungkan data & latih PCA ({N_KOMPONEN_PCA} komponen)...")

X_train_total = np.vstack([X_olivetti, X_lfw, X_fgnet, X_aaf, X_kel_aug])
y_train_total = np.concatenate([y_olivetti, y_lfw, y_fgnet, y_aaf, y_kel_aug])

print(f"\n   Komposisi X_train_total:")
print(f"   • Olivetti      : {X_olivetti.shape[0]} foto")
print(f"   • LFW           : {X_lfw.shape[0]} foto")
print(f"   • FG-NET        : {X_fgnet.shape[0]} foto")
print(f"   • AAF           : {X_aaf.shape[0]} foto")
print(f"   • Kelompok (aug): {X_kel_aug.shape[0]} foto ({n_anggota} orang × 5)")
print(
    f"   • TOTAL         : {X_train_total.shape[0]} foto × {X_train_total.shape[1]} fitur"
)
print(f"   • Orang unik    : {len(np.unique(y_train_total))}")

from skimage.feature import hog, local_binary_pattern

print(
    f"\n   ⏳ Mengekstrak fitur LBP (Tekstur) dan HOG (Bentuk) untuk {X_train_total.shape[0]} foto..."
)
X_train_lbp = []
X_train_hog = []

for vec in X_train_total:
    img_2d = vec.reshape(100, 100)
    img_uint8 = (np.clip(img_2d, 0, 1) * 255).astype(np.uint8)

    lbp = local_binary_pattern(img_uint8, P=8, R=1, method="uniform")
    X_train_lbp.append(lbp.flatten().astype(np.float32) / (lbp.max() + 1e-8))

    h = hog(
        img_uint8,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
    )
    X_train_hog.append(h.astype(np.float32))

X_train_lbp = np.array(X_train_lbp)
X_train_hog = np.array(X_train_hog)

print("   ⏳ Melatih 3 Model PCA secara bersamaan (Pixel, LBP, HOG)...")
pca = PCA(n_components=N_KOMPONEN_PCA, whiten=True, random_state=42)
pca.fit(X_train_total)

pca_lbp = PCA(n_components=N_KOMPONEN_PCA, whiten=True, random_state=42)
pca_lbp.fit(X_train_lbp)

pca_hog = PCA(n_components=N_KOMPONEN_PCA, whiten=True, random_state=42)
pca_hog.fit(X_train_hog)

variance_total = np.sum(pca.explained_variance_ratio_) * 100
print(f"\n   ✅ PCA selesai! Variance explained (Pixel): {variance_total:.1f}%")
print(f"   Dimensi: 10.000 → {N_KOMPONEN_PCA}")

print("   ⏳ Mengekstrak Vektor Injeksi Penuaan (FG-NET)...")
start_fgnet = len(X_olivetti) + len(X_lfw)
end_fgnet = start_fgnet + len(X_fgnet)

# Proyeksikan FG-NET ke ruang PCA
X_fgnet_pca = pca.transform(X_train_total[start_fgnet:end_fgnet])
X_fgnet_lbp_pca = pca_lbp.transform(X_train_lbp[start_fgnet:end_fgnet])
X_fgnet_hog_pca = pca_hog.transform(X_train_hog[start_fgnet:end_fgnet])

age_arr = np.array(age_fgnet)
child_mask = (age_arr > 0) & (age_arr <= 12)
adult_mask = (age_arr >= 18)

if np.any(child_mask) and np.any(adult_mask):
    W_child_pix = np.mean(X_fgnet_pca[child_mask], axis=0)
    W_adult_pix = np.mean(X_fgnet_pca[adult_mask], axis=0)
    aging_vector_fgnet_pix = W_adult_pix - W_child_pix

    W_child_lbp = np.mean(X_fgnet_lbp_pca[child_mask], axis=0)
    W_adult_lbp = np.mean(X_fgnet_lbp_pca[adult_mask], axis=0)
    aging_vector_fgnet_lbp = W_adult_lbp - W_child_lbp

    W_child_hog = np.mean(X_fgnet_hog_pca[child_mask], axis=0)
    W_adult_hog = np.mean(X_fgnet_hog_pca[adult_mask], axis=0)
    aging_vector_fgnet_hog = W_adult_hog - W_child_hog
else:
    aging_vector_fgnet_pix = np.zeros(N_KOMPONEN_PCA, dtype=np.float32)
    aging_vector_fgnet_lbp = np.zeros(N_KOMPONEN_PCA, dtype=np.float32)
    aging_vector_fgnet_hog = np.zeros(N_KOMPONEN_PCA, dtype=np.float32)


print("   ⏳ Mengekstrak Vektor Injeksi Penuaan (AAF)...")
start_aaf = end_fgnet
end_aaf = start_aaf + len(X_aaf)

# Proyeksikan AAF ke ruang PCA
if len(X_aaf) > 0:
    X_aaf_pca = pca.transform(X_train_total[start_aaf:end_aaf])
    X_aaf_lbp_pca = pca_lbp.transform(X_train_lbp[start_aaf:end_aaf])
    X_aaf_hog_pca = pca_hog.transform(X_train_hog[start_aaf:end_aaf])

    age_arr_aaf = np.array(age_aaf)
    child_mask_aaf = (age_arr_aaf > 0) & (age_arr_aaf <= 12)
    adult_mask_aaf = (age_arr_aaf >= 18)

    if np.any(child_mask_aaf) and np.any(adult_mask_aaf):
        W_child_aaf_pix = np.mean(X_aaf_pca[child_mask_aaf], axis=0)
        W_adult_aaf_pix = np.mean(X_aaf_pca[adult_mask_aaf], axis=0)
        aging_vector_aaf_pix = W_adult_aaf_pix - W_child_aaf_pix

        W_child_aaf_lbp = np.mean(X_aaf_lbp_pca[child_mask_aaf], axis=0)
        W_adult_aaf_lbp = np.mean(X_aaf_lbp_pca[adult_mask_aaf], axis=0)
        aging_vector_aaf_lbp = W_adult_aaf_lbp - W_child_aaf_lbp

        W_child_aaf_hog = np.mean(X_aaf_hog_pca[child_mask_aaf], axis=0)
        W_adult_aaf_hog = np.mean(X_aaf_hog_pca[adult_mask_aaf], axis=0)
        aging_vector_aaf_hog = W_adult_aaf_hog - W_child_aaf_hog
    else:
        aging_vector_aaf_pix = np.zeros(N_KOMPONEN_PCA, dtype=np.float32)
        aging_vector_aaf_lbp = np.zeros(N_KOMPONEN_PCA, dtype=np.float32)
        aging_vector_aaf_hog = np.zeros(N_KOMPONEN_PCA, dtype=np.float32)
else:
    aging_vector_aaf_pix = np.zeros(N_KOMPONEN_PCA, dtype=np.float32)
    aging_vector_aaf_lbp = np.zeros(N_KOMPONEN_PCA, dtype=np.float32)
    aging_vector_aaf_hog = np.zeros(N_KOMPONEN_PCA, dtype=np.float32)

print(f"    Menyimpan model otak (pretrained_eigenspace.npz) DENGAN LBP, HOG, & AGING VECTOR...")
np.savez_compressed(
    "pretrained_eigenspace.npz",
    mean_face=pca.mean_,
    eigenfaces=pca.components_,
    singular_values=pca.singular_values_,
    explained_variance_pct=pca.explained_variance_ratio_ * 100,
    mean_lbp=pca_lbp.mean_,
    eigenfaces_lbp=pca_lbp.components_,
    singular_values_lbp=pca_lbp.singular_values_,
    mean_hog=pca_hog.mean_,
    eigenfaces_hog=pca_hog.components_,
    singular_values_hog=pca_hog.singular_values_,
    aging_vector_pix=aging_vector_fgnet_pix,
    aging_vector_lbp=aging_vector_fgnet_lbp,
    aging_vector_hog=aging_vector_fgnet_hog,
    aging_vector_fgnet_pix=aging_vector_fgnet_pix,
    aging_vector_fgnet_lbp=aging_vector_fgnet_lbp,
    aging_vector_fgnet_hog=aging_vector_fgnet_hog,
    aging_vector_aaf_pix=aging_vector_aaf_pix,
    aging_vector_aaf_lbp=aging_vector_aaf_lbp,
    aging_vector_aaf_hog=aging_vector_aaf_hog,
    n_samples=np.array([X_train_total.shape[0]]),
    k_components=np.array([pca.n_components_]),
    image_shape=np.array([100, 100]),
    feature_mode="fusion",
)
print(f"    ✅ Model otak FUSION berhasil disimpan! Bisa didownload untuk Backend.")

X_train_pca = pca.transform(X_train_total)
X_test_sama_pca = pca.transform(X_test_sama)
X_test_lintas_pca = pca.transform(X_test_lintas)

X_kel_aug_pca = pca.transform(X_kel_aug)


def label_ke_nama(label: int) -> str:
    return KAMUS_NAMA.get(label, f"Olivetti-{label:02d}")


def hitung_kemiripan_per_identitas(vec_query, X_db, y_db):
    # Compare against the mean/closest of each unique IDENTITY
    # rather than each sample to avoid bias towards identities with more photos.
    identitas_unik = np.unique(y_db)

    dist_per_id = {}
    sim_per_id = {}

    for uid in identitas_unik:
        mask = y_db == uid
        X_id = X_db[mask]
        dists = np.linalg.norm(X_id - vec_query, axis=1)
        dist_per_id[uid] = np.min(dists)  # ambil minimum (nearest sample)
        sims = cosine_similarity(X_id, vec_query.reshape(1, -1)).flatten()
        sim_per_id[uid] = np.max(sims)  # ambil maksimum

    sorted_e = sorted(dist_per_id.items(), key=lambda x: x[1])
    top3_e = [(uid, d) for uid, d in sorted_e[:3]]
    pred_e = sorted_e[0][0]
    score_e = sorted_e[0][1]

    sorted_c = sorted(sim_per_id.items(), key=lambda x: x[1], reverse=True)
    top3_c = [(uid, s) for uid, s in sorted_c[:3]]
    pred_c = sorted_c[0][0]
    score_c = sorted_c[0][1]

    return {
        "euclidean": {"label_pred": pred_e, "score": score_e, "top3": top3_e},
        "cosine": {"label_pred": pred_c, "score": score_c, "top3": top3_c},
    }


def evaluasi_skenario(X_uji_pca, nama_skenario, deskripsi):
    print()
    print(f" EVALUASI: {nama_skenario}")
    print(f"   {deskripsi}")

    benar_e = benar_c = 0
    total = len(X_uji_pca)
    hasil_detail = []

    for i in range(total):
        lbl_asli = int(y_kel[i])
        nama_asli = label_ke_nama(lbl_asli)

        hasil = hitung_kemiripan_per_identitas(X_uji_pca[i], X_kel_aug_pca, y_kel_aug)

        pred_e = hasil["euclidean"]["label_pred"]
        dist_e = hasil["euclidean"]["score"]
        top3_e = hasil["euclidean"]["top3"]

        pred_c = hasil["cosine"]["label_pred"]
        sim_c = hasil["cosine"]["score"]
        top3_c = hasil["cosine"]["top3"]

        ok_e = pred_e == lbl_asli
        ok_c = pred_c == lbl_asli
        if ok_e:
            benar_e += 1
        if ok_c:
            benar_c += 1

        interp_e = "Sangat Mirip" if dist_e <= THRESHOLD_EUCLIDEAN else "Kurang Mirip"
        interp_c = "Sangat Mirip" if sim_c >= THRESHOLD_COSINE else "Kurang Mirip"

        print(f"\n  Target: {nama_asli} (ID: {lbl_asli})")
        print(f"  EUCLIDEAN {'BENAR' if ok_e else 'SALAH'}")
        print(
            f"    Prediksi : {label_ke_nama(pred_e)} | Jarak: {dist_e:.4f} ({interp_e})"
        )
        print(
            f"    Top-3    : "
            + ", ".join([f"{label_ke_nama(l)}({d:.2f})" for l, d in top3_e])
        )
        print(f"  COSINE {'BENAR' if ok_c else 'SALAH'}")
        print(f"    Prediksi : {label_ke_nama(pred_c)} | Sim: {sim_c:.4f} ({interp_c})")
        print(
            f"    Top-3    : "
            + ", ".join([f"{label_ke_nama(l)}({s:.3f})" for l, s in top3_c])
        )

        hasil_detail.append(
            {
                "label_asli": lbl_asli,
                "nama_asli": nama_asli,
                "pred_e": pred_e,
                "dist_e": dist_e,
                "ok_e": ok_e,
                "pred_c": pred_c,
                "sim_c": sim_c,
                "ok_c": ok_c,
            }
        )

    akurasi_e = (benar_e / total) * 100
    akurasi_c = (benar_c / total) * 100

    print()
    print(f" KESIMPULAN [{nama_skenario}]:")
    print(f"   Benar/Total Euclidean : {benar_e}/{total}")
    print(f"   Benar/Total Cosine    : {benar_c}/{total}")
    print(f"   Akurasi Euclidean (L2): {akurasi_e:.1f}%")
    print(f"   Akurasi Cosine        : {akurasi_c:.1f}%")

    if akurasi_e > akurasi_c:
        print(f"   Euclidean unggul (+{akurasi_e - akurasi_c:.1f}%)")
    elif akurasi_c > akurasi_e:
        print(f"   Cosine unggul (+{akurasi_c - akurasi_e:.1f}%)")
    else:
        print(f"   Kedua metrik setara!")

    return {
        "nama": nama_skenario,
        "akurasi_e": akurasi_e,
        "akurasi_c": akurasi_c,
        "detail": hasil_detail,
    }


print("\n" + "=" * 65)
print(" STEP 5: Evaluasi Skenario A & B...")

hasil_A = evaluasi_skenario(
    X_test_sama_pca,
    "SKENARIO A - USIA SAMA",
    "Foto dewasa (augmentasi) vs database dewasa training",
)

hasil_B = evaluasi_skenario(
    X_test_lintas_pca,
    "SKENARIO B - LINTAS USIA",
    "Foto masa kecil vs database foto dewasa",
)


print()
print(" LAPORAN PERBANDINGAN FINAL: SKENARIO A vs SKENARIO B")
print()
print(f"  {'Metrik':<30} {'Skenario A':>14} {'Skenario B':>14} {'Gap':>10}")
print(f"  {'-' * 70}")
print(
    f"  {'Akurasi Euclidean (L2)':<30} "
    f"{hasil_A['akurasi_e']:>12.1f}% "
    f"{hasil_B['akurasi_e']:>12.1f}% "
    f"{hasil_A['akurasi_e'] - hasil_B['akurasi_e']:>+9.1f}%"
)
print(
    f"  {'Akurasi Cosine Similarity':<30} "
    f"{hasil_A['akurasi_c']:>12.1f}% "
    f"{hasil_B['akurasi_c']:>12.1f}% "
    f"{hasil_A['akurasi_c'] - hasil_B['akurasi_c']:>+9.1f}%"
)

print()
print(" ANALISIS OTOMATIS:")
gap_e = hasil_A["akurasi_e"] - hasil_B["akurasi_e"]
gap_c = hasil_A["akurasi_c"] - hasil_B["akurasi_c"]

if gap_e > 0:
    print(f"   Skenario A unggul {gap_e:.1f}% atas B (Euclidean)")
    print(f"   → Perubahan usia terbukti mengurangi akurasi PCA.")
else:
    print(f"   Gap Euclidean negatif: perlu dicek kualitas foto.")

if gap_c > 0:
    print(f"   Skenario A unggul {gap_c:.1f}% atas B (Cosine)")
    print(f"   → Cosine Similarity membuktikan gap lintas usia.")
else:
    print(f"   Gap Cosine tidak signifikan.")

if MODE_SIMULASI:
    print("\n   PERHATIAN: Hasil MODE SIMULASI - data tidak nyata!")

print()


print("\n Membuat visualisasi...")

rows = 10
cols = N_KOMPONEN_PCA // rows
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
fig.suptitle(
    f"{N_KOMPONEN_PCA} Eigenfaces - Komponen Utama PCA\n"
    f"(Variance: {variance_total:.1f}%)",
    fontsize=14,
    fontweight="bold",
)
for i, ax in enumerate(axes.flat):
    if i < N_KOMPONEN_PCA:
        ef = pca.components_[i].reshape(100, 100)
        ax.imshow(ef, cmap="gray", interpolation="nearest")
        ax.set_title(f"EF-{i + 1}", fontsize=6)
    ax.axis("off")
plt.tight_layout()
plt.savefig("eigenfaces.png", dpi=100, bbox_inches="tight")
plt.show()
print("   Disimpan: eigenfaces.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Analisis Variance PCA", fontsize=13, fontweight="bold")
k_idx = np.arange(1, N_KOMPONEN_PCA + 1)
ax1.bar(
    k_idx,
    pca.explained_variance_ratio_ * 100,
    color="steelblue",
    alpha=0.8,
    edgecolor="white",
    linewidth=0.3,
)
ax1.set_xlabel("Komponen Utama")
ax1.set_ylabel("Variance (%)")
ax1.set_title("Variance Per Komponen")
cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
ax2.plot(k_idx, cum_var, "b-", linewidth=2)
ax2.fill_between(k_idx, cum_var, alpha=0.2)
ax2.axhline(
    variance_total,
    color="red",
    linestyle="--",
    label=f"{N_KOMPONEN_PCA} komponen = {variance_total:.1f}%",
)
ax2.set_xlabel("Jumlah Komponen")
ax2.set_ylabel("Kumulatif Variance (%)")
ax2.set_title("Kumulatif Variance")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca_variance.png", dpi=120, bbox_inches="tight")
plt.show()
print("   Disimpan: pca_variance.png")

fig, axes = plt.subplots(3, n_anggota, figsize=(3 * n_anggota, 9))
if n_anggota == 1:
    axes = axes.reshape(3, 1)
fig.suptitle(
    "Rekonstruksi Wajah: Dewasa Training vs PCA vs Foto Kecil",
    fontsize=12,
    fontweight="bold",
)
for i in range(n_anggota):
    nama = KAMUS_NAMA.get(int(y_kel[i]), f"Anggota {i + 1}")
    img_d = X_latih_kel[i].reshape(100, 100)
    img_r = pca.inverse_transform(pca.transform(X_latih_kel[i : i + 1])).reshape(
        100, 100
    )
    img_k = X_test_lintas[i].reshape(100, 100)
    axes[0, i].imshow(img_d, cmap="gray")
    axes[0, i].set_title(f"{nama}\nDewasa", fontsize=7)
    axes[1, i].imshow(np.clip(img_r, 0, 1), cmap="gray")
    axes[1, i].set_title("Rekon PCA", fontsize=7)
    axes[2, i].imshow(img_k, cmap="gray")
    axes[2, i].set_title("Foto Kecil", fontsize=7)
    for row in range(3):
        axes[row, i].axis("off")
plt.tight_layout()
plt.savefig("rekonstruksi.png", dpi=100, bbox_inches="tight")
plt.show()
print("   Disimpan: rekonstruksi.png")

fig, ax = plt.subplots(figsize=(10, 6))
labels_plot = ["Skenario A\n(Usia Sama)", "Skenario B\n(Lintas Usia)"]
acc_e = [hasil_A["akurasi_e"], hasil_B["akurasi_e"]]
acc_c = [hasil_A["akurasi_c"], hasil_B["akurasi_c"]]
x = np.arange(len(labels_plot))
width = 0.3
bars1 = ax.bar(
    x - width / 2, acc_e, width, label="Euclidean (L2)", color="steelblue", alpha=0.85
)
bars2 = ax.bar(
    x + width / 2, acc_c, width, label="Cosine Similarity", color="coral", alpha=0.85
)
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 1.5,
        f"{h:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )
ax.set_ylabel("Akurasi (%)", fontsize=12)
ax.set_title(
    "Perbandingan Akurasi: Skenario A vs B\nEuclidean Distance vs Cosine Similarity",
    fontsize=12,
    fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(labels_plot, fontsize=12)
ax.set_ylim(0, 115)
ax.legend(fontsize=11)
ax.axhline(50, color="green", linestyle="--", alpha=0.4)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("perbandingan_akurasi.png", dpi=120, bbox_inches="tight")
plt.show()
print("   Disimpan: perbandingan_akurasi.png")

print()
print("Semua visualisasi selesai!")
