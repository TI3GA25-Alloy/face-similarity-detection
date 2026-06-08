
import os
import sys
import cv2
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PATH_SAMPEL     = 'dataset_kelompok'         # Folder sumber foto kelompok
PATH_SELFIE_OLD = 'Selfie & ID Data - Sample'  # Dataset lama (untuk info saja)
LABEL_MULAI     = 40                         # ID pertama anggota kelompok
TARGET_SIZE     = (100, 100)                 # Ukuran output gambar
NAMA_OUTPUT     = 'privasi_kelompok_100x100.npz'

NAMA_DEWASA = ['dewasa', 'adult', 'baru', 'new', 'dewasa_bawaan']
NAMA_KECIL  = ['kecil', 'child', 'childhood', 'small', 'lama', 'old']
VALID_EXT   = ('.jpg', '.jpeg', '.png', '.webp')



def cari_file_fleksibel(folder: str, kandidat_nama: list) -> str | None:
    """
    Cari file di folder berdasarkan daftar nama kandidat.
    Case-insensitive, mendukung semua ekstensi gambar.
    """
    semua_file = os.listdir(folder)
    for nama in kandidat_nama:
        for f in semua_file:
            nama_tanpa_ext, ext = os.path.splitext(f)
            if (nama_tanpa_ext.lower() == nama.lower() and
                    ext.lower() in VALID_EXT):
                return os.path.join(folder, f)
    return None


cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_and_crop_face(gray_img: np.ndarray) -> np.ndarray:
    """
    Mendeteksi wajah dari gambar.
    Jika terdeteksi, kembalikan crop area wajah dengan sedikit padding.
    Jika tidak terdeteksi, kembalikan gambar asli sebagai fallback.
    """
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    if len(faces) > 0:
        x, y, w, h = faces[0]
        
        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(gray_img.shape[1], x + w + pad_x)
        y2 = min(gray_img.shape[0], y + h + pad_y)
        
        return gray_img[y1:y2, x1:x2]
    
    return gray_img


def baca_gambar_grayscale_dan_crop(img_path: str) -> np.ndarray | None:
    """Baca gambar, konversi ke grayscale, lalu deteksi & crop wajahnya."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        try:
            from PIL import Image
            pil_img = Image.open(img_path).convert('L')
            img = np.array(pil_img)
        except Exception:
            pass
            
    if img is not None:
        img = detect_and_crop_face(img)
        
    return img


def preprocess_dewasa(img: np.ndarray) -> np.ndarray:
    """
    Preprocessing untuk foto dewasa:
    Grayscale → CLAHE → Resize 100x100 → Normalize [0,1] → Flatten
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    img   = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0).flatten()


def preprocess_kecil(img: np.ndarray) -> np.ndarray:
    """
    Preprocessing untuk foto masa kecil (lintas usia):
    Grayscale → CLAHE → Gaussian Blur → Resize 100x100 → Gamma → Normalize → Flatten
    Gamma 1.2 = menerangkan foto lama yang cenderung gelap.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    img   = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5)
    img   = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    img   = np.power(img.astype(np.float32) / 255.0, 1.0 / 1.2)
    return img.astype(np.float32).flatten()


def augmentasi_untuk_uji(vec_dewasa: np.ndarray) -> np.ndarray:
    """
    Buat versi augmentasi foto dewasa sebagai data uji Skenario A.
    Teknik: flip horizontal (cermin) + koreksi gamma ringan.
    Ini mensimulasikan 'foto dewasa ke-2' yang tidak tersedia.
    """
    img     = vec_dewasa.reshape(TARGET_SIZE)
    flipped = np.fliplr(img)
    aug     = np.power(flipped, 1.0 / 1.1)   # sedikit lebih terang
    return np.clip(aug, 0.0, 1.0).astype(np.float32).flatten()



def main():
    print("=" * 65)
    print("  ENKRIPSI DATASET KELOMPOK ke .NPZ")
    print("  Sistem Face Comparison berbasis PCA (PRD v3)")
    print("=" * 65)
    print()

    if not os.path.exists(PATH_SAMPEL):
        print(f"  ERROR: Folder '{PATH_SAMPEL}' tidak ditemukan!")
        sys.exit(1)

    folder_list = sorted([
        f for f in os.listdir(PATH_SAMPEL)
        if os.path.isdir(os.path.join(PATH_SAMPEL, f))
        and not f.startswith('.')
    ])

    print(f"  Sumber          : {PATH_SAMPEL}/")
    print(f"  Orang ditemukan : {len(folder_list)}")
    print(f"  Urutan folder   : {folder_list}")
    print()

    wajah_dewasa_latih = []
    wajah_dewasa_uji   = []
    wajah_kecil_uji    = []
    labels             = []
    nama_anggota       = []   # simpan nama asli untuk KAMUS_NAMA di Colab

    print(f"  {'No':<4} {'Nama':<14} {'ID':<6} {'Dewasa':<20} {'Kecil':<20} Status")
    print("  " + "-" * 68)

    for idx, nama_folder in enumerate(folder_list):
        label_id   = LABEL_MULAI + idx
        path_orang = os.path.join(PATH_SAMPEL, nama_folder)

        path_dewasa = cari_file_fleksibel(path_orang, NAMA_DEWASA)
        path_kecil  = cari_file_fleksibel(path_orang, NAMA_KECIL)

        file_d = os.path.basename(path_dewasa) if path_dewasa else "-- TIDAK ADA --"
        file_k = os.path.basename(path_kecil)  if path_kecil  else "-- TIDAK ADA --"
        status = "OK" if (path_dewasa and path_kecil) else "SKIP"

        print(f"  {idx:<4} {nama_folder:<14} {label_id:<6} {file_d:<20} {file_k:<20} {status}")

        if not path_dewasa or not path_kecil:
            continue

        img_dewasa = baca_gambar_grayscale_dan_crop(path_dewasa)
        img_kecil  = baca_gambar_grayscale_dan_crop(path_kecil)

        if img_dewasa is None or img_kecil is None:
            print(f"        => GAGAL membaca gambar, lewati.")
            continue

        h_d, w_d = img_dewasa.shape[:2]
        h_k, w_k = img_kecil.shape[:2]
        if h_d < 50 or w_d < 50 or h_k < 50 or w_k < 50:
            print(f"        => Resolusi terlalu kecil, lewati.")
            continue

        vec_dewasa_latih = preprocess_dewasa(img_dewasa)
        vec_dewasa_uji   = augmentasi_untuk_uji(vec_dewasa_latih)
        vec_kecil        = preprocess_kecil(img_kecil)

        wajah_dewasa_latih.append(vec_dewasa_latih)
        wajah_dewasa_uji.append(vec_dewasa_uji)
        wajah_kecil_uji.append(vec_kecil)
        labels.append(label_id)
        nama_anggota.append(nama_folder)

    print()

    if len(labels) == 0:
        print("  ERROR: Tidak ada data berhasil diproses!")
        sys.exit(1)

    X_latih       = np.array(wajah_dewasa_latih, dtype=np.float32)
    X_test_sama   = np.array(wajah_dewasa_uji,   dtype=np.float32)
    X_test_lintas = np.array(wajah_kecil_uji,    dtype=np.float32)
    y             = np.array(labels,             dtype=np.int32)

    np.savez(
        NAMA_OUTPUT,
        X_latih       = X_latih,
        X_test_sama   = X_test_sama,
        X_test_lintas = X_test_lintas,
        y             = y,
        nama_anggota  = np.array(nama_anggota),   # tambahan: nama asli
    )

    print("=" * 65)
    print("  BERHASIL! File tersimpan.")
    print("=" * 65)
    print(f"  Output file      : {NAMA_OUTPUT}")
    print(f"  Jumlah anggota   : {len(y)} orang")
    print()
    print(f"  {'ID':<6} {'Nama'}")
    print(f"  {'-' * 25}")
    for i, (lbl, nama) in enumerate(zip(y, nama_anggota)):
        print(f"  {lbl:<6} {nama}")

    print()
    print(f"  Shapes:")
    print(f"    X_latih (training)       : {X_latih.shape}")
    print(f"    X_test_sama  (Skenario A): {X_test_sama.shape}  <- augmentasi flip")
    print(f"    X_test_lintas (Skenario B): {X_test_lintas.shape}  <- foto kecil")

    n_lama = 0
    if os.path.exists(PATH_SELFIE_OLD):
        n_lama = len([f for f in os.listdir(PATH_SELFIE_OLD)
                      if os.path.isdir(os.path.join(PATH_SELFIE_OLD, f))])
    print()
    if n_lama > 0:
        print(f"  Info: Dataset lama '{PATH_SELFIE_OLD}' tersedia ({n_lama} orang).")
        print(f"        Di Colab, aktifkan USE_DATASET_LAMA = True untuk basis training lebih kaya.")
    print()
    print("  LANGKAH SELANJUTNYA:")
    print(f"    1. Upload '{NAMA_OUTPUT}' ke Google Colab")
    print(f"    2. Jalankan colab_pca_evaluation.py")
    print("=" * 65)


if __name__ == '__main__':
    main()
