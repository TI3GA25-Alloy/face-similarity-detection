import os
import sys
import cv2
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PATH_SAMPEL = "dataset_kelompok"
PATH_SELFIE = "Selfie & ID Data - Sample"
LABEL_MULAI = 40
TARGET_SIZE = (100, 100)
NAMA_OUTPUT = "privasi_kelompok_100x100.npz"

NAMA_DEWASA = ["dewasa", "adult", "baru", "new", "dewasa_bawaan"]
NAMA_KECIL = ["kecil", "child", "childhood", "small", "lama", "old"]
VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")


def cari_file_fleksibel(folder: str, kandidat_nama: list) -> str | None:
    semua_file = os.listdir(folder)
    for nama in kandidat_nama:
        for f in semua_file:
            nama_tanpa_ext, ext = os.path.splitext(f)
            if nama_tanpa_ext.lower() == nama.lower() and ext.lower() in VALID_EXT:
                return os.path.join(folder, f)
    return None


from streamlit_app.core.face_utils import align_face_lbf

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)


def detect_and_crop_face(gray_img: np.ndarray) -> np.ndarray:
    """
    Mendeteksi wajah dari gambar.
    Jika terdeteksi, kembalikan crop area wajah dengan sedikit padding.
    Jika tidak terdeteksi, kembalikan gambar asli sebagai fallback.
    """
    faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) > 0:
        x, y, w, h = faces[0]

        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(gray_img.shape[1], x + w + pad_x)
        y2 = min(gray_img.shape[0], y + h + pad_y)
        return gray_img[y1:y2, x1:x2], True
    return gray_img, False


def baca_gambar_grayscale_dan_crop(img_path: str) -> tuple[np.ndarray | None, bool]:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        try:
            from PIL import Image
            pil_img = Image.open(img_path).convert("L")
            img = np.array(pil_img)
        except Exception:
            pass
    wajah_terdeteksi = False
    if img is not None:
        img, wajah_terdeteksi = detect_and_crop_face(img)
    return img, wajah_terdeteksi


def preprocess(img: np.ndarray, target_size=TARGET_SIZE) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    img = clahe.apply(img)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0).flatten()


def preprocess_kecil(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    img = np.power(img.astype(np.float32) / 255.0, 1.0 / 1.2)
    return img.astype(np.float32).flatten()


def preprocess_old(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    img = np.power(img.astype(np.float32) / 255.0, 1.0 / 1.2)
    return img.astype(np.float32).flatten()


def load_selfie_images():
    old_photos = []
    new_photos = []
    old_labels = []
    new_labels = []
    person_id = 0

    for entry in sorted(os.listdir(PATH_SELFIE)):
        person_dir = os.path.join(PATH_SELFIE, entry)
        if not os.path.isdir(person_dir):
            continue

        # Docs = old / childhood photos
        doc_dir = os.path.join(person_dir, "docs")
        if os.path.isdir(doc_dir):
            for fname in sorted(os.listdir(doc_dir)):
                if not fname.lower().endswith(VALID_EXT):
                    continue
                img = cv2.imread(os.path.join(doc_dir, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                face, _ = detect_and_crop_face(img)
                if face is None or face.shape[0] < 30 or face.shape[1] < 30:
                    face = img
                old_photos.append(preprocess(face))
                old_labels.append(person_id)

        # Selfies = new / adult photos
        selfie_dir = os.path.join(person_dir, "selfies")
        if os.path.isdir(selfie_dir):
            for fname in sorted(os.listdir(selfie_dir)):
                if not fname.lower().endswith(VALID_EXT):
                    continue
                img = cv2.imread(os.path.join(selfie_dir, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                face, _ = detect_and_crop_face(img)
                if face is None or face.shape[0] < 30 or face.shape[1] < 30:
                    face = img
                new_photos.append(preprocess(face))
                new_labels.append(person_id)

        person_id += 1

    return np.array(old_photos), np.array(new_photos), np.array(old_labels), np.array(new_labels), person_id


def main():
    print("  ENKRIPSI DATASET KELOMPOK ke .NPZ (Selfie & ID Pipeline)")
    print()

    if not os.path.exists(PATH_SELFIE):
        print(f"  ERROR: Folder '{PATH_SELFIE}' tidak ditemukan!")
        sys.exit(1)

    print(f"  Loading Selfie & ID Data - Sample...")
    old_imgs, new_imgs, old_labels, new_labels, n_selfie_people = load_selfie_images()
    print(f"    Old photos (docs/): {len(old_imgs)} dari {len(np.unique(old_labels))} orang")
    print(f"    New photos (selfies/): {len(new_imgs)} dari {len(np.unique(new_labels))} orang")

    # --- X_latih (gallery): 1 selfie per orang (pilih yg pertama) ---
    X_latih_list = []
    y_list = []
    nama_gallery = []

    for pid in range(n_selfie_people):
        mask = new_labels == pid
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        X_latih_list.append(new_imgs[idx[0]])
        y_list.append(pid)
        nama_gallery.append(f"selfie_{pid}")

    # --- X_test_sama (same-age): selfie ke-2 per orang (kalau ada) ---
    X_test_sama_list = []
    y_sama_list = []

    for pid in range(n_selfie_people):
        mask = new_labels == pid
        idx = np.where(mask)[0]
        if len(idx) >= 2:
            X_test_sama_list.append(new_imgs[idx[1]])
            y_sama_list.append(pid)

    # --- X_test_lintas (cross-age): doc pertama per orang (kalau ada) ---
    X_test_lintas_list = []
    y_lintas_list = []

    for pid in range(n_selfie_people):
        mask = old_labels == pid
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        X_test_lintas_list.append(old_imgs[idx[0]])
        y_lintas_list.append(pid)

    # --- Tambah 18 anggota kelompok ke gallery + cross-age test ---
    n_group_added = 0
    if not os.path.exists(PATH_SAMPEL):
        print(f"  Warning: '{PATH_SAMPEL}' tidak ditemukan, skip group members")
    else:
        for idx, nama_folder in enumerate(sorted([
            f for f in os.listdir(PATH_SAMPEL)
            if os.path.isdir(os.path.join(PATH_SAMPEL, f)) and not f.startswith(".")
        ])):
            path_orang = os.path.join(PATH_SAMPEL, nama_folder)
            path_dewasa = cari_file_fleksibel(path_orang, NAMA_DEWASA)
            path_kecil = cari_file_fleksibel(path_orang, NAMA_KECIL)

            if not path_dewasa or not path_kecil:
                continue

            img_dewasa, ok_d = baca_gambar_grayscale_dan_crop(path_dewasa)
            img_kecil, ok_k = baca_gambar_grayscale_dan_crop(path_kecil)
            if img_dewasa is None or img_kecil is None:
                continue

            pid = LABEL_MULAI + idx
            # Dewasa → gallery
            X_latih_list.append(preprocess(img_dewasa))
            y_list.append(pid)
            nama_gallery.append(nama_folder)
            # Kecil → cross-age test (label sama)
            X_test_lintas_list.append(preprocess(img_kecil))
            n_group_added += 1

    # --- Convert ke numpy ---
    X_latih = np.array(X_latih_list, dtype=np.float32)
    X_test_sama = np.array(X_test_sama_list, dtype=np.float32) if X_test_sama_list else np.empty((0, 10000), dtype=np.float32)
    X_test_lintas = np.array(X_test_lintas_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    n_total_people = n_selfie_people + n_group_added

    np.savez(
        NAMA_OUTPUT,
        X_latih=X_latih,
        X_test_sama=X_test_sama,
        X_test_lintas=X_test_lintas,
        y=y,
        nama_anggota=np.array(nama_gallery),
    )

    print()
    print("  BERHASIL! File tersimpan.")
    print(f"  Output file      : {NAMA_OUTPUT}")
    print()
    print(f"  Shapes:")
    print(f"    X_latih (gallery)          : {X_latih.shape}")
    print(f"    X_test_sama (same-age)     : {X_test_sama.shape}")
    print(f"    X_test_lintas (cross-age)  : {X_test_lintas.shape}")
    print(f"    y (labels)                 : {y.shape}")
    print()
    print(f"  Komposisi:")
    print(f"    Gallery: {n_selfie_people} selfies + {n_group_added} group dewasa = {len(X_latih)}")
    print(f"    Cross-age test: {len(X_test_lintas)} (Selfie & ID docs + group kecil)")
    print(f"    Random baseline: 1/{n_total_people} = {1/n_total_people*100:.2f}%")


if __name__ == "__main__":
    main()
