import os
import sys
import cv2
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PATH_DATASET = 'Selfie & ID Data - Sample'
VALID_EXT    = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG')

def get_semua_gambar(folder_path):
    return [
        f for f in os.listdir(folder_path)
        if f.endswith(VALID_EXT)
    ]

def main():
    print("=" * 65)
    print("🔍 DIAGNOSTIC: Inspeksi Dataset Lama")
    print(f"   Path: {PATH_DATASET}")
    print("=" * 65)

    if not os.path.exists(PATH_DATASET):
        print(f"\n❌ Folder '{PATH_DATASET}' tidak ditemukan!")
        return

    folder_list = sorted([
        f for f in os.listdir(PATH_DATASET)
        if os.path.isdir(os.path.join(PATH_DATASET, f))
    ])

    print(f"\n📁 Total folder orang: {len(folder_list)}")

    stats = {
        'total_orang': len(folder_list),
        'total_foto_docs': 0,
        'total_foto_selfies': 0,
        'orang_tanpa_docs': [],
        'orang_tanpa_selfies': [],
        'orang_docs_1_foto': 0,
        'orang_docs_multi': 0,
        'orang_selfies_1_foto': 0,
        'orang_selfies_multi': 0,
        'resolusi_terlalu_kecil': [],
    }

    print(f"\n{'No':<6} {'Folder':<20} {'docs':<10} {'selfies':<10} {'Status'}")
    print("-" * 65)

    for idx, nama_folder in enumerate(folder_list[:20]):   # tampilkan 20 pertama
        path_orang   = os.path.join(PATH_DATASET, nama_folder)
        path_docs    = os.path.join(path_orang, 'docs')
        path_selfies = os.path.join(path_orang, 'selfies')

        n_docs    = len(get_semua_gambar(path_docs))    if os.path.exists(path_docs)    else 0
        n_selfies = len(get_semua_gambar(path_selfies)) if os.path.exists(path_selfies) else 0

        stats['total_foto_docs']    += n_docs
        stats['total_foto_selfies'] += n_selfies

        if n_docs == 0:
            stats['orang_tanpa_docs'].append(nama_folder)
        elif n_docs == 1:
            stats['orang_docs_1_foto'] += 1
        else:
            stats['orang_docs_multi'] += 1

        if n_selfies == 0:
            stats['orang_tanpa_selfies'].append(nama_folder)
        elif n_selfies == 1:
            stats['orang_selfies_1_foto'] += 1
        else:
            stats['orang_selfies_multi'] += 1

        status = "" if n_docs > 0 and n_selfies > 0 else " "
        print(f"  {idx+1:<4} {nama_folder:<20} {n_docs:<10} {n_selfies:<10} {status}")

    if len(folder_list) > 20:
        print(f"  ... (dan {len(folder_list) - 20} folder lainnya)")

    for nama_folder in folder_list[20:]:
        path_orang   = os.path.join(PATH_DATASET, nama_folder)
        path_docs    = os.path.join(path_orang, 'docs')
        path_selfies = os.path.join(path_orang, 'selfies')
        n_docs    = len(get_semua_gambar(path_docs))    if os.path.exists(path_docs)    else 0
        n_selfies = len(get_semua_gambar(path_selfies)) if os.path.exists(path_selfies) else 0
        stats['total_foto_docs']    += n_docs
        stats['total_foto_selfies'] += n_selfies
        if n_docs == 0:    stats['orang_tanpa_docs'].append(nama_folder)
        elif n_docs == 1:  stats['orang_docs_1_foto'] += 1
        else:              stats['orang_docs_multi']  += 1
        if n_selfies == 0:    stats['orang_tanpa_selfies'].append(nama_folder)
        elif n_selfies == 1:  stats['orang_selfies_1_foto'] += 1
        else:                 stats['orang_selfies_multi']  += 1

    print()
    print("=" * 65)
    print(" RINGKASAN STATISTIK:")
    print("=" * 65)
    print(f"  Total orang            : {stats['total_orang']}")
    print(f"  Total foto docs (lama) : {stats['total_foto_docs']}")
    print(f"  Total foto selfies     : {stats['total_foto_selfies']}")
    print(f"  Total semua foto       : {stats['total_foto_docs'] + stats['total_foto_selfies']}")
    print()
    print(f"  Orang dengan 1 foto docs    : {stats['orang_docs_1_foto']}")
    print(f"  Orang dengan banyak docs    : {stats['orang_docs_multi']}")
    print(f"  Orang TANPA docs            : {len(stats['orang_tanpa_docs'])}")
    print()
    print(f"  Orang dengan 1 foto selfie  : {stats['orang_selfies_1_foto']}")
    print(f"  Orang dengan banyak selfies : {stats['orang_selfies_multi']}")
    print(f"  Orang TANPA selfies         : {len(stats['orang_tanpa_selfies'])}")

    n_siap = stats['total_orang'] - len(stats['orang_tanpa_docs']) - len(stats['orang_tanpa_selfies'])

    print()
    print("─" * 65)
    print(f"   Orang siap dipakai (punya docs + selfies): ~{n_siap}")
    print(f"    Orang yang perlu dicek                  : {len(stats['orang_tanpa_docs']) + len(stats['orang_tanpa_selfies'])}")

    if stats['orang_tanpa_docs']:
        print(f"\n  Folder TANPA docs: {stats['orang_tanpa_docs']}")
    if stats['orang_tanpa_selfies']:
        print(f"  Folder TANPA selfies: {stats['orang_tanpa_selfies']}")

    print()
    print("=" * 65)
    print("💡 REKOMENDASI PENGGUNAAN DI COLAB:")
    print("=" * 65)
    print(f"""
  Dataset lama ini cocok dipakai sebagai basis training TAMBAHAN
  di samping Olivetti Faces. Cara pakainya di Colab:

  1. Zip seluruh folder '{PATH_DATASET}'
  2. Upload .zip ke Google Colab
  3. Extract: !unzip dataset.zip
  4. Di colab_pca_evaluation.py, ubah:
       USE_DATASET_LAMA = True
       PATH_LAMA = '{PATH_DATASET}'

  Dengan {stats['total_orang']} orang tambahan, basis training akan jauh
  lebih kaya dan eigenspace yang terbentuk lebih representatif.

  Mapping folder lama → PRD:
    docs/     = foto lama/ID    → peran 'kecil.jpg' (masa kecil/masa lalu)
    selfies/  = foto baru/selfie → peran 'dewasa_bawaan.jpg'
""")


if __name__ == '__main__':
    main()
