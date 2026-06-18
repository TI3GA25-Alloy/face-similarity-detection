from typing import Tuple

import cv2
import numpy as np


def extract_lbp_features(image: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    if image.dtype != np.uint8:
        img_uint8 = (
            (image * 255).astype(np.uint8)
            if image.max() <= 1.0
            else image.astype(np.uint8)
        )
    else:
        img_uint8 = image

    H, W = img_uint8.shape
    lbp = np.zeros((H, W), dtype=np.uint8)

    angles = [2 * np.pi * p / P for p in range(P)]
    for y in range(R, H - R):
        for x in range(R, W - R):
            center = int(img_uint8[y, x])
            code = 0
            for p, angle in enumerate(angles):
                nx = x + R * np.cos(angle)
                ny = y - R * np.sin(angle)
                nx_i, ny_i = int(round(nx)), int(round(ny))
                nx_i = np.clip(nx_i, 0, W - 1)
                ny_i = np.clip(ny_i, 0, H - 1)
                if int(img_uint8[ny_i, nx_i]) >= center:
                    code |= 1 << p
            lbp[y, x] = code

    return lbp.flatten().astype(np.float64) / 255.0


def extract_lbp_fast(image: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    try:
        from skimage.feature import local_binary_pattern

        if image.dtype != np.uint8:
            img_uint8 = (
                (image * 255).astype(np.uint8)
                if image.max() <= 1.0
                else image.astype(np.uint8)
            )
        else:
            img_uint8 = image
        lbp = local_binary_pattern(img_uint8, P=P, R=R, method="uniform")
        return lbp.flatten().astype(np.float64) / lbp.max()
    except ImportError:
        return extract_lbp_features(image, P=P, R=R)


def extract_hog_features(
    image: np.ndarray,
    orientations: int = 8,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    try:
        from skimage.feature import hog

        if image.dtype != np.uint8:
            img_uint8 = (
                (image * 255).astype(np.uint8)
                if image.max() <= 1.0
                else image.astype(np.uint8)
            )
        else:
            img_uint8 = image
        features = hog(
            img_uint8,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            visualize=False,
        )
        return features.astype(np.float64)
    except ImportError:
        return _hog_numpy(image, orientations, pixels_per_cell)


def _hog_numpy(
    image: np.ndarray,
    orientations: int = 8,
    pixels_per_cell: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    if image.dtype != np.uint8:
        img = (
            (image * 255).astype(np.uint8)
            if image.max() <= 1.0
            else image.astype(np.uint8)
        )
    else:
        img = image.copy()

    img_float = img.astype(np.float64)

    Gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=1)
    Gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=1)

    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx) * (180.0 / np.pi) % 180.0

    H, W = img.shape
    cy, cx = pixels_per_cell
    n_cells_y = H // cy
    n_cells_x = W // cx

    histograms = []
    for y in range(n_cells_y):
        for x in range(n_cells_x):
            cell_mag = magnitude[y * cy : (y + 1) * cy, x * cx : (x + 1) * cx]
            cell_ori = orientation[y * cy : (y + 1) * cy, x * cx : (x + 1) * cx]
            hist, _ = np.histogram(
                cell_ori, bins=orientations, range=(0, 180), weights=cell_mag
            )
            norm = np.linalg.norm(hist)
            histograms.append(hist / (norm + 1e-6))

    return np.concatenate(histograms).astype(np.float64)


def extract_pixel_features(image: np.ndarray) -> np.ndarray:
    # PENTING: Jangan L2-normalize di sini!
    # Training di colab_pca_evaluation.py hanya membagi /255 (range [0,1]),
    # tidak ada L2-normalization. Kalau kita L2-normalize saat inferensi,
    # vektor berada di "alam semesta" yang berbeda → skor PCA kacau.
    # Gambar yang masuk ke sini sudah [0,1] dari preprocess_face().
    return image.flatten().astype(np.float64)


def extract_all_features(image: np.ndarray) -> dict:
    return {
        "lbp": extract_lbp_fast(image),
        "hog": extract_hog_features(image),
        "pixel": extract_pixel_features(image),
    }
