import numpy as np
from typing import Tuple, Optional, Dict, Any


DECISION_THRESHOLD = 0.68

THRESHOLDS = {
    "identical"   : 0.95,
    "very_similar": 0.85,
    "similar"     : 0.70,
    "uncertain"   : 0.55,
}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(float)
    b_flat = b.flatten().astype(float)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.clip(np.dot(a_flat, b_flat) / (norm_a * norm_b), -1.0, 1.0))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.flatten().astype(float) - b.flatten().astype(float)))


def normalized_euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 / (1.0 + euclidean_distance(a, b)))


def structural_similarity_pixels(img1: np.ndarray, img2: np.ndarray) -> float:
    a = img1.flatten().astype(float)
    b = img2.flatten().astype(float)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = np.mean(a), np.mean(b)
    s1, s2   = np.var(a), np.var(b)
    s12      = np.mean((a - mu1) * (b - mu2))
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return float(np.clip(num / den if den != 0 else 0.0, 0.0, 1.0))


def compute_all_metrics(
    weights1: np.ndarray,
    weights2: np.ndarray,
    face1_pixel: np.ndarray,
    face2_pixel: np.ndarray,
) -> Dict[str, float]:
    cos_sim    = cosine_similarity(weights1, weights2)
    euc_dist   = euclidean_distance(weights1, weights2)
    euc_sim    = normalized_euclidean_similarity(weights1, weights2)
    ssim_score = structural_similarity_pixels(face1_pixel, face2_pixel)
    pixel_cos  = cosine_similarity(face1_pixel, face2_pixel)
    # TIE-BREAKER: Pinalti Jarak Euclidean
    # Diperingan menjadi 0.90 + 0.10 (sebelumnya 0.80 + 0.20)
    # Karena foto lama (blur) dan foto baru (tajam) memiliki perbedaan magnitudo yang besar.
    penalty_factor = 0.90 + (0.10 * euc_sim)

    composite = float(max(0, cos_sim)) * penalty_factor
    return {
        "cosine_similarity_eigenspace" : round(cos_sim, 4),
        "euclidean_distance_eigenspace": round(euc_dist, 4),
        "euclidean_similarity_norm"    : round(euc_sim, 4),
        "ssim_pixel"                   : round(ssim_score, 4),
        "cosine_similarity_pixel"      : round(pixel_cos, 4),
        "composite_score"              : round(composite, 4),
    }


def make_decision(
    metrics: Dict[str, float],
    threshold: float = DECISION_THRESHOLD,
) -> Dict[str, Any]:
    score   = metrics["composite_score"]
    cos     = metrics["cosine_similarity_eigenspace"]
    is_same = score >= threshold

    if cos >= THRESHOLDS["identical"]:
        level, confidence, color = "Identik", "Sangat Tinggi", "#10b981"
    elif cos >= THRESHOLDS["very_similar"]:
        level, confidence, color = "Sangat Mirip", "Tinggi", "#22c55e"
    elif cos >= THRESHOLDS["similar"]:
        level, confidence, color = "Mirip", "Sedang", "#f59e0b"
    elif cos >= THRESHOLDS["uncertain"]:
        level, confidence, color = "Kurang Mirip", "Rendah", "#f97316"
    else:
        level, confidence, color = "Tidak Mirip", "Sangat Rendah", "#ef4444"

    euc  = metrics["euclidean_distance_eigenspace"]
    ssim = metrics["ssim_pixel"]
    reasoning = [
        f"Cosine similarity eigenspace: {cos:.2%} ({'tinggi' if cos >= 0.70 else 'rendah'})",
        f"Euclidean distance: {euc:.3f} ({'berdekatan' if euc < 1.0 else 'berjauhan'})",
        f"SSIM pixel: {ssim:.2%}",
    ]

    return {
        "is_same_person" : is_same,
        "verdict"        : "[SAMA] Orang yang Sama" if is_same else "[BEDA] Orang yang Berbeda",
        "verdict_display": "\u2705 Orang yang Sama" if is_same else "\u274c Orang yang Berbeda",
        "verdict_en"     : "Same Person" if is_same else "Different Person",
        "score"          : score,
        "level"          : level,
        "confidence"     : confidence,
        "color"          : color,
        "threshold_used" : threshold,
        "reasoning"      : reasoning,
        "metrics"        : metrics,
    }
