import numpy as np
from typing import Dict, Any

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

def custom_weighted_cosine_sim(w1: np.ndarray, w2: np.ndarray) -> float:
    weights = np.ones_like(w1)
    if len(weights) > 3:
        weights[:3] = 0.2
    if len(weights) > 15:
        weights[15:] = 0.5
        
    w1_scaled = w1 * weights
    w2_scaled = w2 * weights
    return cosine_similarity(w1_scaled, w2_scaled)

def ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    a, b = img1.flatten(), img2.flatten()
    C1, C2 = 0.01**2, 0.03**2
    mu1, mu2 = np.mean(a), np.mean(b)
    s1, s2 = np.var(a), np.var(b)
    s12 = np.mean((a - mu1) * (b - mu2))
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return float(np.clip(num / den if den != 0 else 0, 0, 1))

def compute_all_metrics(
    weights1: np.ndarray,
    weights2: np.ndarray,
    face1_display: np.ndarray,
    face2_display: np.ndarray,
    S_joint: np.ndarray,
    penalty_factor: float = 0.05,
    weights1_lbp: np.ndarray = None,
    weights2_lbp: np.ndarray = None,
    weights1_hog: np.ndarray = None,
    weights2_hog: np.ndarray = None,
    S_lbp: np.ndarray = None,
    S_hog: np.ndarray = None,
    alpha: float = 0.35,
    beta: float = 0.50,
    gamma: float = 0.15,
) -> Dict[str, float]:

    s_j = S_joint if S_joint is not None else np.ones_like(weights1)
    w1s = weights1 / (s_j ** 0.5 + 1e-8)
    w2s = weights2 / (s_j ** 0.5 + 1e-8)
    
    cos_eigen = float(custom_weighted_cosine_sim(w1s, w2s))
 
    # PERBAIKAN: Jarak Euclidean HARUS dihitung dari matriks yang sudah di-Scaling (Mahalanobis)!
    # Kalau pakai raw weights1, angkanya raksasa dan pinaltinya menghancurkan skor.
    w1_clean = w1s[3:] if len(w1s) > 3 else w1s
    w2_clean = w2s[3:] if len(w2s) > 3 else w2s
    euc_d     = float(np.linalg.norm(w1_clean - w2_clean))
    euc_sim   = float(np.exp(-0.02 * euc_d))
    
    score_pix = float(max(0, cos_eigen)) * euc_sim

    ssim      = ssim_simple(face1_display, face2_display)
    cos_pixel = float(cosine_similarity(face1_display.flatten(), face2_display.flatten()))

    result = {
        "cosine_similarity_eigenspace" : round(cos_eigen, 4),
        "euclidean_distance_eigenspace": round(euc_d, 4),
        "euclidean_similarity_norm"    : round(euc_sim, 4),
        "ssim_pixel"                   : round(ssim, 4),
        "cosine_similarity_pixel"      : round(cos_pixel, 4),
        "feature_mode"                 : "pixel",
    }

    is_fusion = (
        weights1_lbp is not None and weights2_lbp is not None and
        weights1_hog is not None and weights2_hog is not None
    )

    if is_fusion:
        s_lbp = S_lbp if S_lbp is not None else np.ones_like(weights1_lbp)
        w1_lbp_s = weights1_lbp / (s_lbp ** 0.5 + 1e-8)
        w2_lbp_s = weights2_lbp / (s_lbp ** 0.5 + 1e-8)
        
        cos_lbp   = float(custom_weighted_cosine_sim(w1_lbp_s, w2_lbp_s))
        wl1_clean = w1_lbp_s[3:] if len(w1_lbp_s) > 3 else w1_lbp_s
        wl2_clean = w2_lbp_s[3:] if len(w2_lbp_s) > 3 else w2_lbp_s
        d_lbp     = float(np.linalg.norm(wl1_clean - wl2_clean))
        score_lbp = float(max(0, cos_lbp)) * float(np.exp(-0.02 * d_lbp))

        s_hog = S_hog if S_hog is not None else np.ones_like(weights1_hog)
        w1_hog_s = weights1_hog / (s_hog ** 0.5 + 1e-8)
        w2_hog_s = weights2_hog / (s_hog ** 0.5 + 1e-8)
        
        cos_hog   = float(custom_weighted_cosine_sim(w1_hog_s, w2_hog_s))
        wh1_clean = w1_hog_s[3:] if len(w1_hog_s) > 3 else w1_hog_s
        wh2_clean = w2_hog_s[3:] if len(w2_hog_s) > 3 else w2_hog_s
        d_hog     = float(np.linalg.norm(wh1_clean - wh2_clean))
        score_hog = float(max(0, cos_hog)) * float(np.exp(-0.02 * d_hog))

        total_w   = alpha + beta + gamma
        composite = (alpha * score_lbp + beta * score_hog + gamma * score_pix) / total_w

        result.update({
            "cosine_lbp"     : round(cos_lbp, 4),
            "score_lbp"      : round(score_lbp, 4),
            "cosine_hog"     : round(cos_hog, 4),
            "score_hog"      : round(score_hog, 4),
            "score_pix"      : round(score_pix, 4),
            "composite_score": round(composite, 4),
            "feature_mode"   : "fusion",
        })
    else:
        result["composite_score"] = round(score_pix, 4)

    return result



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
