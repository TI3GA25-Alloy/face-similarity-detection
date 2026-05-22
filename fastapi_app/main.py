import json
import base64
import numpy as np
import cv2
import os
from io import BytesIO
from PIL import Image
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="FaceMatch PCA/SVD")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

TARGET_SIZE = (128, 128)
HAAR_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
NPZ_PATH = Path(__file__).parent.parent / "pretrained_eigenspace.npz"

PRETRAINED = None
if os.path.exists(NPZ_PATH):
    try:
        data = np.load(NPZ_PATH)
        PRETRAINED = {
            "mean_face": data['mean_face'],
            "eigenfaces": data['eigenfaces'],
            "singular_values": data['singular_values'],
        }
        print(f"✅ Pretrained model loaded! ({len(PRETRAINED['eigenfaces'])} eigenfaces)")
    except Exception as e:
        print(f"❌ Error loading .npz: {e}")

def extract_edges(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def detect_and_crop(contents: str):
    import base64
    if "," in contents:
        contents = contents.split(",")[1]
    img_bytes = base64.b64decode(contents)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gambar tidak valid")
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cascade = cv2.CascadeClassifier(HAAR_FRONTAL)
    faces = cascade.detectMultiScale(img_gray, 1.1, 5, minSize=(30, 30))
    
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        pad = int(min(w, h) * 0.2)
        H, W = img_gray.shape
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
        return img_gray[y1:y2, x1:x2], True
    return img_gray, False

def process_cropped(gray, angle=0.0):
    if angle != 0.0:
        h, w = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
        
    resized    = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    edge_map   = extract_edges(resized)
    return edge_map.astype(np.float64) / 255.0

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def ssim_simple(img1, img2):
    a, b   = img1.flatten(), img2.flatten()
    C1, C2 = 0.01**2, 0.03**2
    mu1, mu2   = np.mean(a), np.mean(b)
    s1, s2     = np.var(a), np.var(b)
    s12        = np.mean((a - mu1) * (b - mu2))
    num = (2*mu1*mu2 + C1) * (2*s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return float(np.clip(num / den if den != 0 else 0, 0, 1))

def run_pca_svd(face1: np.ndarray, face2: np.ndarray):
    f1, f2 = face1.flatten(), face2.flatten()

    U1, S1, _  = np.linalg.svd(face1, full_matrices=False)
    U2, S2, _  = np.linalg.svd(face2, full_matrices=False)

    if PRETRAINED is not None:
        ef = PRETRAINED["eigenfaces"]
        mf = PRETRAINED["mean_face"]
        w1 = ef @ (f1 - mf)
        w2 = ef @ (f2 - mf)
        S_joint = PRETRAINED["singular_values"]
        eigenvalues = (S_joint ** 2) / 2
    else:
        stack = np.stack([f1, f2], axis=0)
        mf = np.mean(stack, axis=0)
        centered = stack - mf
        _, S_joint, Vt_joint = np.linalg.svd(centered, full_matrices=False)
        n_comp = min(2, len(S_joint))
        ef = Vt_joint[:n_comp]
        eigenvalues = (S_joint[:n_comp] ** 2) / 2
        w1 = ef @ (f1 - mf)
        w2 = ef @ (f2 - mf)

    cos_eigen = cosine_sim(w1, w2)
    euc_d     = float(np.linalg.norm(w1 - w2))
    euc_sim   = 1.0 / (1.0 + euc_d)
    
    # Kita tetap menghitungnya untuk ditampilkan di UI (meski tidak dipakai di skor akhir)
    ssim      = ssim_simple(f1, f2)
    cos_pixel = cosine_sim(f1, f2)
    
    # TIE-BREAKER: Pinalti Jarak Euclidean
    # Jika arah kosinusnya sama (mirip) tapi jarak vektornya berjauhan (orang berbeda/adik),
    # kita berikan diskon pemotongan skor maksimal 20%.
    penalty_factor = 0.90 + (0.10 * euc_sim)
    
    # Karena ini tugas Aljabar Linear, skor akhir WAJIB mencerminkan hasil SVD/PCA.
    composite = float(max(0, cos_eigen)) * penalty_factor

    def sv_info(S):
        total = np.sum(S**2)
        return [
            {"rank": int(i+1), "value": float(S[i]),
             "variance_pct": float(S[i]**2 / total * 100)}
            for i in range(min(15, len(S)))
        ]

    return {
        "metrics": {
            "cosine_similarity_eigenspace": round(cos_eigen, 4),
            "euclidean_distance_eigenspace": round(euc_d, 4),
            "euclidean_similarity_norm": round(euc_sim, 4),
            "ssim_pixel": round(ssim, 4),
            "cosine_similarity_pixel": round(cos_pixel, 4),
            "composite_score": round(composite, 4),
        },
        "eigenvalues": [float(v) for v in eigenvalues],
        "weights_face1": [float(v) for v in w1[:15]],
        "weights_face2": [float(v) for v in w2[:15]],
        "singular_values_face1": sv_info(S1),
        "singular_values_face2": sv_info(S2),
        "singular_values_joint": [float(v) for v in S_joint[:15]],
    }


def make_decision(composite: float, cos_eigen: float, threshold: float = 0.60):
    is_same = composite >= threshold

    if   composite >= 0.85: level, confidence, color = "Sangat Mirip", "Sangat Tinggi", "#10b981"
    elif composite >= 0.75: level, confidence, color = "Mirip", "Tinggi", "#22c55e"
    elif composite >= 0.65: level, confidence, color = "Cukup Mirip", "Sedang", "#f59e0b"
    elif composite >= 0.50: level, confidence, color = "Kurang Mirip", "Rendah", "#f97316"
    else:                   level, confidence, color = "Tidak Mirip", "Sangat Rendah", "#ef4444"
    
    return {
        "is_same_person": bool(is_same),
        "verdict":      "Orang yang Sama" if is_same else "Orang yang Berbeda",
        "verdict_icon": "✅" if is_same else "❌",
        "level": level, "confidence": confidence, "color": color,
        "threshold_used": threshold,
    }

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/analyze")
async def analyze(request: Request):
    try:
        if PRETRAINED is None:
            print("⚠️ Peringatan: pretrained_eigenspace.npz tidak dimuat, akurasi akan buruk!")
            
        body      = await request.json()
        image1_b64 = body.get("image1")
        image2_b64 = body.get("image2")
        threshold  = float(body.get("threshold", 0.68))

        if not image1_b64 or not image2_b64:
            return JSONResponse({"error": "Kedua gambar diperlukan"}, status_code=400)

        crop1, detected1 = detect_and_crop(image1_b64)
        crop2, detected2 = detect_and_crop(image2_b64)

        face1 = process_cropped(crop1, angle=0.0)

        best_cos = -1.0
        best_result = None

        for angle in [0.0, -10.0, 10.0, -5.0, 5.0]:
            f2 = process_cropped(crop2, angle=angle)
            res = run_pca_svd(face1, f2)
            c = res["metrics"]["cosine_similarity_eigenspace"]
            if c > best_cos:
                best_cos = c
                best_result = res

        result = best_result
        decision = make_decision(
            result["metrics"]["composite_score"],
            result["metrics"]["cosine_similarity_eigenspace"],
            threshold,
        )

        return JSONResponse({
            "success": True,
            "decision": decision,
            "metrics": result["metrics"],
            "math_data": {
                "eigenvalues":           result["eigenvalues"],
                "weights_face1":         result["weights_face1"],
                "weights_face2":         result["weights_face2"],
                "singular_values_face1": result["singular_values_face1"],
                "singular_values_face2": result["singular_values_face2"],
                "singular_values_joint": result["singular_values_joint"],
            },
            "preprocessing": {
                "face1_detected": detected1,
                "face2_detected": detected2,
                "image_size": f"{TARGET_SIZE[0]}×{TARGET_SIZE[1]}",
            },
        })

    except Exception as e:
        return JSONResponse({"error": str(e), "success": False}, status_code=500)
