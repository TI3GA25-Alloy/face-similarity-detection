import json
import base64
import numpy as np
import cv2
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

def preprocess_image(image_b64: str):
    img_bytes = base64.b64decode(image_b64.split(",")[-1])
    pil_img   = Image.open(BytesIO(img_bytes)).convert("RGB")
    np_img    = np.array(pil_img)
    gray      = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    cascade      = cv2.CascadeClassifier(HAAR_FRONTAL)
    faces        = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    face_detected = len(faces) > 0

    if face_detected:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        pad = int(min(w, h) * 0.2)
        H, W = gray.shape
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
        gray = gray[y1:y2, x1:x2]

    gray       = cv2.equalizeHist(gray)
    resized    = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float64) / 255.0

    return normalized, face_detected

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

    stack      = np.stack([f1, f2], axis=0)
    mean_face  = np.mean(stack, axis=0)
    centered   = stack - mean_face

    _, S_joint, Vt_joint = np.linalg.svd(centered, full_matrices=False)
    n_comp     = min(2, len(S_joint))
    eigenfaces = Vt_joint[:n_comp]
    eigenvalues = (S_joint[:n_comp] ** 2) / 2

    w1 = eigenfaces @ (f1 - mean_face)
    w2 = eigenfaces @ (f2 - mean_face)

    cos_eigen = cosine_sim(w1, w2)
    euc_d     = float(np.linalg.norm(w1 - w2))
    euc_sim   = 1.0 / (1.0 + euc_d)
    ssim      = ssim_simple(face1, face2)
    cos_pixel = cosine_sim(f1, f2)
    composite = 0.45*max(0, cos_eigen) + 0.25*euc_sim + 0.20*ssim + 0.10*max(0, cos_pixel)

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
        "weights_face1": [float(v) for v in w1],
        "weights_face2": [float(v) for v in w2],
        "singular_values_face1": sv_info(S1),
        "singular_values_face2": sv_info(S2),
        "singular_values_joint": [float(v) for v in S_joint],
    }


def make_decision(composite: float, cos_eigen: float, threshold: float = 0.70):
    is_same = composite >= threshold
    if   cos_eigen >= 0.95: level, confidence, color = "Identik",      "Sangat Tinggi", "#10b981"
    elif cos_eigen >= 0.85: level, confidence, color = "Sangat Mirip", "Tinggi",        "#22c55e"
    elif cos_eigen >= 0.70: level, confidence, color = "Mirip",        "Sedang",        "#f59e0b"
    elif cos_eigen >= 0.55: level, confidence, color = "Kurang Mirip", "Rendah",        "#f97316"
    else:                   level, confidence, color = "Tidak Mirip",  "Sangat Rendah", "#ef4444"
    return {
        "is_same_person": bool(is_same),
        "verdict":      "Orang yang Sama" if is_same else "Orang yang Berbeda",
        "verdict_icon": "✅" if is_same else "❌",
        "level": level, "confidence": confidence, "color": color,
        "threshold_used": threshold,
    }

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyze")
async def analyze(request: Request):
    try:
        body      = await request.json()
        image1_b64 = body.get("image1")
        image2_b64 = body.get("image2")
        threshold  = float(body.get("threshold", 0.70))

        if not image1_b64 or not image2_b64:
            return JSONResponse({"error": "Kedua gambar diperlukan"}, status_code=400)

        face1, detected1 = preprocess_image(image1_b64)
        face2, detected2 = preprocess_image(image2_b64)

        result   = run_pca_svd(face1, face2)
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
