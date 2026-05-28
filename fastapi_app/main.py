import sys
import os
from pathlib import Path
import base64
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from streamlit_app.core.face_utils import detect_face, preprocess_face
from streamlit_app.core.pca_svd import load_pretrained_eigenspace
from streamlit_app.core.feature_extractor import analyze_two_faces_with_dataset, analyze_two_faces
from streamlit_app.core.similarity import compute_all_metrics


app = FastAPI(title="FaceMatch PCA/SVD API")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

def get_npz_path():
    paths = [
        Path(__file__).parent.parent / "pretrained_eigenspace.npz",
        Path(os.getcwd()) / "pretrained_eigenspace.npz",
        Path("/var/task/pretrained_eigenspace.npz")
    ]
    for p in paths:
        if p.exists():
            return p
    return None

NPZ_PATH = get_npz_path()
PRETRAINED = None

if NPZ_PATH is not None:
    try:
        PRETRAINED = load_pretrained_eigenspace(str(NPZ_PATH))
        print(f"[OK] Pretrained loaded from {NPZ_PATH}")
    except Exception as e:
        print(f"[Error] loading .npz: {e}")
else:
    print("[Error] pretrained_eigenspace.npz not found anywhere!")

def decode_b64_image(contents: str) -> np.ndarray:
    if "," in contents:
        contents = contents.split(",")[1]
    img_bytes = base64.b64decode(contents)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gambar tidak valid")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def make_decision(composite: float, threshold: float = 0.60):
    is_same = composite >= threshold
    if   composite >= 0.85: level, confidence, color = "Sangat Mirip", "Sangat Tinggi", "#10b981"
    elif composite >= 0.75: level, confidence, color = "Mirip", "Tinggi", "#22c55e"
    elif composite >= 0.65: level, confidence, color = "Cukup Mirip", "Sedang", "#f59e0b"
    elif composite >= 0.50: level, confidence, color = "Kurang Mirip", "Rendah", "#f97316"
    else:                   level, confidence, color = "Tidak Mirip", "Sangat Rendah", "#ef4444"
    return {
        "score": composite,
        "is_same_person": bool(is_same),
        "verdict":      "Orang yang Sama" if is_same else "Orang yang Berbeda",
        "verdict_icon": "✅" if is_same else "❌",
        "level": level, "confidence": confidence, "color": color,
        "threshold_used": threshold,
    }

def sv_info(S):
    total = np.sum(np.array(S)**2) if len(S) > 0 else 1
    return [
        {"rank": int(i+1), "value": float(S[i]), "variance_pct": float(S[i]**2 / total * 100)}
        for i in range(min(15, len(S)))
    ]

# --- ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/analyze")
async def analyze(request: Request):
    try:
        if PRETRAINED is None:
            print("[Warning] pretrained_eigenspace.npz tidak dimuat, akurasi akan buruk!")
            
        body      = await request.json()
        image1_b64 = body.get("image1")
        image2_b64 = body.get("image2")
        threshold  = float(body.get("threshold", 0.68))

        if not image1_b64 or not image2_b64:
            return JSONResponse({"error": "Kedua gambar diperlukan"}, status_code=400)

        gray1 = decode_b64_image(image1_b64)
        gray2 = decode_b64_image(image2_b64)

        bbox1 = detect_face(gray1)
        bbox2 = detect_face(gray2)

        # Fallback to entire image if no face detected
        if not bbox1:
            H1, W1 = gray1.shape
            bbox1 = (0, 0, W1, H1)
        if not bbox2:
            H2, W2 = gray2.shape
            bbox2 = (0, 0, W2, H2)

        # Loop pre-processing and comparison with multiple angles
        best_cos = -1.0
        best_result = None
        best_metrics = None

        face1_proc, _ = preprocess_face(gray1, bbox1, forced_angle=0.0)

        for angle in [0.0, -10.0, 10.0, -5.0, 5.0]:
            for do_flip in [False, True]:
                f2_proc_base, _ = preprocess_face(gray2, bbox2, forced_angle=angle)
                
                if do_flip:
                    f2_proc = cv2.flip(f2_proc_base, 1)
                else:
                    f2_proc = f2_proc_base
                
                if PRETRAINED is not None:
                    res = analyze_two_faces_with_dataset(face1_proc, f2_proc, PRETRAINED)
                else:
                    res = analyze_two_faces(face1_proc, f2_proc)
                
                w1 = res["weights_face1"]
                w2 = res["weights_face2"]
                S_joint = res["singular_values_joint"]

                # Prepare fusion arguments if available
                fusion_args = {}
                if "weights_face1_lbp" in res:
                    fusion_args["weights1_lbp"] = res["weights_face1_lbp"]
                    fusion_args["weights2_lbp"] = res["weights_face2_lbp"]
                    fusion_args["weights1_hog"] = res["weights_face1_hog"]
                    fusion_args["weights2_hog"] = res["weights_face2_hog"]
                    fusion_args["S_lbp"] = res.get("singular_values_lbp")
                    fusion_args["S_hog"] = res.get("singular_values_hog")

                mets = compute_all_metrics(
                    w1, w2, 
                    res["face1_resized"], res["face2_resized"], 
                    S_joint, 
                    penalty_factor=0.05, 
                    **fusion_args
                )

                c = mets["cosine_similarity_eigenspace"]
                if c > best_cos:
                    best_cos = c
                    best_result = res
                    best_metrics = mets

        decision = make_decision(best_metrics["composite_score"], threshold)

        # Build Response
        S1 = best_result["svd_face1"]["S"]
        S2 = best_result["svd_face2"]["S"]
        S_joint_array = np.array(best_result["singular_values_joint"])
        eigenvalues = (S_joint_array ** 2) / 2

        return JSONResponse({
            "success": True,
            "decision": decision,
            "metrics": best_metrics,
            "math_data": {
                "eigenvalues":           [float(v) for v in eigenvalues],
                "weights_face1":         [float(v) for v in best_result["weights_face1"][:15]],
                "weights_face2":         [float(v) for v in best_result["weights_face2"][:15]],
                "singular_values_face1": sv_info(S1),
                "singular_values_face2": sv_info(S2),
                "singular_values_joint": [float(v) for v in best_result["singular_values_joint"][:15]],
            },
            "preprocessing": {
                "face1_detected": bool(bbox1),
                "face2_detected": bool(bbox2),
                "image_size": f"128x128",
            },
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e), "success": False}, status_code=500)
