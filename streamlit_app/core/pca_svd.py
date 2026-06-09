from typing import Dict, Optional, Tuple

import numpy as np


def svd_decompose(
    image_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image_matrix.ndim == 1:
        image_matrix = image_matrix.reshape(1, -1)
    return np.linalg.svd(image_matrix, full_matrices=False)


def get_singular_values_info(S: np.ndarray) -> dict:
    total_energy = np.sum(S**2)
    explained_var = (S**2) / total_energy * 100
    cumulative_var = np.cumsum(explained_var)
    return {
        "singular_values": S,
        "explained_variance_pct": explained_var,
        "cumulative_variance_pct": cumulative_var,
        "total_energy": total_energy,
    }


def load_olivetti_dataset() -> Dict:
    from sklearn.datasets import fetch_olivetti_faces

    ds = fetch_olivetti_faces(shuffle=True, random_state=42)
    return {
        "images": ds.data,
        "images_2d": ds.images,
        "targets": ds.target,
        "n_samples": ds.data.shape[0],
        "n_people": len(np.unique(ds.target)),
        "image_shape": (64, 64),
        "pixel_size": 4096,
        "source": "Olivetti Faces (AT&T — 400 foto, 40 orang)",
        "description": "400 foto wajah dari 40 orang berbeda (10 foto/orang). 64x64 px, grayscale [0,1].",
    }


def load_lfw_dataset(min_faces: int = 20, resize: float = 0.4) -> Optional[Dict]:
    try:
        from sklearn.datasets import fetch_lfw_people

        ds = fetch_lfw_people(min_faces_per_person=min_faces, resize=resize)
        h, w = ds.images.shape[1], ds.images.shape[2]
        n = ds.data.shape[0]
        return {
            "images": ds.data,
            "images_2d": ds.images,
            "targets": ds.target,
            "target_names": ds.target_names,
            "n_samples": n,
            "n_people": len(ds.target_names),
            "image_shape": (h, w),
            "pixel_size": h * w,
            "source": f"LFW — {n} foto",
            "description": f"LFW: {n} foto dari {len(ds.target_names)} orang. Ukuran {h}x{w} px.",
        }
    except Exception:
        return None


def load_custom_selfie_dataset(
    base_path: str, target_size: Tuple[int, int] = (64, 64)
) -> Optional[Dict]:
    import os

    import cv2

    from .face_utils import preprocess_face
    from .feature_extractor import (
        extract_hog_features,
        extract_lbp_fast,
        extract_pixel_features,
    )

    if not os.path.exists(base_path):
        return None

    images_list = []
    lbp_list = []
    hog_list = []
    targets = []
    target_names = []
    person_id = 0

    for entry in sorted(os.listdir(base_path)):
        person_dir = os.path.join(base_path, entry)
        if not os.path.isdir(person_dir):
            continue

        target_names.append(entry)
        for sub in ["docs", "selfies"]:
            sub_dir = os.path.join(person_dir, sub)
            if not os.path.isdir(sub_dir):
                continue

            for fname in os.listdir(sub_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    img_path = os.path.join(sub_dir, fname)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        proc_face, info = preprocess_face(
                            img, detect=True, target_size=target_size
                        )
                        if proc_face is not None:
                            images_list.append(extract_pixel_features(proc_face))
                            lbp_list.append(extract_lbp_fast(proc_face))
                            hog_list.append(extract_hog_features(proc_face))
                            targets.append(person_id)

        person_id += 1

    if not images_list:
        return None

    images = np.array(images_list)
    lbp_arr = np.array(lbp_list)
    hog_arr = np.array(hog_list)
    images_2d = images.reshape(-1, target_size[0], target_size[1])
    n = images.shape[0]

    return {
        "images": images,
        "lbp_features": lbp_arr,
        "hog_features": hog_arr,
        "images_2d": images_2d,
        "targets": np.array(targets),
        "target_names": target_names,
        "n_samples": n,
        "n_people": len(target_names),
        "image_shape": target_size,
        "pixel_size": target_size[0] * target_size[1],
        "source": f"Selfie & ID ({n} foto, {len(target_names)} orang)",
        "description": f"Custom Dataset Selfie & ID lokal: {n} foto dari {len(target_names)} orang. Ukuran {target_size[0]}x{target_size[1]} px.",
    }


def load_pretrained_eigenspace(filepath: str) -> Optional[Dict]:
    import os

    if not os.path.exists(filepath):
        return None
    try:
        data = np.load(filepath)
        n = int(
            data["n_samples"].item()
            if data["n_samples"].ndim == 0
            else data["n_samples"][0]
        )
        k = int(
            data["k_components"].item()
            if data["k_components"].ndim == 0
            else data["k_components"][0]
        )
        shape = tuple(int(x) for x in data["image_shape"])
        sv = data["singular_values"]

        whiten_scale = sv / np.sqrt(max(1, n - 1))

        result = {
            "mean_face": data["mean_face"],
            "eigenfaces": data["eigenfaces"],
            "singular_values": sv,
            "whiten_scale": whiten_scale,
            "explained_variance_pct": data["explained_variance_pct"],
            "explained_variance_ratio": data["explained_variance_pct"] / 100.0,
            "eigenvalues": (sv**2) / max(1, n),
            "n_samples": n,
            "n_components": k,
            "image_shape": shape,
            "target_size": shape,
            "feature_mode": str(data.get("feature_mode", "pixel")),
        }

        if "aging_vector_pix" in data:
            result["aging_vector_pix"] = data["aging_vector_pix"]
            
        if "mean_lbp" in data:
            sv_lbp = data["singular_values_lbp"]
            result["mean_lbp"] = data["mean_lbp"]
            result["eigenfaces_lbp"] = data["eigenfaces_lbp"]
            result["singular_values_lbp"] = sv_lbp
            result["whiten_scale_lbp"] = sv_lbp / np.sqrt(max(1, n - 1))
            result["feature_mode"] = "fusion"
            if "aging_vector_lbp" in data:
                result["aging_vector_lbp"] = data["aging_vector_lbp"]

        if "mean_hog" in data:
            sv_hog = data["singular_values_hog"]
            result["mean_hog"] = data["mean_hog"]
            result["eigenfaces_hog"] = data["eigenfaces_hog"]
            result["singular_values_hog"] = sv_hog
            result["whiten_scale_hog"] = sv_hog / np.sqrt(max(1, n - 1))
            if "aging_vector_hog" in data:
                result["aging_vector_hog"] = data["aging_vector_hog"]

        n_str = f"{n} foto"
        mode = result["feature_mode"]
        result["source"] = f"Pretrained Model ({n_str}, mode={mode})"
        result["description"] = f"Model dilatih dengan {k} Eigenfaces. Mode: {mode}."
        return result
    except Exception:
        return None


def compute_mean_face(images: np.ndarray) -> np.ndarray:
    return np.mean(images, axis=0)


def compute_covariance_matrix(images_centered: np.ndarray) -> np.ndarray:
    return (images_centered.T @ images_centered) / images_centered.shape[0]


def compute_eigenfaces(
    images: np.ndarray,
    n_components: int = 50,
    use_svd: bool = True,
) -> dict:
    if images.ndim == 3:
        images = images.reshape(images.shape[0], -1)
    images = images.astype(float)
    mean_face = compute_mean_face(images)
    centered = images - mean_face
    n_components = min(n_components, images.shape[0], images.shape[1])

    if use_svd:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        eigenfaces = Vt[:n_components]
        eigenvalues = (S[:n_components] ** 2) / images.shape[0]
        ev_ratio = (S**2) / np.sum(S**2)
        singular_values = S
    else:
        C = compute_covariance_matrix(centered)
        vals, vecs = np.linalg.eig(C)
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx].real, vecs[:, idx].real
        eigenfaces = vecs[:, :n_components].T
        eigenvalues = vals[:n_components]
        ev_ratio = vals / np.sum(vals)
        singular_values = None

    return {
        "eigenfaces": eigenfaces,
        "eigenvalues": eigenvalues,
        "mean_face": mean_face,
        "explained_variance_ratio": ev_ratio[:n_components],
        "n_components": n_components,
        "singular_values": singular_values,
        "n_training_images": images.shape[0],
    }


def build_eigenspace_from_dataset(
    dataset: dict,
    n_components: int = 50,
    target_size: Tuple[int, int] = (64, 64),
) -> dict:
    images = dataset if isinstance(dataset, np.ndarray) else dataset.get("images")
    ef = compute_eigenfaces(images, n_components=n_components)
    cumvar = np.cumsum(ef["explained_variance_ratio"])
    result = {
        **ef,
        "target_size": target_size,
        "k_for_95pct_variance": int(np.searchsorted(cumvar, 0.95)) + 1,
        "k_for_99pct_variance": int(np.searchsorted(cumvar, 0.99)) + 1,
        "total_variance_captured": float(np.sum(ef["explained_variance_ratio"])),
        "dataset_size": images.shape[0],
        "feature_mode": "pixel",
    }

    if isinstance(dataset, dict):
        if "lbp_features" in dataset and dataset["lbp_features"] is not None:
            ef_lbp = compute_eigenfaces(
                dataset["lbp_features"], n_components=n_components
            )
            result["mean_lbp"] = ef_lbp["mean_face"]
            result["eigenfaces_lbp"] = ef_lbp["eigenfaces"]
            result["singular_values_lbp"] = ef_lbp["singular_values"]
            result["ev_ratio_lbp"] = ef_lbp["explained_variance_ratio"]
            result["feature_mode"] = "fusion"

        if "hog_features" in dataset and dataset["hog_features"] is not None:
            ef_hog = compute_eigenfaces(
                dataset["hog_features"], n_components=n_components
            )
            result["mean_hog"] = ef_hog["mean_face"]
            result["eigenfaces_hog"] = ef_hog["eigenfaces"]
            result["singular_values_hog"] = ef_hog["singular_values"]
            result["ev_ratio_hog"] = ef_hog["explained_variance_ratio"]

    return result


def project_to_eigenspace(
    face: np.ndarray,
    eigenfaces: np.ndarray,
    mean_face: np.ndarray,
) -> np.ndarray:
    if face.ndim > 1:
        face = face.flatten()
    return eigenfaces @ (face.astype(float) - mean_face)


def reconstruct_from_eigenspace(
    weights: np.ndarray,
    eigenfaces: np.ndarray,
    mean_face: np.ndarray,
) -> np.ndarray:
    return mean_face + weights @ eigenfaces


def resize_face_for_eigenspace(
    face: np.ndarray,
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    if face.shape == target_size:
        return face.astype(np.float64)
    import cv2

    return cv2.resize(
        face.astype(np.float32),
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float64)


def analyze_two_faces(
    face1: np.ndarray,
    face2: np.ndarray,
    n_components: int = 2,
) -> dict:
    f1 = face1.flatten().astype(float)
    f2 = face2.flatten().astype(float)
    ef = compute_eigenfaces(np.stack([f1, f2]), n_components=min(n_components, 2))
    w1 = project_to_eigenspace(f1, ef["eigenfaces"], ef["mean_face"])
    w2 = project_to_eigenspace(f2, ef["eigenfaces"], ef["mean_face"])
    r1 = reconstruct_from_eigenspace(w1, ef["eigenfaces"], ef["mean_face"])
    r2 = reconstruct_from_eigenspace(w2, ef["eigenfaces"], ef["mean_face"])
    U1, S1, Vt1 = svd_decompose(face1.astype(float))
    U2, S2, Vt2 = svd_decompose(face2.astype(float))
    return {
        "face1_flat": f1,
        "face2_flat": f2,
        "eigenface_data": ef,
        "weights_face1": w1,
        "weights_face2": w2,
        "reconstructed_face1": r1,
        "reconstructed_face2": r2,
        "svd_face1": {"U": U1, "S": S1, "Vt": Vt1},
        "svd_face2": {"U": U2, "S": S2, "Vt": Vt2},
        "singular_values_joint": ef["singular_values"],
    }


def analyze_two_faces_with_dataset(
    face1: np.ndarray,
    face2: np.ndarray,
    eigenspace: dict,
    apply_aging: bool = False,
    prob_asian: float = None,
) -> dict:
    """
    Proyeksikan dua wajah ke Eigenspace.
    Mendukung mode Pixel-only dan LBP+HOG+Pixel Fusion secara otomatis.
    Jika apply_aging=True, vektor wajah 1 (foto lama) akan disuntik dengan Vektor Penuaan Hybrid.
    """
    from .feature_extractor import (
        extract_hog_features,
        extract_lbp_fast,
        extract_pixel_features,
    )
    from .ethnicity_classifier import get_hybrid_aging_vector

    target_size = eigenspace.get("target_size", (64, 64))
    face1_r = resize_face_for_eigenspace(face1, target_size)
    face2_r = resize_face_for_eigenspace(face2, target_size)

    f1_pix = extract_pixel_features(face1_r)
    f2_pix = extract_pixel_features(face2_r)

    ef_pix = eigenspace["eigenfaces"]
    mf_pix = eigenspace["mean_face"]
    w1_pix = project_to_eigenspace(f1_pix, ef_pix, mf_pix)
    w2_pix = project_to_eigenspace(f2_pix, ef_pix, mf_pix)

    if apply_aging and ("aging_vector_pix" in eigenspace or "aging_vector_aaf_pix" in eigenspace):
        s_pix = eigenspace.get("singular_values")
        n_samp = eigenspace.get("eigenspace_info", {}).get("dataset_size", 3359)
        if s_pix is not None:
            # Mengambil vektor FGNET dan AAF. Jika belum ada AAF, pakai FGNET sebagai fallback
            v_fgnet = eigenspace.get("aging_vector_pix", eigenspace.get("aging_vector_fgnet_pix"))
            v_aaf = eigenspace.get("aging_vector_aaf_pix", v_fgnet)
            
            # Jika prob_asian diberikan, hitung hybrid. Jika tidak, pakai yang default
            if prob_asian is not None and v_aaf is not None and v_fgnet is not None:
                hybrid_vector = get_hybrid_aging_vector(prob_asian, v_aaf, v_fgnet)
            else:
                hybrid_vector = v_fgnet
                
            if hybrid_vector is not None:
                unwhiten_scale = s_pix / np.sqrt(max(1, n_samp - 1))
                w1_pix = w1_pix + (hybrid_vector * unwhiten_scale * 0.35)

    ws_pix = eigenspace.get("whiten_scale")
    if ws_pix is not None:
        w1_pix = w1_pix / (ws_pix + 1e-8)
        w2_pix = w2_pix / (ws_pix + 1e-8)

    r1 = reconstruct_from_eigenspace(w1_pix, ef_pix, mf_pix).reshape(target_size)
    r2 = reconstruct_from_eigenspace(w2_pix, ef_pix, mf_pix).reshape(target_size)

    U1, S1, Vt1 = svd_decompose(face1_r)
    U2, S2, Vt2 = svd_decompose(face2_r)

    result = {
        "face1_resized": face1_r,
        "face2_resized": face2_r,
        "face1_flat": f1_pix,
        "face2_flat": f2_pix,
        "weights_face1": w1_pix,
        "weights_face2": w2_pix,
        "reconstructed_face1": r1,
        "reconstructed_face2": r2,
        "svd_face1": {"U": U1, "S": S1, "Vt": Vt1},
        "svd_face2": {"U": U2, "S": S2, "Vt": Vt2},
        "singular_values_joint": eigenspace["singular_values"],
        "n_components_used": len(w1_pix),
        "feature_mode": eigenspace.get("feature_mode", "pixel"),
        "eigenspace_info": {
            "dataset_size": eigenspace.get("dataset_size"),
            "n_components": eigenspace.get("n_components"),
            "total_variance_captured": eigenspace.get("total_variance_captured", 0),
            "k_for_95pct": eigenspace.get("k_for_95pct_variance"),
            "feature_mode": eigenspace.get("feature_mode", "pixel"),
        },
    }

    if eigenspace.get("feature_mode") == "fusion":
        f1_lbp = extract_lbp_fast(face1_r)
        f2_lbp = extract_lbp_fast(face2_r)
        ef_lbp = eigenspace["eigenfaces_lbp"]
        mf_lbp = eigenspace["mean_lbp"]
        w1_lbp = project_to_eigenspace(f1_lbp, ef_lbp, mf_lbp)
        w2_lbp = project_to_eigenspace(f2_lbp, ef_lbp, mf_lbp)
        
        if apply_aging and ("aging_vector_lbp" in eigenspace or "aging_vector_aaf_lbp" in eigenspace):
            s_lbp = eigenspace.get("singular_values_lbp")
            if s_lbp is not None:
                n_samp = eigenspace.get("eigenspace_info", {}).get("dataset_size", 3359)
                v_fgnet_lbp = eigenspace.get("aging_vector_lbp", eigenspace.get("aging_vector_fgnet_lbp"))
                v_aaf_lbp = eigenspace.get("aging_vector_aaf_lbp", v_fgnet_lbp)
                if prob_asian is not None and v_aaf_lbp is not None and v_fgnet_lbp is not None:
                    hybrid_vector_lbp = get_hybrid_aging_vector(prob_asian, v_aaf_lbp, v_fgnet_lbp)
                else:
                    hybrid_vector_lbp = v_fgnet_lbp
                if hybrid_vector_lbp is not None:
                    unwhiten_lbp = s_lbp / np.sqrt(max(1, n_samp - 1))
                    w1_lbp = w1_lbp + (hybrid_vector_lbp * unwhiten_lbp * 0.35)
            
        ws_lbp = eigenspace.get("whiten_scale_lbp")
        if ws_lbp is not None:
            w1_lbp = w1_lbp / (ws_lbp + 1e-8)
            w2_lbp = w2_lbp / (ws_lbp + 1e-8)
        result["weights_face1_lbp"] = w1_lbp
        result["weights_face2_lbp"] = w2_lbp
        result["singular_values_lbp"] = eigenspace.get("singular_values_lbp")

        f1_hog = extract_hog_features(face1_r)
        f2_hog = extract_hog_features(face2_r)
        ef_hog = eigenspace["eigenfaces_hog"]
        mf_hog = eigenspace["mean_hog"]
        w1_hog = project_to_eigenspace(f1_hog, ef_hog, mf_hog)
        
        if apply_aging and ("aging_vector_hog" in eigenspace or "aging_vector_aaf_hog" in eigenspace):
            s_hog = eigenspace.get("singular_values_hog")
            if s_hog is not None:
                n_samp = eigenspace.get("eigenspace_info", {}).get("dataset_size", 3359)
                v_fgnet_hog = eigenspace.get("aging_vector_hog", eigenspace.get("aging_vector_fgnet_hog"))
                v_aaf_hog = eigenspace.get("aging_vector_aaf_hog", v_fgnet_hog)
                if prob_asian is not None and v_aaf_hog is not None and v_fgnet_hog is not None:
                    hybrid_vector_hog = get_hybrid_aging_vector(prob_asian, v_aaf_hog, v_fgnet_hog)
                else:
                    hybrid_vector_hog = v_fgnet_hog
                if hybrid_vector_hog is not None:
                    unwhiten_hog = s_hog / np.sqrt(max(1, n_samp - 1))
                    w1_hog = w1_hog + (hybrid_vector_hog * unwhiten_hog * 0.35)
        w2_hog = project_to_eigenspace(f2_hog, ef_hog, mf_hog)
        ws_hog = eigenspace.get("whiten_scale_hog")
        if ws_hog is not None:
            w1_hog = w1_hog / (ws_hog + 1e-8)
            w2_hog = w2_hog / (ws_hog + 1e-8)
        result["weights_face1_hog"] = w1_hog
        result["weights_face2_hog"] = w2_hog
        result["singular_values_hog"] = eigenspace.get("singular_values_hog")

    return result
