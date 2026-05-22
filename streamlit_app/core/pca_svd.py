import numpy as np
from typing import Tuple, Optional, Dict


def svd_decompose(image_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image_matrix.ndim == 1:
        image_matrix = image_matrix.reshape(1, -1)
    return np.linalg.svd(image_matrix, full_matrices=False)


def get_singular_values_info(S: np.ndarray) -> dict:
    total_energy     = np.sum(S ** 2)
    explained_var    = (S ** 2) / total_energy * 100
    cumulative_var   = np.cumsum(explained_var)
    return {
        "singular_values"       : S,
        "explained_variance_pct": explained_var,
        "cumulative_variance_pct": cumulative_var,
        "total_energy"          : total_energy,
    }


def load_olivetti_dataset() -> Dict:
    from sklearn.datasets import fetch_olivetti_faces
    ds = fetch_olivetti_faces(shuffle=True, random_state=42)
    return {
        "images"     : ds.data,
        "images_2d"  : ds.images,
        "targets"    : ds.target,
        "n_samples"  : ds.data.shape[0],
        "n_people"   : len(np.unique(ds.target)),
        "image_shape": (64, 64),
        "pixel_size" : 4096,
        "source"     : "Olivetti Faces (AT&T — 400 foto, 40 orang)",
        "description": "400 foto wajah dari 40 orang berbeda (10 foto/orang). 64x64 px, grayscale [0,1].",
    }


def load_lfw_dataset(min_faces: int = 20, resize: float = 0.4) -> Optional[Dict]:
    try:
        from sklearn.datasets import fetch_lfw_people
        ds   = fetch_lfw_people(min_faces_per_person=min_faces, resize=resize)
        h, w = ds.images.shape[1], ds.images.shape[2]
        n    = ds.data.shape[0]
        return {
            "images"      : ds.data,
            "images_2d"   : ds.images,
            "targets"     : ds.target,
            "target_names": ds.target_names,
            "n_samples"   : n,
            "n_people"    : len(ds.target_names),
            "image_shape" : (h, w),
            "pixel_size"  : h * w,
            "source"      : f"LFW — {n} foto",
            "description" : f"LFW: {n} foto dari {len(ds.target_names)} orang. Ukuran {h}x{w} px.",
        }
    except Exception:
        return None


def load_custom_selfie_dataset(base_path: str, target_size: Tuple[int, int] = (64, 64)) -> Optional[Dict]:
    import os
    import cv2
    from .face_utils import preprocess_face
    
    if not os.path.exists(base_path):
        return None
        
    images_list = []
    targets = []
    target_names = []
    person_id = 0
    
    for entry in sorted(os.listdir(base_path)):
        person_dir = os.path.join(base_path, entry)
        if not os.path.isdir(person_dir):
            continue
            
        target_names.append(entry)
        
        for sub in ['docs', 'selfies']:
            sub_dir = os.path.join(person_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
                
            for fname in os.listdir(sub_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    img_path = os.path.join(sub_dir, fname)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        proc_face, info = preprocess_face(img, detect=True)
                        if proc_face is not None:
                            resized = cv2.resize(proc_face, target_size)
                            images_list.append(resized.flatten())
                            targets.append(person_id)
        
        person_id += 1
        
    if not images_list:
        return None
        
    images = np.array(images_list)
    images_2d = images.reshape(-1, target_size[0], target_size[1])
    n = images.shape[0]
    
    return {
        "images"      : images,
        "images_2d"   : images_2d,
        "targets"     : np.array(targets),
        "target_names": target_names,
        "n_samples"   : n,
        "n_people"    : len(target_names),
        "image_shape" : target_size,
        "pixel_size"  : target_size[0] * target_size[1],
        "source"      : f"Selfie & ID ({n} foto, {len(target_names)} orang)",
        "description" : f"Custom Dataset Selfie & ID lokal: {n} foto dari {len(target_names)} orang (docs & selfies). Ukuran {target_size[0]}x{target_size[1]} px.",
    }


def load_pretrained_eigenspace(filepath: str) -> Optional[Dict]:
    import os
    if not os.path.exists(filepath):
        return None
    try:
        data = np.load(filepath)
        n = data['n_samples'].item() if data['n_samples'].ndim == 0 else data['n_samples'][0]
        k = data['k_components'].item() if data['k_components'].ndim == 0 else data['k_components'][0]
        shape = tuple(data['image_shape'])
        
        return {
            "mean_face": data['mean_face'],
            "eigenfaces": data['eigenfaces'],
            "singular_values": data['singular_values'],
            "explained_variance_pct": data['explained_variance_pct'],
            "n_samples": n,
            "n_components": k,
            "image_shape": shape,
            "source": f"Pretrained Colab Model ({n} foto)",
            "description": f"Model dilatih di Colab. Menggunakan {k} Eigenfaces."
        }
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
    images        = images.astype(float)
    mean_face     = compute_mean_face(images)
    centered      = images - mean_face
    n_components  = min(n_components, images.shape[0], images.shape[1])

    if use_svd:
        U, S, Vt    = np.linalg.svd(centered, full_matrices=False)
        eigenfaces  = Vt[:n_components]
        eigenvalues = (S[:n_components] ** 2) / images.shape[0]
        ev_ratio    = (S ** 2) / np.sum(S ** 2)
        singular_values = S
    else:
        C = compute_covariance_matrix(centered)
        vals, vecs  = np.linalg.eig(C)
        idx         = np.argsort(vals)[::-1]
        vals, vecs  = vals[idx].real, vecs[:, idx].real
        eigenfaces  = vecs[:, :n_components].T
        eigenvalues = vals[:n_components]
        ev_ratio    = vals / np.sum(vals)
        singular_values = None

    return {
        "eigenfaces"              : eigenfaces,
        "eigenvalues"             : eigenvalues,
        "mean_face"               : mean_face,
        "explained_variance_ratio": ev_ratio[:n_components],
        "n_components"            : n_components,
        "singular_values"         : singular_values,
        "n_training_images"       : images.shape[0],
    }


def build_eigenspace_from_dataset(
    dataset_images: np.ndarray,
    n_components: int = 50,
    target_size: Tuple[int, int] = (64, 64),
) -> dict:
    ef = compute_eigenfaces(dataset_images, n_components=n_components)
    cumvar = np.cumsum(ef["explained_variance_ratio"])
    ef.update({
        "target_size"            : target_size,
        "k_for_95pct_variance"   : int(np.searchsorted(cumvar, 0.95)) + 1,
        "k_for_99pct_variance"   : int(np.searchsorted(cumvar, 0.99)) + 1,
        "total_variance_captured": float(np.sum(ef["explained_variance_ratio"])),
        "dataset_size"           : dataset_images.shape[0],
    })
    return ef


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
        "face1_flat": f1, "face2_flat": f2,
        "eigenface_data": ef,
        "weights_face1": w1, "weights_face2": w2,
        "reconstructed_face1": r1, "reconstructed_face2": r2,
        "svd_face1": {"U": U1, "S": S1, "Vt": Vt1},
        "svd_face2": {"U": U2, "S": S2, "Vt": Vt2},
    }


def analyze_two_faces_with_dataset(
    face1: np.ndarray,
    face2: np.ndarray,
    eigenspace: dict,
) -> dict:
    ef          = eigenspace["eigenfaces"]
    mean_face   = eigenspace["mean_face"]
    target_size = eigenspace.get("target_size", (64, 64))

    face1_r = resize_face_for_eigenspace(face1, target_size)
    face2_r = resize_face_for_eigenspace(face2, target_size)
    f1, f2  = face1_r.flatten(), face2_r.flatten()

    w1 = project_to_eigenspace(f1, ef, mean_face)
    w2 = project_to_eigenspace(f2, ef, mean_face)
    r1 = reconstruct_from_eigenspace(w1, ef, mean_face).reshape(target_size)
    r2 = reconstruct_from_eigenspace(w2, ef, mean_face).reshape(target_size)

    U1, S1, Vt1 = svd_decompose(face1_r)
    U2, S2, Vt2 = svd_decompose(face2_r)

    return {
        "face1_resized"      : face1_r,
        "face2_resized"      : face2_r,
        "face1_flat"         : f1,
        "face2_flat"         : f2,
        "weights_face1"      : w1,
        "weights_face2"      : w2,
        "reconstructed_face1": r1,
        "reconstructed_face2": r2,
        "svd_face1"          : {"U": U1, "S": S1, "Vt": Vt1},
        "svd_face2"          : {"U": U2, "S": S2, "Vt": Vt2},
        "n_components_used"  : len(w1),
        "eigenspace_info"    : {
            "dataset_size"           : eigenspace.get("dataset_size"),
            "n_components"           : eigenspace.get("n_components"),
            "total_variance_captured": eigenspace.get("total_variance_captured", 0),
            "k_for_95pct"            : eigenspace.get("k_for_95pct_variance"),
        },
    }
