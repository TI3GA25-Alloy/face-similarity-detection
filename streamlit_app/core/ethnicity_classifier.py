import numpy as np

def predict_ethnicity(face_image: np.ndarray, manual_prob: float = None) -> float:
    """
    Memprediksi probabilitas wajah adalah Asian / South East Asian.
    Menerima input gambar wajah (numpy array) dan mengembalikan float (0.0 sampai 1.0).
    
    Catatan:
    Untuk sementara, karena kita belum melatih SVM+LBP atau menggunakan DeepFace,
    kita menyediakan opsi manual_prob untuk diatur dari Streamlit UI (slider).
    
    Jika manual_prob diberikan, kembalikan manual_prob.
    Jika tidak, asumsikan 0.5 (netral).
    """
    if manual_prob is not None:
        return manual_prob
    
    return 0.5

def get_hybrid_aging_vector(prob_asian: float, vector_aaf: np.ndarray, vector_fgnet: np.ndarray) -> np.ndarray:
    """
    Menggabungkan vektor penuaan dari dua dataset berdasarkan probabilitas etnis.
    Rumus: V_hybrid = (prob_asian * V_aaf) + ((1 - prob_asian) * V_fgnet)
    """
    prob_caucasian = 1.0 - prob_asian
    
    # Pastikan shape sama
    if vector_aaf is not None and vector_fgnet is not None:
        return (prob_asian * vector_aaf) + (prob_caucasian * vector_fgnet)
        
    # Fallback jika salah satu vektor kosong
    if vector_aaf is not None:
        return vector_aaf
    if vector_fgnet is not None:
        return vector_fgnet
        
    return None
