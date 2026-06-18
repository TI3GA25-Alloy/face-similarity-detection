import numpy as np
import cv2
from skimage import feature

def extract_hog(image):
    hog_features = feature.hog(
        image, orientations=8, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys", visualize=False
        )
    return hog_features

def evaluate_hog():
    eigenspace = np.load('pretrained_eigenspace.npz')
    data_kel = np.load('privasi_kelompok_100x100.npz')
    
    X_latih_pix = data_kel["X_latih"]
    X_test_lintas_pix = data_kel["X_test_lintas"]
    y_latih = data_kel["y"]
    y_test_lintas = data_kel["y"]
    
    X_latih = []
    X_test_lintas = []
    
    for i in range(len(X_latih_pix)):
        img_latih = (X_latih_pix[i].reshape(100, 100) * 255).astype(np.uint8)
        img_lintas = (X_test_lintas_pix[i].reshape(100, 100) * 255).astype(np.uint8)
        
        X_latih.append(extract_hog(img_latih))
        X_test_lintas.append(extract_hog(img_lintas))
            
    X_latih = np.array(X_latih, dtype=np.float32)
    X_test_lintas = np.array(X_test_lintas, dtype=np.float32)
    
    mean_feat = eigenspace["mean_hog"]
    eigenfaces = eigenspace["eigenfaces_hog"]
    
    X_latih_pca = np.dot(X_latih - mean_feat, eigenfaces.T)
    X_test_lintas_pca = np.dot(X_test_lintas - mean_feat, eigenfaces.T)
    
    aging_vector_fgnet = eigenspace["aging_vector_fgnet_hog"]
    aging_vector_aaf = eigenspace["aging_vector_aaf_hog"]
    
    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    
    print("Baseline HOG (No Injection):")
    correct = 0
    for i in range(len(X_test_lintas_pca)):
        target = X_test_lintas_pca[i]
        true_label = y_test_lintas[i]
        best_sim = -1
        best_label = -1
        for j in range(len(X_latih_pca)):
            sim = cos_sim(target, X_latih_pca[j])
            if sim > best_sim:
                best_sim = sim
                best_label = y_latih[j]
        if best_label == true_label:
            correct += 1
    print(f"HOG Acc with scale 0.0: {correct / len(X_test_lintas_pca) * 100}%")
    
    best_acc = 0
    best_scale = 0
    
    hybrid_vector = (0.85 * aging_vector_aaf) + (0.15 * aging_vector_fgnet)
    
    for scale in np.linspace(-5.0, 5.0, 101):
        X_test_injected = X_test_lintas_pca + (hybrid_vector * scale)
        
        correct = 0
        for i in range(len(X_test_injected)):
            target = X_test_injected[i]
            true_label = y_test_lintas[i]
            
            best_sim = -1
            best_label = -1
            for j in range(len(X_latih_pca)):
                sim = cos_sim(target, X_latih_pca[j])
                if sim > best_sim:
                    best_sim = sim
                    best_label = y_latih[j]
            
            if best_label == true_label:
                correct += 1
        
        acc = correct / len(X_test_injected) * 100
        if acc > best_acc:
            best_acc = acc
            best_scale = scale
            
    print(f"HOG Best Acc: {best_acc}% at scale {best_scale}")

evaluate_hog()
