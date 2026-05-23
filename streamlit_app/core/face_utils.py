import numpy as np
import cv2
from PIL import Image
import io
from typing import Tuple, Optional


TARGET_SIZE  = (128, 128)
HAAR_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_PROFILE = cv2.data.haarcascades + "haarcascade_profileface.xml"


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)


def load_image_from_pil(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)


def detect_face(gray_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    for cascade_path in [HAAR_FRONTAL, HAAR_PROFILE]:
        cascade = cv2.CascadeClassifier(cascade_path)
        faces   = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            return tuple(sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0])
    return None


def crop_face(gray_image: np.ndarray, bbox: Tuple[int, int, int, int], padding: float = 0.2) -> np.ndarray:
    x, y, w, h = bbox
    H, W = gray_image.shape
    x1 = max(0, x - int(w * padding))
    y1 = max(0, y - int(h * padding))
    x2 = min(W, x + w + int(w * padding))
    y2 = min(H, y + h + int(h * padding))
    return gray_image[y1:y2, x1:x2]


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_elliptical_mask(image: np.ndarray) -> np.ndarray:
    """
    Applies an elliptical mask to remove background, hair, and ear edges.
    Forces the PCA model to focus purely on internal facial features.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.35), int(h * 0.45))
    
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)


def preprocess_face(
    gray_image: np.ndarray,
    detect: bool = True,
    target_size: Tuple[int, int] = TARGET_SIZE,
    blur: bool = False,
    angle: float = 0.0,
    pre_bbox: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline: Detection, Rotation, CLAHE, Resize, Sobel, and Elliptical Masking.
    """
    info = {"face_detected": False, "bbox": None}
    face_crop = gray_image

    if detect:
        bbox = pre_bbox if pre_bbox is not None else detect_face(gray_image)
        if bbox is not None:
            info["face_detected"] = True
            info["bbox"] = bbox
            face_crop = crop_face(gray_image, bbox)
            
    if angle != 0.0:
        h, w = face_crop.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        face_crop = cv2.warpAffine(face_crop, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_crop = clahe.apply(face_crop)

    if blur:
        face_crop = apply_gaussian_blur(face_crop)

    resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
    
    masked_gray = apply_elliptical_mask(resized)
    
    normalized = masked_gray.astype(np.float64) / 255.0
    return normalized, info


def draw_face_box(original_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    display = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB) if original_image.ndim == 2 else original_image.copy()
    x, y, w, h = bbox
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 100), 2)
    return display
