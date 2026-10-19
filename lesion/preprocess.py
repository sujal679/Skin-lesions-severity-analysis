"""Image preprocessing for dermoscopy images: resizing, hair removal, vignette masking, color constancy."""
import cv2
import numpy as np

TARGET_SIZE = 512


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize(img, size=TARGET_SIZE):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def shades_of_gray(img, p=6):
    """Shades-of-Gray color constancy (Finlayson 2004), commonly used on ISIC data."""
    f = img.astype(np.float32)
    illum = np.power(np.mean(np.power(f, p), axis=(0, 1)), 1.0 / p)
    illum = illum / (np.linalg.norm(illum) + 1e-6)
    f = f / (illum * np.sqrt(3) + 1e-6)
    return np.clip(f, 0, 255).astype(np.uint8)


def remove_hair(img, kernel_size=17, threshold=10):
    """DullRazor-style hair removal: black-hat morphology finds dark thin hairs, then inpaint them."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(img, hair_mask, 3, cv2.INPAINT_TELEA), hair_mask


def vignette_mask(img, dark_thresh=25):
    """Mask of the valid field of view (excludes black circular dermatoscope borders)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray > dark_thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    return mask.astype(bool)


def preprocess(img):
    img = resize(img)
    img = shades_of_gray(img)
    clean, hair_mask = remove_hair(img)
    clean = cv2.GaussianBlur(clean, (5, 5), 0)
    return clean, hair_mask, vignette_mask(clean)
