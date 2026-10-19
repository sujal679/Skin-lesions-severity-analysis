"""Lesion segmentation: Otsu thresholding on a lesion-contrast channel + morphological cleanup."""
import cv2
import numpy as np
from scipy import ndimage as ndi


def lesion_channel(img, fov_mask=None):
    """Colour distance (CIELab) of each pixel from this image's own skin colour.

    Skin is estimated from a band around the image edge, where lesions rarely are. Pixels lighter
    than the skin (glare, scars, white structures) count less, since pigment makes skin darker.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    band = np.ones((h, w), bool)
    bh, bw = int(0.12 * h), int(0.12 * w)
    band[bh:h - bh, bw:w - bw] = False
    if fov_mask is not None and (band & fov_mask).sum() > 100:
        band &= fov_mask
    skin = np.median(lab[band], axis=0)
    d = lab - skin
    dL = np.where(d[..., 0] < 0, d[..., 0], 0.3 * d[..., 0])
    ch = np.sqrt(dL ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    return cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def segment(img, fov_mask=None):
    """Otsu gives a reliable but tight core; GrabCut then grows it into lighter peripheral pigment."""
    core = _otsu_segment(img, fov_mask)
    if not core.any():
        return core
    refined = _grabcut_refine(img, core, fov_mask)
    # Fall back to the Otsu mask if GrabCut collapses or leaks into the surrounding skin.
    ratio = refined.sum() / core.sum()
    return refined if 0.8 <= ratio <= 3.0 else core


def _grabcut_refine(img, core, fov_mask, iters=5):
    h, w = core.shape
    r = max(5, int(0.04 * max(h, w)))
    disk = lambda s: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * s + 1, 2 * s + 1))
    core_u8 = core.astype(np.uint8)

    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(core_u8, disk(4 * r)) > 0] = cv2.GC_PR_BGD   # wide band where the lesion may extend
    gc[core] = cv2.GC_PR_FGD
    gc[cv2.erode(core_u8, disk(r)) > 0] = cv2.GC_FGD         # dark centre is certainly lesion
    if fov_mask is not None:
        gc[~fov_mask] = cv2.GC_BGD                            # dermatoscope vignette is never lesion

    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return core
    mask = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(3))

    # Keep the component that contains the core.
    n, labels = cv2.connectedComponents(mask)
    if n <= 1:
        return core
    best = np.bincount(labels[core & (labels > 0)], minlength=n).argmax()
    return ndi.binary_fill_holes(labels == best) if best else core


def _otsu_segment(img, fov_mask=None):
    ch = lesion_channel(img, fov_mask)
    valid = fov_mask if fov_mask is not None else np.ones(ch.shape, bool)
    t, _ = cv2.threshold(ch[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((ch > t) & valid).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # Keep the component that is largest and closest to the image centre (lesions are usually centred).
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros_like(mask, bool)
    h, w = mask.shape
    centre = np.array([w / 2, h / 2])
    diag = np.hypot(h, w)
    edge = np.zeros((h, w), bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    edge_labels = np.bincount(labels[edge], minlength=n)
    best, best_score = 0, -1
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        dist = np.linalg.norm(cents[i] - centre) / diag
        # Penalise in proportion to how much of the image edge the region runs along: a lesion can
        # be cut off by the frame, but a region hugging long stretches of edge is usually skin.
        edge_frac = edge_labels[i] / edge.sum()
        score = area * (1 - dist) ** 2 * (1 - min(0.9, 4 * edge_frac))
        if score > best_score:
            best, best_score = i, score
    return ndi.binary_fill_holes(labels == best)
