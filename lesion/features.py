"""Feature extraction: ABCD dermoscopy rule features plus colour and texture descriptors."""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Reference colours from the ABCD rule (RGB), as used in computer-aided ABCD literature.
DERM_COLOURS = {
    "white": (220, 220, 220),
    "red": (180, 40, 40),
    "light_brown": (190, 120, 80),
    "dark_brown": (90, 50, 30),
    "blue_gray": (80, 110, 140),
    "black": (30, 25, 25),
}


def _principal_axes_flip(mask):
    """Rotate the lesion so its principal axis is horizontal, centred on its centroid."""
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    coords = np.stack([xs - cx, ys - cy])
    evals, evecs = np.linalg.eigh(np.cov(coords))
    angle = np.degrees(np.arctan2(evecs[1, -1], evecs[0, -1]))
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += w / 2 - cx
    M[1, 2] += h / 2 - cy
    return M


def asymmetry(mask, img):
    """Shape + colour asymmetry about the two principal axes. Returns (score 0-2, raw ratios)."""
    h, w = mask.shape
    M = _principal_axes_flip(mask)
    m = cv2.warpAffine(mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
    im = cv2.warpAffine(img, M, (w, h))
    area = m.sum()
    ratios, colour_diffs = [], []
    for flip in (0, 1):  # 0: vertical flip (horizontal axis), 1: horizontal flip (vertical axis)
        mf = cv2.flip(m.astype(np.uint8), flip).astype(bool)
        ratios.append(np.logical_xor(m, mf).sum() / (area + 1e-6))
        imf = cv2.flip(im, flip)
        both = m & mf
        if both.any():
            colour_diffs.append(np.abs(im[both].astype(float) - imf[both].astype(float)).mean() / 255)
        else:
            colour_diffs.append(0)
    # An axis is "asymmetric" if shape overlap error > 12% or mean colour difference > 8%.
    score = sum(int(r > 0.12 or c > 0.08) for r, c in zip(ratios, colour_diffs))
    return score, ratios, colour_diffs


# Fraction of the lesion-skin contrast that must fall inside the thin edge band for a segment to count
# as abrupt (see border()). Calibrated on HAM10000 training images: separates melanoma (mean B 4.4)
# from nevi (mean B 2.3) best, AUC 0.72.
ABRUPT_THRESHOLD = 0.20


def border(mask, img):
    """Border score per the Stolz ABCD rule: segments (of 8) with an abrupt cut-off of pigment.

    For each 45-degree segment, compare lightness just inside vs just outside the border (a band of
    a few pixels) with the lesion's full contrast (lesion core vs surrounding skin). If a large share
    of the contrast happens within that thin band, the pigment stops abruptly there.
    Also returns shape descriptors (compactness, solidity, segments with an irregular outline).
    """
    m8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    compactness = perim ** 2 / (4 * np.pi * area + 1e-6)  # 1.0 for a circle
    hull = cv2.convexHull(c)
    solidity = area / (cv2.contourArea(hull) + 1e-6)

    M = cv2.moments(c)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    seg = ((np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi) * 8).astype(int) % 8

    disk = lambda s: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * s + 1, 2 * s + 1))
    band = max(3, int(0.01 * max(h, w)))
    inner = mask & ~cv2.erode(m8, disk(band)).astype(bool)
    outer = cv2.dilate(m8, disk(band)).astype(bool) & ~mask
    skin = cv2.dilate(m8, disk(8 * band)).astype(bool) & ~cv2.dilate(m8, disk(4 * band)).astype(bool)
    core = cv2.erode(m8, disk(4 * band)).astype(bool)
    if not core.any():
        core = mask
    L = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0].astype(float)

    abrupt, shape_segs, sharpness = 0, 0, []
    pts = c[:, 0, :].astype(float)
    pang = ((np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    rad = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    for s in range(8):
        sel = seg == s
        i, o, sk, co = inner & sel, outer & sel, skin & sel, core & sel
        if i.any() and o.any() and sk.any():
            total = L[sk].mean() - (L[co].mean() if co.any() else L[core].mean())
            ratio = (L[o].mean() - L[i].mean()) / total if total > 5 else 0.0
            sharpness.append(ratio)
            abrupt += ratio > ABRUPT_THRESHOLD
        r = rad[pang == s]
        shape_segs += len(r) > 5 and r.std() / (r.mean() + 1e-6) > 0.10
    mean_sharp = float(np.mean(sharpness)) if sharpness else 0.0
    return int(abrupt), compactness, solidity, perim, area, mean_sharp, int(shape_segs)


def colour(img, mask):
    """Count of dermoscopic colours present (1-6) plus colour statistics."""
    px = img[mask].astype(float)
    names = list(DERM_COLOURS)
    refs = np.array([DERM_COLOURS[n] for n in names], float)
    d = np.linalg.norm(px[:, None, :] - refs[None], axis=2)
    nearest = d.argmin(1)
    frac = np.bincount(nearest, minlength=len(names)) / len(px)
    present = {n: float(f) for n, f in zip(names, frac)}
    count = int(max(1, (frac > 0.05).sum()))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[mask].astype(float)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[mask].astype(float)
    stats = {}
    for space, arr, chans in (("rgb", px, "RGB"), ("hsv", hsv, "HSV"), ("lab", lab, "Lab")):
        for i, ch in enumerate(chans):
            stats[f"{space}_{ch}_mean"] = arr[:, i].mean()
            stats[f"{space}_{ch}_std"] = arr[:, i].std()
    return count, present, stats


def structures(img, mask):
    """Differential-structure proxy (0-5): heterogeneity of local texture inside the lesion.

    True dermoscopic structures (network, dots, globules, streaks, structureless areas) need a
    trained detector; here we estimate structural diversity from texture statistics.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ys, xs = np.nonzero(mask)
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    cmask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop = np.where(cmask, crop, 0).astype(np.uint8)

    glcm = graycomatrix(crop, [1, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 256, symmetric=True, normed=True)
    tex = {f"glcm_{p}": float(graycoprops(glcm, p).mean())
           for p in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")}

    lbp = local_binary_pattern(crop, 8, 1, "uniform")
    hist, _ = np.histogram(lbp[cmask], bins=10, range=(0, 10), density=True)
    for i, v in enumerate(hist):
        tex[f"lbp_{i}"] = float(v)
    lbp_entropy = -np.sum(hist * np.log2(hist + 1e-12))

    # Blob-like dots/globules via LoG-style detection (dark spots inside lesion)
    blobs = cv2.SimpleBlobDetector_create()
    n_dots = len([k for k in blobs.detect(crop) if cmask[int(k.pt[1]), int(k.pt[0])]])
    tex["n_dots"] = n_dots

    score = 0
    score += tex["glcm_contrast"] > 60          # pigment network / strong local contrast
    score += n_dots >= 3                         # dots / globules
    score += lbp_entropy > 3.0                   # structural heterogeneity
    score += tex["glcm_homogeneity"] < 0.35      # branched streaks / irregular texture
    score += gray[mask].std() > 30               # structureless + structured regions mixed
    return int(score), tex


def extract(img, mask, px_per_mm=None):
    A, sym_ratios, sym_colour = asymmetry(mask, img)
    B, compact, solidity, perim, area, edge_sharpness, shape_segs = border(mask, img)
    C, colours_present, colour_stats = colour(img, mask)
    D, texture = structures(img, mask)
    eq_diam_px = np.sqrt(4 * area / np.pi)
    feats = {
        "A": A, "B": B, "C": C, "D": D,
        "asym_ratio_1": sym_ratios[0], "asym_ratio_2": sym_ratios[1],
        "asym_colour_1": sym_colour[0], "asym_colour_2": sym_colour[1],
        "compactness": compact, "solidity": solidity,
        "edge_sharpness": edge_sharpness, "border_shape_segments": shape_segs,
        "area_frac": area / mask.size, "eq_diameter_px": eq_diam_px,
        **{f"colour_{k}": v for k, v in colours_present.items()},
        **colour_stats, **texture,
    }
    if px_per_mm:
        feats["diameter_mm"] = eq_diam_px / px_per_mm
    return feats
