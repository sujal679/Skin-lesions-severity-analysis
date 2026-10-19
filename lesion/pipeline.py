"""End-to-end analysis: preprocess -> segment -> features -> classify -> severity."""
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from . import features, preprocess, segment, severity

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "lesion_rf.joblib"


class SegmentationError(RuntimeError):
    pass


def load_model(path=MODEL_PATH):
    return joblib.load(path) if Path(path).exists() else None


def analyze(img_rgb, model=None, px_per_mm=None):
    clean, hair_mask, fov = preprocess.preprocess(img_rgb)
    mask = segment.segment(clean, fov)
    if mask.sum() < 0.005 * mask.size:
        raise SegmentationError("No lesion found - check the image is a dermoscopy image.")
    feats = features.extract(clean, mask, px_per_mm)

    prediction = None
    if model is not None:
        X = pd.DataFrame([feats]).reindex(columns=model["columns"], fill_value=0)
        proba = model["clf"].predict_proba(X)[0]
        classes = list(model["clf"].classes_)
        mal_p = float(sum(p for c, p in zip(classes, proba) if c in model["malignant"]))
        threshold = model.get("threshold", 0.5)
        flagged = mal_p >= threshold
        top = classes[int(np.argmax(proba))]
        if classes == ["benign", "malignant"]:
            top = "malignant" if flagged else "benign"
        prediction = {
            "class": top,
            "probabilities": dict(zip(classes, map(float, proba))),
            "malignancy_prob": mal_p,
            "threshold": threshold,
            "flagged_malignant": bool(flagged),
        }
    if prediction:
        sev = severity.severity(feats, prediction["malignancy_prob"], prediction["threshold"],
                                model.get("tds_cutoffs", severity.STOLZ_CUTOFFS))
    else:
        sev = severity.severity(feats)
    return {"image": clean, "mask": mask, "hair_mask": hair_mask,
            "features": feats, "prediction": prediction, "severity": sev}


def overlay(img, mask, colour=(0, 255, 0)):
    out = img.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, colour, 2)
    return out
