"""Severity grading with the Stolz ABCD rule (Total Dermoscopy Score) and optional ML fusion."""

TDS_WEIGHTS = {"A": 1.3, "B": 0.1, "C": 0.5, "D": 0.5}
# Stolz et al. cut-offs for clinician-scored TDS. Automated scores run lower, so train.py recalibrates
# these on the dataset and stores them with the model; these are only used when no model exists.
STOLZ_CUTOFFS = (4.75, 5.45)
LEVELS = ["Low", "Moderate", "High"]
ADVICE = {
    "Low": "Benign-appearing lesion. Routine self-monitoring.",
    "Moderate": "Suspicious lesion. Recommend dermatologist review / short-term follow-up.",
    "High": "Highly suspicious for melanoma. Recommend prompt specialist evaluation / biopsy.",
}


def tds(feats):
    return sum(TDS_WEIGHTS[k] * feats[k] for k in TDS_WEIGHTS)


def grade(score, cutoffs=STOLZ_CUTOFFS):
    """Moderate at or above cutoffs[0], High at or above cutoffs[1]."""
    return LEVELS[sum(score >= c for c in cutoffs)]


def severity(feats, malignancy_prob=None, threshold=0.5, cutoffs=STOLZ_CUTOFFS):
    """Combine rule-based TDS with an ML malignancy probability (if a model is available).

    The TDS is mapped to a 0-1 risk (TDS range 1.0-8.9) and averaged with the model probability.
    `threshold` is the screening cutoff tuned in train.py: at or above it the lesion is at least Moderate.
    """
    score = tds(feats)
    tds_risk = min(max((score - 1.0) / 7.9, 0), 1)
    risk = tds_risk if malignancy_prob is None else 0.5 * tds_risk + 0.5 * malignancy_prob
    level = grade(score, cutoffs)
    if malignancy_prob is not None:
        if malignancy_prob >= threshold and level == "Low":
            level = "Moderate"
        # Escalate one level if the model is confident the lesion is malignant.
        if malignancy_prob > 0.7 and level != "High":
            level = LEVELS[LEVELS.index(level) + 1]
    return {"tds": round(score, 2), "risk_index": round(risk, 3), "level": level, "advice": ADVICE[level]}
