"""Generate result figures and a metrics summary for the held-out test set.

Requires data/features.csv and models/lesion_rf.joblib (run train.py --binary first).
Usage: python evaluate.py            -> writes results/*.png and results/metrics.md
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from lesion import pipeline, preprocess, severity
from train import MALIGNANT, split

OUT = Path("results")
DX_NAMES = {"mel": "Melanoma", "nv": "Melanocytic nevus", "bcc": "Basal cell carcinoma",
            "akiec": "Actinic keratosis", "bkl": "Benign keratosis", "df": "Dermatofibroma",
            "vasc": "Vascular lesion"}

FEATURE_NAMES = {
    "edge_sharpness": "Border edge sharpness", "B": "Border score (B)", "solidity": "Shape solidity",
    "compactness": "Shape compactness", "asym_ratio_1": "Shape asymmetry, axis 1",
    "asym_ratio_2": "Shape asymmetry, axis 2", "asym_colour_1": "Colour asymmetry, axis 1",
    "asym_colour_2": "Colour asymmetry, axis 2", "lab_a_mean": "Redness mean (Lab a*)",
    "lab_a_std": "Redness variation (Lab a*)", "lab_b_mean": "Yellow-blue mean (Lab b*)",
    "lab_b_std": "Yellow-blue variation (Lab b*)", "lab_L_mean": "Lightness mean (Lab L*)",
    "lab_L_std": "Lightness variation (Lab L*)", "hsv_S_mean": "Saturation mean",
    "hsv_V_std": "Brightness variation", "glcm_ASM": "Texture uniformity (GLCM ASM)",
    "glcm_energy": "Texture energy (GLCM)", "glcm_contrast": "Texture contrast (GLCM)",
    "glcm_homogeneity": "Texture homogeneity (GLCM)", "glcm_correlation": "Texture correlation (GLCM)",
    "A": "Asymmetry score (A)", "C": "Colour count (C)", "D": "Structures score (D)",
}

# Colours: categorical slots 1-2, sequential blue ramp, fixed status palette, chart chrome.
SERIES = ["#2a78d6", "#eb6834"]
STATUS = {"Low": "#0ca30c", "Moderate": "#fab219", "High": "#d03b3b"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("seq", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load():
    feats = pd.read_csv("data/features.csv")
    model = pipeline.load_model()
    if model is None:
        raise SystemExit("No model found - run: python train.py --binary")
    tr, te = split(feats)
    X = feats.reindex(columns=model["columns"], fill_value=0).fillna(0)
    mal = feats.dx.isin(MALIGNANT).values
    proba = model["clf"].predict_proba(X.iloc[te])
    mal_p = proba[:, [c in model["malignant"] for c in model["clf"].classes_]].sum(1)
    tds = X.apply(severity.tds, axis=1).values
    return feats, model, X, tr, te, mal, mal_p, tds


def fig_roc(y, mal_p, tds_te, threshold):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=AXIS, zorder=1)
    ax.text(0.62, 0.55, "chance", color=MUTED, rotation=41, fontsize=9)
    for score, label, colour in ((mal_p, "Random Forest classifier", SERIES[0]),
                                 (tds_te, "ABCD rule (TDS) alone", SERIES[1])):
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, lw=2, color=colour, label=f"{label}  (AUC {roc_auc_score(y, score):.3f})", zorder=3)
    for t, name in ((0.5, "default cutoff 0.50"), (threshold, f"tuned cutoff {threshold:.2f}")):
        pred = mal_p >= t
        fpr_t, tpr_t = pred[~y].mean(), pred[y].mean()
        ax.scatter(fpr_t, tpr_t, s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=4)
        # Labels sit in empty space: below-right of the default point, above-left of the tuned point.
        xy_text = (fpr_t + 0.06, tpr_t - 0.07) if t == 0.5 else (0.03, 0.9)
        ax.annotate(f"{name}\n{tpr_t:.0%} caught, {fpr_t:.0%} false alarms", (fpr_t, tpr_t),
                    xytext=xy_text, fontsize=9, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate (benign flagged)",
           ylabel="True positive rate (malignant caught)")
    ax.set_title("ROC curve: malignant vs benign, held-out test set")
    ax.legend(loc="lower right")
    fig.savefig(OUT / "roc_curve.png")
    plt.close(fig)


def fig_confusion(y, mal_p, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    labels = ["Benign", "Malignant"]
    for ax, (t, title) in zip(axes, ((0.5, "Default cutoff (0.50)"), (threshold, f"Tuned cutoff ({threshold:.2f})"))):
        cm = confusion_matrix(y, mal_p >= t)
        pct = cm / cm.sum(1, keepdims=True)
        ax.imshow(pct, cmap=BLUES, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j]:.0%} of row", ha="center", va="center",
                        fontsize=11, color="#ffffff" if pct[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Confusion matrices, held-out test set (n = {:,})".format(len(y)),
                 x=0.02, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png")
    plt.close(fig)


def severity_table(X, te, y, mal_p, threshold, cutoffs):
    rows = []
    for name, cuts in (("Stolz", severity.STOLZ_CUTOFFS), ("Recalibrated", cutoffs)):
        levels = np.array([severity.severity(X.iloc[i], p, threshold, cuts)["level"] for i, p in zip(te, mal_p)])
        for cls, sel in (("Malignant", y), ("Benign", ~y)):
            rows.append({"group": f"{cls}, {name}", **{l: (levels[sel] == l).mean() for l in severity.LEVELS}})
    return pd.DataFrame(rows).set_index("group")


def fig_severity(table):
    fig, ax = plt.subplots(figsize=(9, 3.9))
    order = ["Malignant, Stolz", "Malignant, Recalibrated", "Benign, Stolz", "Benign, Recalibrated"]
    ypos = [3.3, 2.3, 1, 0]
    gap = 0.004  # surface gap between stacked segments
    for yp, g in zip(ypos, order):
        left = 0
        for lvl in severity.LEVELS:
            v = table.loc[g, lvl]
            ax.barh(yp, v - gap, left=left, height=0.62, color=STATUS[lvl])
            if v >= 0.06:
                ax.text(left + v / 2, yp, f"{v:.0%}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if lvl != "Moderate" else INK, fontweight="bold")
            left += v
    ax.set_yticks(ypos, ["Stolz cut-offs", "Recalibrated", "Stolz cut-offs", "Recalibrated"])
    ax.text(-0.01, 3.95, "Malignant lesions", transform=ax.get_yaxis_transform(), ha="right",
            fontweight="bold", color=INK)
    ax.text(-0.01, 1.65, "Benign lesions", transform=ax.get_yaxis_transform(), ha="right",
            fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS[l]) for l in severity.LEVELS]
    ax.legend(handles, [f"{l} severity" for l in severity.LEVELS], ncol=3, loc="upper left",
              bbox_to_anchor=(0, 1.16))
    ax.set_title("Final severity level (TDS + classifier), held-out test set", pad=34)
    fig.savefig(OUT / "severity_levels.png")
    plt.close(fig)


def fig_tds(tds_te, y, cutoffs):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.arange(np.floor(tds_te.min()) - 0.05, tds_te.max() + 0.25, 0.2)
    for sel, label, colour in ((~y, "Benign", SERIES[0]), (y, "Malignant", SERIES[1])):
        ax.hist(tds_te[sel], bins=bins, density=True, histtype="step", lw=2, color=colour, label=label)
    # Cut-off labels sit above the plot, recalibrated and Stolz on separate rows so close values don't collide.
    trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x, name, ls, yl in ((cutoffs[0], "Moderate", "-", 1.02), (cutoffs[1], "High", "-", 1.02),
                            (severity.STOLZ_CUTOFFS[0], "Stolz", ":", 1.09),
                            (severity.STOLZ_CUTOFFS[1], "Stolz", ":", 1.09)):
        ax.axvline(x, color=INK2, lw=1, ls=ls, zorder=0)
        ax.text(x, yl, f"{name} {x:g}", transform=trans, fontsize=8.5, color=INK2, ha="center", va="bottom")
    ax.set(xlabel="Total Dermoscopy Score (TDS)", ylabel="Share of lesions (density)")
    ax.set_title("TDS distribution with recalibrated (solid) and Stolz (dotted) cut-offs", pad=34)
    ax.legend(loc="upper left")
    fig.savefig(OUT / "tds_distribution.png")
    plt.close(fig)


def fig_importance(model):
    imp = pd.Series(model["clf"].feature_importances_, model["columns"]).sort_values().tail(15)
    imp.index = [FEATURE_NAMES.get(n, n) for n in imp.index]
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.barh(imp.index, imp.values, height=0.68, color=SERIES[0])
    for i, v in enumerate(imp.values):
        ax.text(v + 0.0008, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Top 15 features, Random Forest")
    fig.savefig(OUT / "feature_importance.png")
    plt.close(fig)


def fig_segmentation(feats, te, model):
    """Sample held-out images: two melanomas, two nevi, one BCC, one benign keratosis."""
    test = feats.iloc[te]
    picks = pd.concat([test[test.dx == d].sample(n, random_state=5)
                       for d, n in (("mel", 2), ("nv", 2), ("bcc", 1), ("bkl", 1))])
    index = {p.stem: p for p in Path("data").rglob("*.jpg")}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    for ax, (_, row) in zip(axes.flat, picks.iterrows()):
        original = preprocess.load_image(index[row.image_id])
        res = pipeline.analyze(original, model)
        vis = preprocess.resize(original).copy()  # unprocessed image; same size as the mask
        cnts, _ = cv2.findContours(res["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (255, 255, 255), 5)
        cv2.drawContours(vis, cnts, -1, (42, 120, 214), 2)
        ax.imshow(vis)
        ax.set_axis_off()
        sev = res["severity"]
        ax.text(0.5, 1.09, DX_NAMES[row.dx], transform=ax.transAxes, ha="center",
                fontweight="bold", fontsize=10.5)
        ax.text(0.5, 1.02, f"Severity {sev['level']} · TDS {sev['tds']} · "
                f"P(malignant) {res['prediction']['malignancy_prob']:.0%}",
                transform=ax.transAxes, ha="center", fontsize=9, color=INK2)
    fig.suptitle("Segmentation and grading on held-out images (true diagnosis in bold)",
                 x=0.02, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "segmentation_examples.png")
    plt.close(fig)


def write_metrics(y, mal_p, tds_te, threshold, cutoffs, table, n_images):
    lines = ["# Results (held-out test set)", "",
             f"Test images: {len(y):,} ({y.sum():,} malignant, {(~y).sum():,} benign), split by lesion_id from "
             f"{n_images:,} featurized HAM10000 images. Generated by `evaluate.py`.", "",
             "## Classifier", "",
             f"- Malignancy ROC-AUC: **{roc_auc_score(y, mal_p):.3f}** (ABCD/TDS alone: {roc_auc_score(y, tds_te):.3f})",
             "", "| Cutoff | Malignant caught (recall) | Precision | Benign flagged | Accuracy |", "|---|---|---|---|---|"]
    for t, name in ((0.5, "Default 0.50"), (threshold, f"Tuned {threshold:.3f}")):
        p = mal_p >= t
        lines.append(f"| {name} | {p[y].mean():.1%} ({p[y].sum()}/{y.sum()}) | {y[p].mean():.1%} | "
                     f"{p[~y].mean():.1%} | {(p == y).mean():.1%} |")
    lines += ["", "## Severity levels", "",
              f"TDS cut-offs: Stolz {severity.STOLZ_CUTOFFS[0]} / {severity.STOLZ_CUTOFFS[1]}, "
              f"recalibrated {cutoffs[0]} / {cutoffs[1]}.", "",
              "| Group | Low | Moderate | High |", "|---|---|---|---|"]
    for g, r in table.iterrows():
        lines.append(f"| {g} | " + " | ".join(f"{r[l]:.1%}" for l in severity.LEVELS) + " |")
    lines += ["", "## Figures", ""] + [f"![{p.stem}]({p.name})" for p in sorted(OUT.glob("*.png"))]
    (OUT / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    feats, model, X, tr, te, mal, mal_p, tds = load()
    y, tds_te = mal[te], tds[te]
    threshold = model.get("threshold", 0.5)
    cutoffs = model.get("tds_cutoffs", severity.STOLZ_CUTOFFS)
    table = severity_table(X, te, y, mal_p, threshold, cutoffs)

    fig_roc(y, mal_p, tds_te, threshold)
    fig_confusion(y, mal_p, threshold)
    fig_severity(table)
    fig_tds(tds_te, y, cutoffs)
    fig_importance(model)
    fig_segmentation(feats, te, model)
    write_metrics(y, mal_p, tds_te, threshold, cutoffs, table, len(feats))
    print(f"Wrote figures and metrics.md to {OUT.resolve()}")


if __name__ == "__main__":
    main()
