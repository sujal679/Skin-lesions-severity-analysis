"""Train the lesion classifier on HAM10000 (or any ISIC-style dataset).

Expected layout (HAM10000 from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T):
    data/HAM10000_metadata.csv      (columns: image_id, dx, ...)
    data/images/<image_id>.jpg

Usage:
    python train.py --data data --binary          # benign vs malignant
    python train.py --data data                   # 7-class (akiec, bcc, bkl, df, mel, nv, vasc)
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict

from lesion import features, preprocess, segment, severity
from lesion.pipeline import MODEL_PATH

MALIGNANT = {"mel", "bcc", "akiec"}


def featurize(path):
    try:
        clean, _, fov = preprocess.preprocess(preprocess.load_image(path))
        mask = segment.segment(clean, fov)
        if mask.sum() < 0.005 * mask.size:
            return None
        return features.extract(clean, mask)
    except Exception as e:
        print(f"skip {path}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--binary", action="store_true", help="train benign vs malignant")
    ap.add_argument("--limit", type=int, default=0, help="use only N images (quick test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--target-recall", type=float, default=0.80,
                    help="fraction of malignant lesions the cutoff should catch")
    ap.add_argument("--high-specificity", type=float, default=0.90,
                    help="fraction of benign lesions that must stay below the 'High' TDS cut-off")
    args = ap.parse_args()

    data = Path(args.data)
    meta_files = list(data.rglob("HAM10000_metadata*")) if data.exists() else []
    if not meta_files:
        raise SystemExit(
            f"Dataset not found in '{data.resolve()}'.\n"
            "Download HAM10000 (e.g. kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000),\n"
            "extract it into that folder, then re-run. See README.md.")
    meta = pd.read_csv(meta_files[0], sep=None, engine="python")

    # Images may be spread across HAM10000_images_part_1 / part_2 etc. - index them all.
    index = {p.stem: p for p in data.rglob("*.jpg")}
    meta = meta[meta.image_id.isin(index)]
    if meta.empty:
        raise SystemExit(f"Found {meta_files[0].name} but no matching .jpg images under '{data.resolve()}'.")
    if args.limit:
        meta = meta.sample(min(args.limit, len(meta)), random_state=0)
    paths = [index[i] for i in meta.image_id]

    cache = data / "features.csv"
    if cache.exists() and not args.limit:
        feats = pd.read_csv(cache)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            rows = list(ex.map(featurize, paths, chunksize=16))
        keep = [r is not None for r in rows]
        feats = pd.DataFrame([r for r in rows if r is not None])
        feats.insert(0, "image_id", meta.image_id.values[keep])
        feats.insert(1, "lesion_id", meta.lesion_id.values[keep])
        feats.insert(2, "dx", meta.dx.values[keep])
        if not args.limit:
            feats.to_csv(cache, index=False)
    print(f"{len(feats)} images featurized")

    y = feats.dx.map(lambda d: "malignant" if d in MALIGNANT else "benign") if args.binary else feats.dx
    X = feats.drop(columns=["image_id", "lesion_id", "dx"]).fillna(0)

    # Split by lesion_id so images of the same lesion never leak between train and test.
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, feats.lesion_id))
    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                 min_samples_leaf=2, n_jobs=-1, random_state=42)
    malignant = {"malignant"} if args.binary else MALIGNANT

    # Choose the malignancy cutoff on out-of-fold predictions from the training set only,
    # so the test set stays untouched. Pick the highest cutoff that reaches the target recall.
    oof = cross_val_predict(clf, X.iloc[tr], y.iloc[tr], groups=feats.lesion_id.iloc[tr],
                            cv=GroupKFold(5), method="predict_proba")
    classes = sorted(y.unique())
    oof_mal = oof[:, [c in malignant for c in classes]].sum(1)
    fpr, tpr, thr = roc_curve(y.iloc[tr].isin(malignant), oof_mal)
    threshold = float(thr[np.argmax(tpr >= args.target_recall)])
    print(f"Cutoff for {args.target_recall:.0%} malignant recall (cross-validated): {threshold:.3f}")

    clf.fit(X.iloc[tr], y.iloc[tr])
    proba = clf.predict_proba(X.iloc[te])
    mal_p = proba[:, [c in malignant for c in clf.classes_]].sum(1)
    y_true = np.where(y.iloc[te].isin(malignant), "malignant", "benign")
    print(f"Malignancy ROC-AUC: {roc_auc_score(y_true == 'malignant', mal_p):.3f}")
    for name, t in (("default 0.50", 0.5), (f"tuned {threshold:.3f}", threshold)):
        pred = np.where(mal_p >= t, "malignant", "benign")
        print(f"\n--- Test set, cutoff {name} ---")
        print(classification_report(y_true, pred, zero_division=0))
        print("Confusion matrix (rows=true): ['benign', 'malignant']")
        print(confusion_matrix(y_true, pred, labels=["benign", "malignant"]))

    # Recalibrate the TDS severity cut-offs on the training set: Moderate catches the target share of
    # malignant lesions, High is reached by at most (1 - high_specificity) of benign lesions.
    tds_all = X.apply(severity.tds, axis=1)
    tds_tr, mal_tr = tds_all.iloc[tr], y.iloc[tr].isin(malignant)
    values = np.sort(tds_tr.unique())
    moderate_cut = max(v for v in values if (tds_tr[mal_tr] >= v).mean() >= args.target_recall)
    high_cut = min(v for v in values if (tds_tr[~mal_tr] >= v).mean() <= 1 - args.high_specificity)
    tds_cutoffs = (round(float(moderate_cut), 2), round(float(high_cut), 2))
    print(f"\nTDS cut-offs (Stolz 4.75 / 5.45) recalibrated to: Moderate >= {tds_cutoffs[0]}, High >= {tds_cutoffs[1]}")

    for name, cuts in (("Stolz cut-offs", severity.STOLZ_CUTOFFS), ("recalibrated cut-offs", tds_cutoffs)):
        for label, prob in (("TDS only", None), ("TDS + classifier", mal_p)):
            levels = [severity.severity(X.iloc[i], None if prob is None else prob[k], threshold, cuts)["level"]
                      for k, i in enumerate(te)]
            table = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(levels, name="level"), normalize="index")
            print(f"\n--- Test set severity, {name}, {label} (% of each true class) ---")
            print((table.reindex(columns=severity.LEVELS, fill_value=0) * 100).round(1).to_string())

    imp = pd.Series(clf.feature_importances_, X.columns).sort_values(ascending=False)
    print("\nTop features:\n", imp.head(15).to_string())

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"clf": clf, "columns": list(X.columns), "malignant": malignant,
                 "threshold": threshold, "tds_cutoffs": tds_cutoffs}, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
