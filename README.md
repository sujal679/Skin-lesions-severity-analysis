# Automated Detection and Severity Analysis of Pigmented Skin Lesions Using Dermoscopy Images

A computer-aided diagnosis (CAD) pipeline that segments a pigmented lesion in a dermoscopy image,
extracts clinically meaningful ABCD features, classifies it, and grades its severity.

> Educational/research project — not a medical device.

## Pipeline

```
Dermoscopy image
  └─ 1. Preprocessing      resize → Shades-of-Gray colour constancy → DullRazor hair removal → vignette mask
  └─ 2. Segmentation       Lab distance from estimated skin colour → Otsu → central component → GrabCut refinement
  └─ 3. Feature extraction ABCD rule + colour statistics (RGB/HSV/Lab) + texture (GLCM, LBP, blob count)
  └─ 4. Classification     Random Forest trained on HAM10000 (7-class or benign/malignant)
  └─ 5. Severity grading   Stolz Total Dermoscopy Score (TDS), fused with the classifier's malignancy probability
```

### ABCD rule / Total Dermoscopy Score
| Criterion | Range | Weight | How it is computed |
|---|---|---|---|
| **A**symmetry | 0–2 | 1.3 | shape XOR + colour difference after mirroring about both principal axes |
| **B**order | 0–8 | 0.1 | 8 segments; a segment scores if the pigment cuts off abruptly (≥20% of lesion–skin contrast within a thin edge band) |
| **C**olour | 1–6 | 0.5 | pixels matched to white, red, light-brown, dark-brown, blue-gray, black (>5% counts) |
| **D**ifferential structures | 0–5 | 0.5 | texture-based proxy (GLCM contrast/homogeneity, LBP entropy, dots) |

`TDS = 1.3·A + 0.1·B + 0.5·C + 0.5·D`

| Stolz TDS | Recalibrated TDS | Severity | Interpretation |
|---|---|---|---|
| < 4.75 | < 4.1 | **Low** | benign-appearing |
| 4.75 – 5.45 | 4.1 – 5.1 | **Moderate** | suspicious |
| ≥ 5.45 | ≥ 5.1 | **High** | highly suspicious for melanoma |

The Stolz cut-offs were set for TDS scored by dermatologists. Automated scores run lower (only 5% of
malignant HAM10000 lesions reach 5.45), so `train.py` recalibrates them on the training split:
**Moderate** is set to catch 80% of malignant lesions (`--target-recall`) and **High** so that at most
10% of benign lesions reach it (`--high-specificity`, ~the 90% specificity Stolz reported). The cut-offs
are saved with the model; without a model the Stolz values are used.

Held-out test set (2,023 images), final level with TDS + classifier, % of each true class:

| | Low | Moderate | High |
|---|---|---|---|
| Malignant, Stolz cut-offs | 16.0 | 72.3 | 11.7 |
| Malignant, recalibrated | **8.4** | 67.4 | **24.2** |
| Benign, Stolz cut-offs | 60.5 | 36.1 | 3.4 |
| Benign, recalibrated | 32.9 | 57.5 | 9.6 |

When a trained model exists, a *risk index* = mean(normalised TDS, malignancy probability) is reported, and
the level is escalated one step if the model's malignancy probability exceeds 0.7.

## Setup
```bash
pip install -r requirements.txt
```

## Dataset (HAM10000)
Download from Kaggle (kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) or Harvard Dataverse
(doi:10.7910/DVN/DBW86T) and extract everything into `data/`. Images are found in any subfolder:
```
data/HAM10000_metadata.csv
data/HAM10000_images_part_1/ISIC_0024306.jpg ...
data/HAM10000_images_part_2/...
```
Malignant classes: `mel` (melanoma), `bcc` (basal cell carcinoma), `akiec` (actinic keratosis / Bowen's).

## Usage
```bash
python train.py --data data --binary      # benign vs malignant (features cached to data/features.csv)
python train.py --data data               # 7-class
python analyze.py image.jpg --save overlay.png
streamlit run app.py                      # web interface
```
Training splits by `lesion_id` to avoid the same lesion appearing in both train and test sets, and
reports precision/recall/F1, confusion matrix, malignancy ROC-AUC and top feature importances.

## Project structure
```
lesion/preprocess.py   hair removal, colour constancy
lesion/segment.py      Otsu-based lesion segmentation
lesion/features.py     ABCD + colour + texture features
lesion/severity.py     TDS and severity grading
lesion/pipeline.py     end-to-end analysis
train.py               classifier training/evaluation
analyze.py             CLI
app.py                 Streamlit UI
```

## Limitations & future work
- "D" is a texture proxy, not a true detector of network/globules/streaks/dots.
- Segmentation was checked visually and with a failure heuristic (3 of 400 images flagged), not against
  ground-truth masks. Very faint lesions can still be under-segmented; a U-Net trained on ISIC 2018 Task 1
  masks would improve this (evaluate with Dice/Jaccard).
- A CNN (e.g. EfficientNet fine-tuned on ISIC) typically outperforms hand-crafted features; its
  probability can be plugged into `severity.severity(feats, malignancy_prob)` unchanged.
