"""Command-line analysis of a single dermoscopy image.

Usage: python analyze.py path/to/image.jpg [--save out.png]
"""
import argparse
import json

import cv2

from lesion import pipeline, preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--save", help="write segmentation overlay to this path")
    ap.add_argument("--px-per-mm", type=float, help="image scale, to report diameter in mm")
    args = ap.parse_args()

    res = pipeline.analyze(preprocess.load_image(args.image), pipeline.load_model(), args.px_per_mm)
    f = res["features"]
    print(f"Asymmetry A={f['A']}/2  Border B={f['B']}/8  Colours C={f['C']}/6  Structures D={f['D']}/5")
    print(json.dumps(res["severity"], indent=2))
    if res["prediction"]:
        print(json.dumps(res["prediction"], indent=2))
    else:
        print("(no trained model found - severity is from the ABCD rule only; run train.py)")
    if args.save:
        cv2.imwrite(args.save, cv2.cvtColor(pipeline.overlay(res["image"], res["mask"]), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
