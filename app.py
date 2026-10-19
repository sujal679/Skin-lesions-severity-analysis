"""Streamlit web app. Run: streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from lesion import pipeline, severity

st.set_page_config(page_title="DermaVision", layout="wide")
st.title("Automated Detection & Severity Analysis of Pigmented Skin Lesions")
st.caption("Educational Tool — not a Medical Device. Always consult a dermatologist.")


@st.cache_resource
def get_model():
    return pipeline.load_model()


model = get_model()
if model is None:
    st.info("No trained classifier found (models/lesion_rf.joblib). Severity uses the ABCD rule only.")

up = st.file_uploader("Upload a Dermoscopy image", type=["jpg", "jpeg", "png", "bmp"])
px_per_mm = st.sidebar.number_input("Pixels per mm (Resize to 512px-Optional)", 0.0, 500.0, 0.0)

if up:
    img = np.array(Image.open(up).convert("RGB"))
    try:
        res = pipeline.analyze(img, model, px_per_mm or None)
    except pipeline.SegmentationError as e:
        st.error(str(e))
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.image(img, caption="Original", use_container_width=True)
    c2.image(res["image"], caption="Preprocessed (colour-normalised, hair removed)", use_container_width=True)
    c3.image(pipeline.overlay(res["image"], res["mask"]), caption="Segmented lesion", use_container_width=True)

    sev, f = res["severity"], res["features"]
    colour = {"Low": "green", "Moderate": "orange", "High": "red"}[sev["level"]]
    st.markdown(f"## Severity: :{colour}[{sev['level']}]")
    st.write(sev["advice"])

    # Two rows of three so labels and values stay readable in narrow windows.
    row1, row2 = st.columns(3), st.columns(3)
    row1[0].metric("Asymmetry", f"{f['A']} / 2", help="Shape and colour asymmetry about two axes")
    row1[1].metric("Border", f"{f['B']} / 8", help="Number of 8 border segments with an abrupt edge")
    row1[2].metric("Colours", f"{f['C']} / 6", help="Dermoscopic colours present (1-6)")
    row2[0].metric("Structures", f"{f['D']} / 5", help="Differential structures (texture-based estimate)")
    lo, hi = model.get("tds_cutoffs", severity.STOLZ_CUTOFFS) if model else severity.STOLZ_CUTOFFS
    row2[1].metric("TDS", sev["tds"], help=f"Total Dermoscopy Score: <{lo} low, {lo}-{hi} moderate, {hi}+ high")
    row2[2].metric("Risk index", f"{sev['risk_index']:.2f}", help="0-1, combines TDS and classifier")
    st.progress(sev["risk_index"])

    if res["prediction"]:
        st.subheader("Classifier prediction")
        st.write(f"Most likely: **{res['prediction']['class']}** — "
                 f"malignancy probability {res['prediction']['malignancy_prob']:.1%}")
        st.bar_chart(pd.Series(res["prediction"]["probabilities"]))

    with st.expander("Colour composition"):
        st.bar_chart(pd.Series({k[7:]: v for k, v in f.items() if k.startswith("colour_")}))
    with st.expander("All extracted features"):
        st.dataframe(pd.Series(f, name="value").astype(float).round(4))
