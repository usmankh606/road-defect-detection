"""
Road Defect Detection Dashboard
================================
Streamlit app for classifying road surface images into:
Pothole, Crack, or Manhole — using a CNN trained in Keras/TensorFlow.

Run with:
    streamlit run main.py

Expects a trained model file named `road_defect_model.keras`
(or `road_defect_model.h5`) in the same directory as this script,
or lets the user upload one from the sidebar.
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# TensorFlow import is wrapped so the app can still show a friendly
# message if TF isn't installed / model isn't found yet.
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Road Defect Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMG_SIZE = 128
CLASS_NAMES = ["Pothole", "Crack", "Manhole"]
CLASS_ICONS = {"Pothole": "🕳️", "Crack": "⚡", "Manhole": "⚫"}
CLASS_COLORS = {"Pothole": "#E63946", "Crack": "#F4A261", "Manhole": "#457B9D"}
DEFAULT_MODEL_PATHS = ["road_defect_model.keras", "road_defect_model.h5"]


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }

    .app-header {
        padding: 1.75rem 2rem;
        background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%);
        border-radius: 14px;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .app-header p {
        color: #dbe7f0;
        font-size: 1rem;
        margin: 0;
    }

    .metric-card {
        background-color: #161b22;
        border: 1px solid #2d333b;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        text-align: center;
    }
    .metric-card h2 {
        margin: 0;
        font-size: 1.7rem;
    }
    .metric-card p {
        margin: 0;
        color: #8b949e;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .result-card {
        border-radius: 14px;
        padding: 1.5rem;
        background-color: #161b22;
        border: 1px solid #2d333b;
    }

    .footer-note {
        text-align: center;
        color: #6e7681;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #2d333b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load and cache the Keras model."""
    return tf.keras.models.load_model(model_path)


def find_default_model():
    for path in DEFAULT_MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None


def preprocess_image(pil_image: Image.Image, img_size: int = IMG_SIZE) -> np.ndarray:
    """Match the notebook's preprocessing: BGR resize + /255 scaling."""
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict(model, pil_image: Image.Image):
    batch = preprocess_image(pil_image)
    start = time.time()
    preds = model.predict(batch, verbose=0)[0]
    elapsed_ms = (time.time() - start) * 1000
    return preds, elapsed_ms


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: filename, class, confidence, time


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛣️ Road Defect Detection")
    st.caption("CNN-based road surface classifier")
    st.divider()

    st.markdown("#### Model")
    default_model_path = find_default_model()

    model_source = st.radio(
        "Model source",
        ["Use bundled model", "Upload a model file"],
        index=0 if default_model_path else 1,
        label_visibility="collapsed",
    )

    model_path = None
    uploaded_model_bytes = None

    if model_source == "Use bundled model":
        if default_model_path:
            model_path = default_model_path
            st.success(f"Found: `{default_model_path}`")
        else:
            st.warning(
                "No bundled model found. Place `road_defect_model.keras` "
                "next to `main.py`, or upload one below."
            )
    else:
        model_file = st.file_uploader(
            "Upload .keras / .h5 model", type=["keras", "h5"]
        )
        if model_file is not None:
            tmp_path = f"_uploaded_{model_file.name}"
            with open(tmp_path, "wb") as f:
                f.write(model_file.getbuffer())
            model_path = tmp_path

    st.divider()
    st.markdown("#### Classes")
    for name in CLASS_NAMES:
        st.markdown(
            f"<span style='color:{CLASS_COLORS[name]}'>●</span> "
            f"{CLASS_ICONS[name]} {name}",
            unsafe_allow_html=True,
        )

    st.divider()
    confidence_threshold = st.slider(
        "Confidence flag threshold (%)", min_value=0, max_value=100, value=60,
        help="Predictions below this confidence are flagged as uncertain.",
    )

    st.divider()
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown(
        "<div class='footer-note'>Built with Streamlit + TensorFlow</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🛣️ Road Defect Detection Dashboard</h1>
        <p>Upload a road surface image to detect potholes, cracks, or manholes using a trained CNN.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not TF_AVAILABLE:
    st.error(
        "TensorFlow isn't installed in this environment. Run "
        "`pip install -r requirements.txt` and restart the app."
    )
    st.stop()

# Summary metric row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"<div class='metric-card'><h2>{len(st.session_state.history)}</h2>"
        f"<p>Images Analyzed</p></div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"<div class='metric-card'><h2>{IMG_SIZE}×{IMG_SIZE}</h2>"
        f"<p>Model Input Size</p></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"<div class='metric-card'><h2>{len(CLASS_NAMES)}</h2>"
        f"<p>Defect Classes</p></div>",
        unsafe_allow_html=True,
    )
with col4:
    status = "🟢 Ready" if model_path else "🔴 No Model"
    st.markdown(
        f"<div class='metric-card'><h2>{status}</h2>"
        f"<p>Model Status</p></div>",
        unsafe_allow_html=True,
    )

st.write("")

# --------------------------------------------------------------------------
# Main layout: upload + result
# --------------------------------------------------------------------------
left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("📤 Upload Image")
    uploaded_files = st.file_uploader(
        "Drop road image(s) here (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    sample_note = st.container()
    if uploaded_files:
        selected_name = st.selectbox(
            "Select image to inspect",
            options=[f.name for f in uploaded_files],
        )
        selected_file = next(f for f in uploaded_files if f.name == selected_name)
        image = Image.open(selected_file)
        st.image(image, caption=selected_name, use_container_width=True)
    else:
        st.info("Upload one or more images to get started.")
        image = None
        selected_file = None

with right:
    st.subheader("🔍 Prediction")

    if image is None:
        st.markdown(
            "<div class='result-card'>Waiting for an image…</div>",
            unsafe_allow_html=True,
        )
    elif model_path is None:
        st.warning("No model loaded — add a model file from the sidebar first.")
    else:
        run_all = len(uploaded_files) > 1
        analyze_clicked = st.button(
            "🚀 Analyze" + (" All" if run_all else ""), type="primary", use_container_width=True
        )

        if analyze_clicked:
            with st.spinner("Loading model and running inference…"):
                model = load_model(model_path)
                targets = uploaded_files if run_all else [selected_file]
                batch_results = []
                for f in targets:
                    img = Image.open(f)
                    preds, elapsed_ms = predict(model, img)
                    top_idx = int(np.argmax(preds))
                    top_class = CLASS_NAMES[top_idx]
                    top_conf = float(preds[top_idx]) * 100
                    batch_results.append(
                        {
                            "filename": f.name,
                            "preds": preds,
                            "class": top_class,
                            "confidence": top_conf,
                            "elapsed_ms": elapsed_ms,
                        }
                    )
                    st.session_state.history.append(
                        {
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Filename": f.name,
                            "Prediction": top_class,
                            "Confidence (%)": round(top_conf, 2),
                        }
                    )

            main_result = batch_results[0]
            preds = main_result["preds"]
            top_class = main_result["class"]
            top_conf = main_result["confidence"]

            flagged = top_conf < confidence_threshold
            badge_color = CLASS_COLORS[top_class]

            st.markdown(
                f"""
                <div class="result-card">
                    <div style="display:flex; align-items:center; gap:0.75rem;">
                        <span style="font-size:2.2rem;">{CLASS_ICONS[top_class]}</span>
                        <div>
                            <div style="font-size:1.4rem; font-weight:700; color:{badge_color};">
                                {top_class}
                            </div>
                            <div style="color:#8b949e;">
                                Confidence: {top_conf:.1f}% &nbsp;|&nbsp; Inference: {main_result['elapsed_ms']:.0f} ms
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if flagged:
                st.warning(
                    f"⚠️ Confidence is below the {confidence_threshold}% threshold — "
                    "consider manual review."
                )

            st.write("")
            st.markdown("**Class probabilities**")
            prob_df = pd.DataFrame(
                {"Class": CLASS_NAMES, "Probability (%)": (preds * 100).round(2)}
            ).sort_values("Probability (%)", ascending=False)
            st.bar_chart(
                prob_df.set_index("Class"), y="Probability (%)", use_container_width=True
            )
            st.dataframe(prob_df, hide_index=True, use_container_width=True)

            if run_all and len(batch_results) > 1:
                st.write("")
                st.markdown("**Batch results**")
                batch_df = pd.DataFrame(
                    [
                        {
                            "Filename": r["filename"],
                            "Prediction": r["class"],
                            "Confidence (%)": round(r["confidence"], 2),
                        }
                        for r in batch_results
                    ]
                )
                st.dataframe(batch_df, hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------
# History section
# --------------------------------------------------------------------------
st.divider()
st.subheader("📜 Prediction History")

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    tab1, tab2 = st.tabs(["Table", "Class Breakdown"])
    with tab1:
        st.dataframe(hist_df.iloc[::-1], hide_index=True, use_container_width=True)
    with tab2:
        counts = hist_df["Prediction"].value_counts()
        st.bar_chart(counts)
else:
    st.caption("No predictions yet — analyzed images will appear here.")

st.markdown(
    "<div class='footer-note'>Road Defect Detection Dashboard · "
    "CNN model trained on pothole / crack / manhole imagery</div>",
    unsafe_allow_html=True,
)