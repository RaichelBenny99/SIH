"""
streamlit_app.py — Plant Disease Detector (Enhanced v2)

Features:
  1. EfficientNet-B0 inference (PyTorch)
  2. Grad-CAM explainability overlay
  3. Treatment & pesticide recommendations
  4. Confidence progress bar + low-confidence warning
  ── NEW ──
  5. Image Quality Assessment   (image_quality.py)
  6. Disease Severity Estimation (severity_estimator.py)
  7. Confidence Calibration      (temperature_scaling.py)
  8. Performance logging (inference time)
"""

import time
import os
import io

import streamlit as st
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

# Local modules (same directory)
from gradcam import generate_gradcam, overlay_heatmap
from treatment_info import get_treatment_info
from image_quality import check_image_quality
from severity_estimator import estimate_severity
from temperature_scaling import TemperatureScaler

# =====================================================================
# CONFIGURATION
# =====================================================================

MODEL_URL = "https://drive.google.com/uc?id=1YvUoajFYHDPHckBmE0Xl1znHi89ZP2pU"
MODEL_PATH = "plant_disease_model.pth"
CLASSES_FILE = "classes.txt"
NUM_CLASSES = 39
LOW_CONFIDENCE_THRESHOLD = 0.60  # warn below 60 %

# Path to a pre-saved temperature file (optional — see README)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_SCALE_PATH = os.path.join(SCRIPT_DIR, "temperature.pth")

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def download_model_from_drive():
    """Download model from Google Drive if not already present."""
    import gdown
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive ...")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            st.stop()
    return True


def load_class_names():
    """Load the 39 class names from classes.txt (same directory as script)."""
    path = os.path.join(SCRIPT_DIR, CLASSES_FILE)
    if not os.path.exists(path):
        st.error(f"Classes file not found at: {path}")
        st.stop()
    with open(path, "r") as f:
        names = [line.strip() for line in f.readlines()]
    if len(names) != NUM_CLASSES:
        st.warning(f"Expected {NUM_CLASSES} classes, found {len(names)}")
    return names


# =====================================================================
# MODEL LOADING (cached so it loads only once)
# =====================================================================

@st.cache_resource
def load_model():
    """Load the trained EfficientNet-B0 with custom classifier."""
    download_model_from_drive()
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model


@st.cache_resource
def load_temperature_scaler():
    """
    Load the temperature scaler.
    If a pre-fitted temperature file exists, load it.
    Otherwise return an unfitted scaler (T=1.0, i.e. uncalibrated).
    """
    scaler = TemperatureScaler()
    if os.path.exists(TEMP_SCALE_PATH):
        scaler.load(TEMP_SCALE_PATH)
    return scaler


# =====================================================================
# IMAGE TRANSFORMS (same as validation in training)
# =====================================================================

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =====================================================================
# PREDICTION + GRAD-CAM  (updated to expose raw logits & heatmap)
# =====================================================================

def predict_and_explain(pil_image, model, class_names, scaler):
    """
    Run prediction + Grad-CAM on a PIL image.

    Returns
    -------
    prediction       : str          — class name
    raw_confidence   : float        — uncalibrated top-1 probability (0-1)
    cal_confidence   : float        — calibrated top-1 probability  (0-1)
    gradcam_img      : PIL.Image    — heatmap overlay
    gradcam_heatmap  : numpy array  — raw (224,224) heatmap for severity
    elapsed_ms       : float        — inference time in milliseconds
    """
    t0 = time.perf_counter()

    # Pre-process for the model
    image_tensor = TRANSFORM(pil_image).unsqueeze(0)  # (1, 3, 224, 224)

    # --- Forward pass (get raw logits) ---
    with torch.no_grad():
        outputs = model(image_tensor)              # (1, 39)
        raw_probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = outputs.argmax(dim=1).item()

    prediction = class_names[pred_idx]
    raw_confidence = raw_probs[pred_idx].item()

    # --- Calibrated confidence ---
    cal_probs = scaler.calibrated_softmax(outputs[0])  # (39,)
    cal_confidence = cal_probs[pred_idx].item()

    # --- Grad-CAM ---
    heatmap = generate_gradcam(model, image_tensor, target_class=pred_idx)
    gradcam_img = overlay_heatmap(pil_image, heatmap, alpha=0.45)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return prediction, raw_confidence, cal_confidence, gradcam_img, heatmap, elapsed_ms


# =====================================================================
# STREAMLIT UI
# =====================================================================

# --- Page config ---
st.set_page_config(
    page_title="Plant Disease Detector - SIH",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS (includes severity card styling) ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .treatment-card {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .pesticide-card {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .severity-card {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .quality-warning {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .perf-badge {
        text-align: right;
        color: #999;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">🌿 Plant Disease Detector</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">SIH Hackathon — PlantVillage Disease Classification '
    '| EfficientNet-B0 + Grad-CAM + Treatment Advisor</div>',
    unsafe_allow_html=True,
)

# --- Load model, classes, scaler ---
try:
    model = load_model()
    class_names = load_class_names()
    scaler = load_temperature_scaler()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Layout: two columns ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### Upload Plant Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image of a plant leaf ...",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    if uploaded_file is not None:
        # Open PIL image once (reused everywhere)
        pil_image = Image.open(uploaded_file).convert("RGB")

        # Show the uploaded image
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

        # ==============================================================
        # FEATURE 1 — Image Quality Assessment
        # ==============================================================
        is_good, quality_msg, quality_metrics = check_image_quality(pil_image)

        # Always show quality metrics in a small expander
        with st.expander("📊 Image Quality Metrics", expanded=False):
            qm = quality_metrics
            q1, q2, q3 = st.columns(3)
            q1.metric("Sharpness", f"{qm['blur_score']:.1f}",
                       help="Laplacian variance — higher is sharper")
            q2.metric("Brightness", f"{qm['mean_brightness']:.0f} / 255")
            q3.metric("Resolution", f"{qm['width']}×{qm['height']} px")

        if not is_good:
            # Show warning and STOP prediction
            st.markdown(f"""
            <div class="quality-warning">
                <h4>⚠️ Image Quality Issue</h4>
                <pre style="white-space: pre-wrap;">{quality_msg}</pre>
            </div>
            """, unsafe_allow_html=True)
            st.warning("Please upload a better-quality image before analysis.")
            st.stop()   # ← prevents prediction from running

        # ==============================================================
        # Analyse button (only reachable if image quality is OK)
        # ==============================================================
        if st.button("Analyze Disease", type="primary",
                      use_container_width=True):
            with st.spinner("Analyzing plant health ..."):
                (prediction, raw_conf, cal_conf,
                 gradcam_img, heatmap, elapsed_ms) = predict_and_explain(
                    pil_image, model, class_names, scaler
                )

            # ---- Performance badge ----
            st.markdown(
                f'<div class="perf-badge">⏱ Inference: {elapsed_ms:.0f} ms</div>',
                unsafe_allow_html=True,
            )

            # ==========================================================
            # Results section
            # ==========================================================
            st.markdown("---")
            st.markdown("### Analysis Results")

            # Use calibrated confidence as the "primary" confidence shown
            confidence = cal_conf
            display_name = prediction.replace("___", " — ").replace("_", " ")

            # ---- Prediction + confidence card ----
            cal_label = " (calibrated)" if scaler.fitted else ""
            st.markdown(f"""
            <div class="result-card">
                <h4>🔬 Disease Detection</h4>
                <p><strong>Disease:</strong> {display_name}</p>
                <p><strong>Confidence:</strong> {confidence * 100:.1f} %{cal_label}</p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence progress bar
            st.progress(confidence)

            # Low-confidence warning
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                st.warning(
                    f"⚠ Low confidence ({confidence * 100:.1f} %). "
                    "The prediction may be unreliable — consider uploading "
                    "a clearer image or consulting an expert."
                )

            # If temperature scaler is fitted, show both raw & calibrated
            if scaler.fitted:
                with st.expander("🔧 Calibration Details", expanded=False):
                    c1, c2 = st.columns(2)
                    c1.metric("Raw Confidence",
                              f"{raw_conf * 100:.1f} %")
                    c2.metric("Calibrated Confidence",
                              f"{cal_conf * 100:.1f} %")
                    st.caption(
                        f"Temperature T = {scaler.temperature:.3f}  "
                        "(learned via temperature scaling on validation set)"
                    )

            # Healthy vs diseased feedback
            if "healthy" in prediction.lower():
                st.success("**Plant appears Healthy!**")
                st.balloons()
            else:
                st.warning("⚠ **Disease Detected** — see treatment below.")

            # ==========================================================
            # FEATURE 2 — Disease Severity Estimation
            # ==========================================================
            if "healthy" not in prediction.lower():
                sev = estimate_severity(heatmap, pil_image)

                st.markdown("### Disease Severity")
                sev_icon = (
                    '🟢' if sev['severity'] == 'Mild'
                    else '🟠' if sev['severity'] == 'Moderate'
                    else '🔴'
                )
                st.markdown(f"""
                <div class="severity-card" style="border-left: 5px solid {sev['color']};
                     background-color: {sev['color']}15;">
                    <h4 style="color: {sev['color']};">
                        {sev_icon} Severity: {sev['severity']}
                    </h4>
                    <p><strong>Estimated Infected Area:</strong> {sev['infected_pct']} %</p>
                </div>
                """, unsafe_allow_html=True)

                # Small progress-style bar for infected area
                st.progress(min(sev["infected_pct"] / 100.0, 1.0))

            # ==========================================================
            # Grad-CAM images (side by side)
            # ==========================================================
            st.markdown("### Grad-CAM Explainability")
            gc_left, gc_right = st.columns(2)
            with gc_left:
                st.image(
                    pil_image,
                    caption="Original Image",
                    use_column_width=True,
                )
            with gc_right:
                st.image(
                    gradcam_img,
                    caption="Grad-CAM Heatmap Overlay",
                    use_column_width=True,
                )

            # ==========================================================
            # Treatment Recommendation
            # ==========================================================
            st.markdown("### Treatment Recommendation")
            info = get_treatment_info(prediction)

            # Description
            st.markdown(f"""
            <div class="treatment-card">
                <h4>📋 Description</h4>
                <p>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Treatment steps
            st.markdown("**Recommended Treatment:**")
            for step in info["treatment"]:
                st.markdown(f"- {step}")

            # Pesticide
            st.markdown(f"""
            <div class="pesticide-card">
                <h4>💊 Recommended Pesticide / Product</h4>
                <p>{info['pesticide']}</p>
            </div>
            """, unsafe_allow_html=True)


with col_right:
    st.markdown("### How to Use")
    st.info("""
    1. **Upload** a clear image of a plant leaf
    2. **Quality check** runs automatically
    3. **Click** 'Analyze Disease'
    4. **View** prediction, severity, Grad-CAM, and treatment
    5. **Check** calibrated confidence level
    """)

    st.markdown("### Tips for Best Results")
    st.success("""
    - Use well-lit, clear images
    - Ensure leaf fills most of the frame
    - Avoid blurry or dark photos
    - Multiple angles can help accuracy
    """)

    st.markdown("### Model Information")
    st.info(f"""
    - **Classes:** {len(class_names)} diseases
    - **Architecture:** EfficientNet-B0
    - **Dataset:** PlantVillage
    - **Explainability:** Grad-CAM
    - **Calibration:** Temperature Scaling
    """)

    st.markdown("### Features")
    st.success("""
    - Image quality gate (blur / brightness / resolution)
    - Grad-CAM heatmap visualisation
    - Disease severity estimation (Mild / Moderate / Severe)
    - Calibrated confidence + progress bar
    - Low-confidence warning (< 60 %)
    - Disease treatment & pesticide advice
    - Inference time logging
    """)

    # ------------------------------------------------------------------
    # Sidebar: Temperature Scaling calibration utility
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### ⚙️ Calibration")

    if scaler.fitted:
        st.caption(f"Temperature loaded: **T = {scaler.temperature:.4f}**")
    else:
        st.caption("No calibration file found — using raw softmax (T = 1.0).")

    with st.expander("Calibrate from validation logits", expanded=False):
        st.markdown(
            "Upload `val_logits.npy` (N×39) and `val_labels.npy` (N,) "
            "saved from your validation run."
        )
        logits_file = st.file_uploader("val_logits.npy", type=["npy"],
                                        key="logits_upload")
        labels_file = st.file_uploader("val_labels.npy", type=["npy"],
                                        key="labels_upload")
        if logits_file and labels_file:
            if st.button("Fit Temperature", key="fit_temp_btn"):
                logits_np = np.load(io.BytesIO(logits_file.read()))
                labels_np = np.load(io.BytesIO(labels_file.read()))
                logits_t = torch.from_numpy(logits_np).float()
                labels_t = torch.from_numpy(labels_np).long()

                new_scaler = TemperatureScaler()
                T = new_scaler.fit(logits_t, labels_t)
                new_scaler.save(TEMP_SCALE_PATH)
                st.success(f"Temperature fitted! T = {T:.4f} — saved to "
                           f"`{TEMP_SCALE_PATH}`. Reload the page to use it.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌱 <strong>SIH Hackathon Project</strong></p>
    <p>Plant Disease Detection using Deep Learning — EfficientNet-B0 + Grad-CAM</p>
</div>
""", unsafe_allow_html=True)
