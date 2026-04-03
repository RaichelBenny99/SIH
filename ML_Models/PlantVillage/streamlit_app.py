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
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS (includes severity card styling) ---
st.markdown("""
<style>
    :root {
        --bg: #07070b;
        --panel: #0f0f16;
        --panel-2: #141420;
        --text: #f2f2ff;
        --muted: #b9b9d6;
        --border: rgba(255, 43, 214, 0.22);
        --magenta: #ff2bd6;
        --magenta-2: #b100ff;
        --success: #2ef2b2;
        --warning: #ffcc00;
        --danger: #ff4d6d;
        --shadow: 0 10px 30px rgba(0,0,0,0.55);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(1200px 600px at 20% 0%, rgba(255, 43, 214, 0.12), transparent 60%),
                    radial-gradient(900px 500px at 100% 10%, rgba(177, 0, 255, 0.10), transparent 55%),
                    var(--bg) !important;
        color: var(--text) !important;
    }

    /* Main content + sidebar panels */
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stSidebar"] {
        background: transparent !important;
        color: var(--text) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, rgba(255, 43, 214, 0.06), transparent 28%),
                    var(--panel) !important;
        border-right: 1px solid var(--border);
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6, p, li, label, span, div {
        color: var(--text);
    }
    .stMarkdown, .stText, .stCaption {
        color: var(--text);
    }
    .stCaption, .perf-badge {
        color: var(--muted) !important;
    }

    /* Buttons */
    .stButton > button {
        border: 1px solid rgba(255, 43, 214, 0.55) !important;
        background: linear-gradient(135deg, rgba(255, 43, 214, 0.25), rgba(177, 0, 255, 0.18)) !important;
        color: var(--text) !important;
        box-shadow: 0 0 0 rgba(255, 43, 214, 0.0);
        transition: box-shadow 120ms ease, transform 120ms ease, border-color 120ms ease;
    }
    .stButton > button:hover {
        border-color: rgba(255, 43, 214, 0.90) !important;
        box-shadow: 0 0 0 3px rgba(255, 43, 214, 0.20), 0 0 22px rgba(255, 43, 214, 0.28);
        transform: translateY(-1px);
    }

    /* Inputs */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] > div,
    .stTextInput input, .stTextArea textarea, .stSelectbox div, .stMultiSelect div {
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid rgba(255, 43, 214, 0.22) !important;
        border-radius: 10px !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: var(--panel) !important;
        border: 1px solid rgba(255, 43, 214, 0.18) !important;
        border-radius: 12px !important;
        box-shadow: var(--shadow);
    }
    [data-testid="stExpander"] summary {
        color: var(--text) !important;
    }

    /* Progress bar */
    [data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, var(--magenta), var(--magenta-2)) !important;
        box-shadow: 0 0 18px rgba(255, 43, 214, 0.25);
    }
    [data-testid="stProgress"] > div > div {
        background: rgba(255, 255, 255, 0.10) !important;
    }

    /* Streamlit status boxes */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 43, 214, 0.18) !important;
        background: rgba(15, 15, 22, 0.9) !important;
        box-shadow: var(--shadow);
    }
    [data-testid="stAlert"] * {
        color: var(--text) !important;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--magenta);
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 18px rgba(255, 43, 214, 0.25);
    }
    .sub-header {
        text-align: center;
        color: var(--muted);
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(180deg, rgba(255, 43, 214, 0.08), rgba(15, 15, 22, 0.95));
        border: 1px solid rgba(255, 43, 214, 0.22);
        border-left: 5px solid var(--magenta);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    .treatment-card {
        background: linear-gradient(180deg, rgba(255, 204, 0, 0.10), rgba(15, 15, 22, 0.95));
        border: 1px solid rgba(255, 43, 214, 0.18);
        border-left: 5px solid var(--warning);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    .pesticide-card {
        background: linear-gradient(180deg, rgba(46, 242, 178, 0.08), rgba(15, 15, 22, 0.95));
        border: 1px solid rgba(255, 43, 214, 0.18);
        border-left: 5px solid var(--success);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    .severity-card {
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 43, 214, 0.18);
        background: rgba(15, 15, 22, 0.85);
        box-shadow: var(--shadow);
    }
    .quality-warning {
        background: linear-gradient(180deg, rgba(255, 204, 0, 0.12), rgba(15, 15, 22, 0.95));
        border: 1px solid rgba(255, 43, 214, 0.18);
        border-left: 5px solid var(--warning);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    .perf-badge {
        text-align: right;
        color: var(--muted);
        font-size: 0.85rem;
    }

    /* KPI cards (avoid Streamlit metric ellipsis) */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin-top: 8px;
    }
    @media (max-width: 1100px) {
        .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    .kpi-card {
        background: linear-gradient(180deg, rgba(255, 43, 214, 0.06), rgba(15, 15, 22, 0.92));
        border: 1px solid rgba(255, 43, 214, 0.18);
        border-radius: 14px;
        padding: 14px 14px 12px;
        box-shadow: var(--shadow);
        min-height: 86px;
    }
    .kpi-title {
        color: var(--muted);
        font-size: 0.92rem;
        margin: 0 0 6px 0;
        letter-spacing: 0.2px;
    }
    .kpi-value {
        color: var(--text);
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.25;
        overflow-wrap: anywhere; /* ensure no ... */
        white-space: normal;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">🌿 Plant Disease Detector</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">PlantVillage Disease Classification '
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

if "analysis" not in st.session_state:
    st.session_state.analysis = None

# ------------------------------------------------------------------
# Professional layout: Sidebar controls + main dashboard
# ------------------------------------------------------------------
st.sidebar.markdown("## Control Panel")
st.sidebar.caption("Upload a leaf image and run analysis.")

uploaded_file = st.sidebar.file_uploader(
    "Leaf image",
    type=["jpg", "jpeg", "png"],
    help="Use a clear, well-lit image where the leaf fills most of the frame.",
    key="leaf_upload",
)

pil_image = None
is_good = False
quality_msg = ""
quality_metrics = None

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    is_good, quality_msg, quality_metrics = check_image_quality(pil_image)

    with st.sidebar.expander("Image quality", expanded=False):
        if quality_metrics:
            qm = quality_metrics
            st.metric("Sharpness", f"{qm['blur_score']:.1f}")
            st.metric("Brightness", f"{qm['mean_brightness']:.0f} / 255")
            st.metric("Resolution", f"{qm['width']}×{qm['height']} px")
        if not is_good:
            st.warning(quality_msg)

run_disabled = (uploaded_file is None) or (not is_good)
run_clicked = st.sidebar.button(
    "Analyze",
    type="primary",
    use_container_width=True,
    disabled=run_disabled,
)

with st.sidebar.expander("Model details", expanded=False):
    st.markdown(f"""
    - **Classes**: {len(class_names)}
    - **Architecture**: EfficientNet-B0
    - **Explainability**: Grad-CAM
    - **Confidence**: Temperature scaling (optional)
    """)

if run_clicked and pil_image is not None and is_good:
    with st.spinner("Analyzing plant health ..."):
        (prediction, raw_conf, cal_conf,
         gradcam_img, heatmap, elapsed_ms) = predict_and_explain(
            pil_image, model, class_names, scaler
        )

    severity = None
    if "healthy" not in prediction.lower():
        severity = estimate_severity(heatmap, pil_image)

    st.session_state.analysis = {
        "prediction": prediction,
        "display_name": prediction.replace("___", " — ").replace("_", " "),
        "raw_conf": raw_conf,
        "cal_conf": cal_conf,
        "confidence": cal_conf,
        "gradcam_img": gradcam_img,
        "heatmap": heatmap,
        "elapsed_ms": elapsed_ms,
        "severity": severity,
        "info": get_treatment_info(prediction),
        "is_healthy": "healthy" in prediction.lower(),
        "pil_image": pil_image,
    }

# -----------------------------
# Main dashboard content
# -----------------------------
analysis = st.session_state.analysis

top = st.container()
with top:
    if uploaded_file is None:
        st.markdown("### Upload an image to get started")
        st.info(
            "Use the **Control Panel** on the left to upload a leaf image, "
            "then click **Analyze** to generate predictions, severity, Grad-CAM, and treatment guidance."
        )
    else:
        st.markdown("### Current image")
        st.image(pil_image, caption="Uploaded image", use_container_width=True)

if analysis is None:
    st.markdown("---")
    st.markdown("### What you’ll get")
    st.markdown("""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Prediction</div>
            <div class="kpi-value">39 classes</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Confidence</div>
            <div class="kpi-value">Calibrated optional</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Explainability</div>
            <div class="kpi-value">Grad CAM</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Guidance</div>
            <div class="kpi-value">Treatment and pesticide</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("---")

    # KPI row
    k1, k2, k3, k4 = st.columns([2, 1, 1, 1])
    with k1:
        st.markdown(f"### {analysis['display_name']}")
    with k2:
        st.metric("Confidence", f"{analysis['confidence'] * 100:.1f} %")
    with k3:
        if analysis["severity"] is None:
            st.metric("Severity", "—")
        else:
            st.metric("Severity", analysis["severity"]["severity"])
    with k4:
        st.metric("Inference", f"{analysis['elapsed_ms']:.0f} ms")

    if analysis["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            f"Low confidence ({analysis['confidence'] * 100:.1f} %). "
            "Consider uploading a clearer image or consulting an expert."
        )

    tab_overview, tab_explain, tab_treat, tab_cal = st.tabs(
        ["Overview", "Explainability", "Treatment", "Calibration"]
    )

    with tab_overview:
        cal_label = " (calibrated)" if scaler.fitted else ""
        st.markdown(f"""
        <div class="result-card">
            <h4>🔬 Detection Summary</h4>
            <p><strong>Disease:</strong> {analysis['display_name']}</p>
            <p><strong>Confidence:</strong> {analysis['confidence'] * 100:.1f} %{cal_label}</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(analysis["confidence"])

        if analysis["is_healthy"]:
            st.success("Plant appears healthy.")
        else:
            st.warning("Disease detected. Review severity and treatment guidance.")

        if analysis["severity"] is not None:
            sev = analysis["severity"]
            sev_icon = (
                '🟢' if sev['severity'] == 'Mild'
                else '🟠' if sev['severity'] == 'Moderate'
                else '🔴'
            )
            st.markdown("#### Severity")
            st.markdown(f"""
            <div class="severity-card" style="border-left: 5px solid {sev['color']};
                 background-color: {sev['color']}15;">
                <h4 style="color: {sev['color']};">
                    {sev_icon} Severity: {sev['severity']}
                </h4>
                <p><strong>Estimated infected area:</strong> {sev['infected_pct']} %</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(sev["infected_pct"] / 100.0, 1.0))

    with tab_explain:
        st.markdown("#### Grad-CAM")
        gc_left, gc_right = st.columns(2)
        with gc_left:
            st.image(analysis["pil_image"], caption="Original", use_container_width=True)
        with gc_right:
            st.image(analysis["gradcam_img"], caption="Grad-CAM overlay", use_container_width=True)

    with tab_treat:
        info = analysis["info"]
        st.markdown("#### Treatment recommendation")
        st.markdown(f"""
        <div class="treatment-card">
            <h4>📋 Description</h4>
            <p>{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Recommended steps**")
        for step in info["treatment"]:
            st.markdown(f"- {step}")

        st.markdown(f"""
        <div class="pesticide-card">
            <h4>💊 Suggested product</h4>
            <p>{info['pesticide']}</p>
        </div>
        """, unsafe_allow_html=True)

    with tab_cal:
        st.markdown("#### Confidence calibration")
        # Show any pending success message carried over from a calibration rerun
        _cal_msg = st.session_state.pop("_cal_success", None)
        if _cal_msg:
            st.success(_cal_msg)
        if scaler.fitted:
            st.success(f"Temperature loaded: T = {scaler.temperature:.4f}")
        else:
            st.info("No calibration file found — using raw softmax (T = 1.0).")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Raw confidence", f"{analysis['raw_conf'] * 100:.1f} %")
        with c2:
            st.metric("Calibrated confidence", f"{analysis['cal_conf'] * 100:.1f} %")

        with st.expander("Fit temperature from local validation logits", expanded=False):
            default_logits = os.path.join(SCRIPT_DIR, "val_logits.npy")
            default_labels = os.path.join(SCRIPT_DIR, "val_labels.npy")

            logits_path = st.text_input(
                "Logits file path (.npy, shape Nx39)",
                value=default_logits,
                key="logits_path_input",
            )
            labels_path = st.text_input(
                "Labels file path (.npy, shape N)",
                value=default_labels,
                key="labels_path_input",
            )

            logits_exists = os.path.exists(logits_path)
            labels_exists = os.path.exists(labels_path)

            if logits_exists and labels_exists:
                st.success("Found both local files. Ready to fit.")
            else:
                missing = []
                if not logits_exists:
                    missing.append(f"`{logits_path}`")
                if not labels_exists:
                    missing.append(f"`{labels_path}`")
                st.warning("Missing file(s): " + ", ".join(missing))

            if st.button("Fit Temperature From Local Files", key="fit_temp_btn_local"):
                if not (logits_exists and labels_exists):
                    st.error("Cannot fit temperature: required files are missing.")
                else:
                    try:
                        logits_np = np.load(logits_path)
                        labels_np = np.load(labels_path)
                        logits_t = torch.from_numpy(logits_np).float()
                        labels_t = torch.from_numpy(labels_np).long()

                        new_scaler = TemperatureScaler()
                        T = new_scaler.fit(logits_t, labels_t)
                        new_scaler.save(TEMP_SCALE_PATH)

                        # Update the live scaler so the change takes effect
                        # immediately without reloading the page
                        scaler.temperature = new_scaler.temperature
                        scaler.fitted = True
                        load_temperature_scaler.clear()

                        # Recalculate calibrated confidence for current image
                        with torch.no_grad():
                            image_tensor = TRANSFORM(analysis["pil_image"]).unsqueeze(0)
                            outputs = model(image_tensor)
                            cal_probs = scaler.calibrated_softmax(outputs[0])
                            pred_idx = outputs.argmax(dim=1).item()
                            analysis["cal_conf"] = cal_probs[pred_idx].item()
                            analysis["confidence"] = analysis["cal_conf"]

                        # Store message in session state so it survives the rerun
                        # (st.success() placed before st.rerun() is never rendered)
                        st.session_state["_cal_success"] = (
                            f"✅ Temperature fitted!  T = {T:.4f} — "
                            f"calibrated confidence: {analysis['cal_conf'] * 100:.1f} %"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Calibration failed: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #b9b9d6;">
    <p>🌱 <strong>Plant Disease Detection Project</strong></p>
    <p>Plant Disease Detection using Deep Learning — EfficientNet-B0 + Grad-CAM</p>
</div>
""", unsafe_allow_html=True)
