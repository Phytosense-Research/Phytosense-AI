import time
import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ==========================================
# 1. PAGE CONFIG & STEALTH MODE (ANONYMITY)
# ==========================================
st.set_page_config(
    page_title="Phytosense AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit header, footer, and hamburger menu to protect peer review anonymity
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CLASS LABELS (PlantVillage Standard)
# ==========================================
CLASS_NAMES = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy',
    'Corn - Cercospora Leaf Spot Gray Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy',
    'Grape - Black Rot', 'Grape - Esca (Black Measles)', 'Grape - Leaf Blight (Isariopsis Leaf Spot)', 'Grape - Healthy',
    'Orange - Haunglongbing (Citrus Greening)', 'Peach - Bacterial Spot', 'Peach - Healthy',
    'Pepper, Bell - Bacterial Spot', 'Pepper, Bell - Healthy', 'Potato - Early Blight', 'Potato - Late Blight',
    'Potato - Healthy', 'Raspberry - Healthy', 'Soybean - Healthy', 'Squash - Powdery Mildew',
    'Strawberry - Leaf Scorch', 'Strawberry - Healthy', 'Tomato - Bacterial Spot', 'Tomato - Early Blight',
    'Tomato - Late Blight', 'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot',
    'Tomato - Spider Mites Two-spotted Spider Mite', 'Tomato - Target Spot',
    'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy'
]

# ==========================================
# 3. MODEL LOADER WITH CACHING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_selected_model(model_filename: str):
    """Loads and caches the model to ensure fast subsequent inferences."""
    start_load = time.perf_counter()
    
    # Try finding the model file in the root or common subdirectories
    possible_paths = [
        model_filename,
        os.path.join("models", model_filename),
        os.path.join("weights", model_filename)
    ]
    
    loaded_model = None
    for path in possible_paths:
        if os.path.exists(path):
            loaded_model = tf.keras.models.load_model(path, compile=False)
            break
            
    load_time_ms = (time.perf_counter() - start_load) * 1000
    return loaded_model, load_time_ms

# ==========================================
# 4. SIDEBAR: MODEL SELECTION & BENCHMARK
# ==========================================
st.sidebar.title("⚙️ Model Configuration")
st.sidebar.markdown("Select a deep learning architecture for real-time comparative inference.")

# Look for available model files in repository or use pre-configured names
# (Change these filenames to match the actual .h5 / .keras files in your GitHub repo)
available_models = {
    "Phytosense-Net (Proposed)": "phytosense_model.h5",
    "MobileNetV2 (Baseline)": "mobilenetv2_model.h5",
    "ResNet50 (Baseline)": "resnet50_model.h5"
}

# Allow fallback if files are named differently
repo_files = [f for f in os.listdir(".") if f.endswith((".h5", ".keras"))]
if repo_files:
    model_options = repo_files
    selected_model_name = st.sidebar.selectbox("Active Model Architecture:", model_options)
    model_file_to_load = selected_model_name
else:
    model_options = list(available_models.keys())
    selected_model_name = st.sidebar.selectbox("Active Model Architecture:", model_options)
    model_file_to_load = available_models[selected_model_name]

st.sidebar.divider()
st.sidebar.markdown("**Environment Information**")
st.sidebar.write(f"• TensorFlow: `{tf.__version__}`")
st.sidebar.write("• Acceleration: `CPU (Cloud Optimized)`")
st.sidebar.caption("Peer-Review Evaluation Dashboard")

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
st.title("🌿 Phytosense AI")
st.caption("Anonymous Dashboard for Double-Blind Peer Review.")

# Load the model
with st.spinner("Loading selected architecture into memory..."):
    model, load_time_ms = load_selected_model(model_file_to_load)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Input Leaf Sample")
    uploaded_file = st.file_uploader(
        "Upload a leaf image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload standard plant leaf image for disease diagnosis."
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Sample", use_container_width=True)

with col2:
    st.subheader("Diagnosis Results")
    
    if uploaded_file is not None:
        if model is None:
            st.error(
                f"Model file `{model_file_to_load}` not found in repository. "
                "Ensure your `.h5` or `.keras` model file is uploaded to GitHub."
            )
        else:
            # 1. Preprocessing
            input_shape = (224, 224)
            img_resized = image.resize(input_shape)
            img_array = np.array(img_resized, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

            # 2. Model Prediction & Latency Timing
            start_infer = time.perf_counter()
            raw_predictions = model.predict(img_array, verbose=0)[0]
            inference_time_ms = (time.perf_counter() - start_infer) * 1000

            # 3. Softmax Check (Avoid double-softmax while ensuring proper scale)
            if not np.isclose(np.sum(raw_predictions), 1.0, atol=1e-2):
                probabilities = tf.nn.softmax(raw_predictions).numpy()
            else:
                probabilities = raw_predictions

            # 4. Top-3 Prediction Extraction
            top_3_indices = np.argsort(probabilities)[::-1][:3]
            top_conf = probabilities[top_3_indices[0]] * 100

            # 5. Latency Metric Display
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Inference Time", f"{inference_time_ms:.1f} ms")
            with m_col2:
                st.metric("Model Load Time", f"{load_time_ms:.1f} ms")

            st.divider()

            # 6. Confidence Warning
            if top_conf < 60.0:
                st.warning("⚠️ Confidence is below 60%. Image may not be a recognized leaf.")

            # 7. Diagnostic Breakdown
            for idx in top_3_indices:
                class_label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
                conf_score = float(probabilities[idx])
                
                st.markdown(f"**{class_label}**: `{conf_score * 100:.1f}%`")
                st.progress(conf_score)
                st.write("")
    else:
        st.info("👈 Please upload a leaf image from the left panel to run diagnostic classification.")