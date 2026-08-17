import os
import time
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

# Protect Double-Blind Peer Review anonymity by hiding default Streamlit menus
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
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
# 3. RECURSIVE MODEL SEARCH
# ==========================================
def scan_for_models():
    """Recursively finds all model files in the repo (.h5, .keras, .tflite)."""
    valid_exts = ('.h5', '.keras', '.tflite')
    ignore_dirs = {'.git', '.venv', 'venv', '__pycache__', '.streamlit'}
    found_models = {}

    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            if file.endswith(valid_exts):
                full_path = os.path.relpath(os.path.join(root, file), '.')
                # Create a clean display label
                label = os.path.splitext(file)[0].replace('_', ' ').replace('-', ' ').title()
                if os.path.dirname(full_path):
                    label = f"{label} ({full_path})"
                found_models[label] = full_path

    return found_models

# ==========================================
# 4. MODEL LOADER & RUNNER
# ==========================================
@st.cache_resource(show_spinner=False)
def load_model_from_path(model_path: str):
    """Loads and caches either a Keras model (.h5/.keras) or TFLite interpreter."""
    if not model_path or not os.path.exists(model_path):
        return None, 0.0, "not_found"

    start_load = time.perf_counter()
    try:
        if model_path.endswith('.tflite'):
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            load_time_ms = (time.perf_counter() - start_load) * 1000
            return interpreter, load_time_ms, "tflite"
        else:
            model = tf.keras.models.load_model(model_path, compile=False)
            load_time_ms = (time.perf_counter() - start_load) * 1000
            return model, load_time_ms, "keras"
    except Exception as e:
        return None, 0.0, str(e)

def run_inference(model_obj, model_type: str, img_array: np.ndarray):
    """Executes inference and returns raw predictions along with latency."""
    start_infer = time.perf_counter()
    
    if model_type == "tflite":
        input_details = model_obj.get_input_details()
        output_details = model_obj.get_output_details()
        
        # Check if model expects quantized input
        if input_details[0]['dtype'] in [np.uint8, np.int8]:
            scale, zero_point = input_details[0]['quantization']
            if scale > 0:
                input_data = (img_array / scale + zero_point).astype(input_details[0]['dtype'])
            else:
                input_data = (img_array * 255.0).astype(input_details[0]['dtype'])
        else:
            input_data = img_array.astype(np.float32)

        model_obj.set_tensor(input_details[0]['index'], input_data)
        model_obj.invoke()
        raw_output = model_obj.get_tensor(output_details[0]['index'])[0]

        # Dequantize output if necessary
        if output_details[0]['dtype'] in [np.uint8, np.int8]:
            o_scale, o_zero = output_details[0]['quantization']
            if o_scale > 0:
                raw_output = (raw_output.astype(np.float32) - o_zero) * o_scale
    else:
        raw_output = model_obj.predict(img_array, verbose=0)[0]

    infer_time_ms = (time.perf_counter() - start_infer) * 1000
    return raw_output, infer_time_ms

# ==========================================
# 5. SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.title("⚙️ Model Configuration")
st.sidebar.markdown("Select a deep learning architecture for real-time comparative inference.")

available_models = scan_for_models()

if available_models:
    selected_label = st.sidebar.selectbox("Active Model Architecture:", list(available_models.keys()))
    selected_model_path = available_models[selected_label]
else:
    st.sidebar.warning("⚠️ No `.h5`, `.keras`, or `.tflite` model files discovered in the repository.")
    selected_model_path = st.sidebar.text_input("Enter exact model file path manually:", "model.h5")

st.sidebar.divider()
st.sidebar.markdown("**Environment Information**")
st.sidebar.write(f"• TensorFlow: `{tf.__version__}`")
st.sidebar.write("• Acceleration: `CPU (Cloud Optimized)`")
st.sidebar.caption("Peer-Review Evaluation Dashboard")

# ==========================================
# 6. MAIN USER INTERFACE
# ==========================================
st.title("🌿 Phytosense AI")
st.caption("Anonymous Dashboard for Double-Blind Peer Review.")

# Load active model
with st.spinner("Loading selected architecture into memory..."):
    model_obj, load_time_ms, model_status = load_model_from_path(selected_model_path)

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
        if model_obj is None:
            st.error(
                f"Could not load model at `{selected_model_path}`. "
                f"Details: {model_status}"
            )
        else:
            # 1. Preprocess input image (resize to 224x224 and normalize to [0, 1])
            target_shape = (224, 224)
            img_resized = image.resize(target_shape)
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # 2. Run Inference
            raw_predictions, inference_time_ms = run_inference(model_obj, model_status, img_array)

            # 3. Softmax Check (Ensures confidence is not flattened)
            if not np.isclose(np.sum(raw_predictions), 1.0, atol=1e-2):
                probabilities = tf.nn.softmax(raw_predictions).numpy()
            else:
                probabilities = raw_predictions

            # 4. Extract Top-3 Predictions
            top_3_indices = np.argsort(probabilities)[::-1][:3]
            top_conf = probabilities[top_3_indices[0]] * 100

            # 5. Display Latency Metrics
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Inference Time", f"{inference_time_ms:.1f} ms")
            with m_col2:
                st.metric("Model Load Time", f"{load_time_ms:.1f} ms")

            st.divider()

            # 6. Confidence Alert
            if top_conf < 60.0:
                st.warning("⚠️ Confidence is below 60%. Image may not be a recognized leaf.")

            # 7. Diagnostic Breakdown Bars
            for idx in top_3_indices:
                class_label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
                conf_score = float(probabilities[idx])

                st.markdown(f"**{class_label}**: `{conf_score * 100:.1f}%`")
                st.progress(conf_score)
                st.write("")
    else:
        st.info("👈 Please upload a leaf image from the left panel to run diagnostic classification.")