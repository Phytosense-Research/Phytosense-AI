import os
# NOTE: this env var alone does NOT disable XNNPACK for tf.lite.Interpreter —
# see load_model_from_path() below for the fix that actually works
# (experimental_op_resolver_type=BUILTIN_WITHOUT_DEFAULT_DELEGATES). Left set
# here too since it's harmless and some TF builds do respect it for other paths.
os.environ["TF_XNNPACK_DISABLE"] = "1"

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

# FIXED STEALTH CSS: Hides menus/GitHub links but keeps the Sidebar toggle button fully visible
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeaderActionElements"] {display: none;}
    
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

# Same threshold used in the Flutter app — kept identical on purpose so both
# platforms agree on what counts as "not a confident diagnosis".
CONFIDENCE_THRESHOLD = 0.60

# ==========================================
# 3. RECURSIVE MODEL SEARCH & CLEAN NAMING
# ==========================================
def clean_model_name(filename):
    """Formats raw filenames into clean, professional UI labels."""
    name = os.path.splitext(filename)[0]
    
    # Replace underscores and title case
    name = name.replace('_', ' ').title()
    
    # Remove the word 'Baseline' and clean up double spaces
    name = name.replace('Baseline', '').replace('  ', ' ').strip()
    
    # Capitalize specific technical terms
    name = name.replace('Int8', 'INT8')
    name = name.replace('Fp32', 'FP32')
    name = name.replace('Fp16', 'FP16')
    name = name.replace('Mobilenet', 'MobileNet')
    name = name.replace('Efficientnet', 'EfficientNet')
    name = name.replace('Nasnet', 'NASNet')
    name = name.replace('Vgg', 'VGG')
    name = name.replace('Resnet', 'ResNet')
    
    return name

def scan_for_models():
    """Finds all model files and returns a dictionary mapping Clean Names -> File Paths."""
    valid_exts = ('.h5', '.keras', '.tflite')
    ignore_dirs = {'.git', '.venv', 'venv', '__pycache__', '.streamlit'}
    found_models = {}

    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            if file.endswith(valid_exts):
                full_path = os.path.relpath(os.path.join(root, file), '.')
                clean_name = clean_model_name(file)
                found_models[clean_name] = full_path

    # Sort alphabetically so they look organized in the dropdown
    return dict(sorted(found_models.items()))

# ==========================================
# 4. MODEL LOADER & RUNNER (WITH MMAP FIX)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_model_from_path(model_path: str):
    if not model_path or not os.path.exists(model_path):
        return None, 0.0, "not_found"

    start_load = time.perf_counter()
    try:
        if model_path.endswith('.tflite'):
            # THE FIX: Read file as bytes into memory to completely bypass "mmap error 22"
            with open(model_path, 'rb') as f:
                model_bytes = f.read()

            # THE REAL XNNPACK FIX:
            # `experimental_delegates=[]` and the TF_XNNPACK_DISABLE env var only
            # control delegates you explicitly ADD — they do nothing to stop
            # TFLite's own default behavior of auto-attaching the XNNPACK
            # delegate at interpreter creation. That auto-attach is what was
            # crashing on MobileNetV3Small INT8 ("Node 117 ... failed to
            # prepare"), because XNNPACK doesn't support every quantized op
            # this model uses. `BUILTIN_WITHOUT_DEFAULT_DELEGATES` is the
            # documented switch that actually prevents that auto-attach, so
            # the interpreter falls back to TFLite's plain reference kernels.
            try:
                interpreter = tf.lite.Interpreter(
                    model_content=model_bytes,
                    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                )
            except AttributeError:
                # Older TF builds without this enum — fall back to the
                # previous behavior rather than crashing the whole loader.
                interpreter = tf.lite.Interpreter(model_content=model_bytes)

            interpreter.allocate_tensors()
            
            load_time_ms = (time.perf_counter() - start_load) * 1000
            return interpreter, load_time_ms, "tflite"
        else:
            model = tf.keras.models.load_model(model_path, compile=False)
            load_time_ms = (time.perf_counter() - start_load) * 1000
            return model, load_time_ms, "keras"
    except Exception as e:
        return None, 0.0, str(e)

# ==========================================
# 4b. ARCHITECTURE-AWARE PREPROCESSING
# ------------------------------------------
# This is the piece that was missing before. Every model family expects
# pixels prepared a different way, and the app must match whatever the
# Flutter build does for the SAME architecture or the two platforms will
# disagree on the same photo:
#   - VGG19            -> BGR channel order + ImageNet mean subtraction
#   - MobileNet/NASNet  -> scaled to [-1, 1]  ( x/127.5 - 1.0 )
#   - EfficientNet(V2)  -> raw 0-255 pixels, no rescaling (baked into the graph)
# ==========================================
def detect_family(label: str) -> str:
    l = label.lower()
    if "vgg" in l:
        return "vgg"
    if "mobilenet" in l or "nasnet" in l:
        return "tf_scale"
    # EfficientNet and anything else defaults to raw passthrough
    return "raw"

def preprocess_for_model(pil_image: Image.Image, target_size_wh, family: str) -> np.ndarray:
    """Resize + apply architecture-specific preprocessing.
    target_size_wh is (width, height), matching PIL's resize() convention."""
    # Explicit BILINEAR to match Flutter's Interpolation.linear — removes a
    # source of train/inference pixel drift caused by differing default
    # resize filters between platforms/libraries.
    img_resized = pil_image.resize(target_size_wh, Image.Resampling.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)  # RGB, range 0-255

    if family == "vgg":
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        # Mirrors the Dart VGG19 branch exactly: channel 0 <- B-123.68,
        # channel 1 <- G-116.779, channel 2 <- R-103.939.
        ch0 = b - 123.68
        ch1 = g - 116.779
        ch2 = r - 103.939
        arr = np.stack([ch0, ch1, ch2], axis=-1)
    elif family == "tf_scale":
        arr = (arr / 127.5) - 1.0
    # "raw": leave arr as 0-255 float — matches EfficientNet's baked-in rescaling

    return np.expand_dims(arr, axis=0)

def run_inference(model_obj, model_type: str, img_array: np.ndarray):
    start_infer = time.perf_counter()
    
    if model_type == "tflite":
        input_details = model_obj.get_input_details()[0]
        output_details = model_obj.get_output_details()[0]
        
        dtype = input_details['dtype']
        
        # INT8 / UINT8 quantization handling
        if dtype in (np.int8, np.uint8):
            scale, zero_point = input_details['quantization']
            if scale > 0:
                quantized = np.round((img_array / scale) + zero_point)
            else:
                quantized = img_array
            # THE FIX: clip into the dtype's real range before casting.
            # Without this, values outside [-128,127] / [0,255] silently
            # wrap around on cast (e.g. 300 -> 44), producing garbage
            # predictions that look like a random, overconfident guess.
            lo, hi = (-128, 127) if dtype == np.int8 else (0, 255)
            input_data = np.clip(quantized, lo, hi).astype(dtype)
        else:
            input_data = img_array.astype(np.float32)

        model_obj.set_tensor(input_details['index'], input_data)
        model_obj.invoke()
        raw_output = model_obj.get_tensor(output_details['index'])[0]

        # INT8 Dequantization handling
        if output_details['dtype'] in (np.int8, np.uint8):
            o_scale, o_zero = output_details['quantization']
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
    st.sidebar.warning("⚠️ No `.h5`, `.keras`, or `.tflite` model files discovered.")
    selected_label = ""
    selected_model_path = ""

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
            st.error(f"Could not load model at `{selected_model_path}`. Details: {model_status}")
        else:
            # 1. Dynamically read the required image dimensions from the model to avoid shape errors
            if model_status == "tflite":
                input_shape = model_obj.get_input_details()[0]['shape']
                target_shape = (input_shape[2], input_shape[1])  # TFLite is [batch,H,W,C] -> PIL wants (W,H)
            else:
                # Pull the real input size off the Keras model instead of hardcoding 224,
                # falling back only if the model reports an unspecified (-1/None) dim.
                k_shape = model_obj.input_shape  # (None, H, W, C)
                h = k_shape[1] if k_shape[1] else 224
                w = k_shape[2] if k_shape[2] else 224
                target_shape = (w, h)

            # 2. Architecture-aware preprocessing (matches the Flutter app per-model)
            family = detect_family(selected_label)
            img_array = preprocess_for_model(image, target_shape, family)

            # 3. Run Inference
            raw_predictions, inference_time_ms = run_inference(model_obj, model_status, img_array)

            # 4. Bulletproof Softmax Trigger
            # Applies Softmax if probabilities look like raw Logits (negative values, numbers > 1.0, or sum not near 1.0)
            if np.min(raw_predictions) < 0 or np.max(raw_predictions) > 1.0 or not np.isclose(np.sum(raw_predictions), 1.0, atol=0.1):
                probabilities = tf.nn.softmax(raw_predictions).numpy()
            else:
                probabilities = raw_predictions

            # 5. Extract Top-3 Predictions
            top_3_indices = np.argsort(probabilities)[::-1][:3]
            top_conf = probabilities[top_3_indices[0]] * 100

            # 6. Display Latency Metrics
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Inference Time", f"{inference_time_ms:.1f} ms")
            with m_col2:
                st.metric("Model Load Time", f"{load_time_ms:.1f} ms")

            st.divider()

            # 7. Confidence Alert — same threshold and wording as the Flutter app
            if top_conf < (CONFIDENCE_THRESHOLD * 100):
                st.warning(
                    f"⚠️ Image not recognized or confidence too low "
                    f"(best guess {top_conf:.1f}%, below the {CONFIDENCE_THRESHOLD*100:.0f}% threshold)."
                )

            # 8. Diagnostic Breakdown Bars
            for idx in top_3_indices:
                class_label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
                conf_score = float(probabilities[idx])

                st.markdown(f"**{class_label}**: `{conf_score * 100:.1f}%`")
                st.progress(conf_score)
                st.write("")
    else:
        st.info("👈 Please upload a leaf image from the left panel to run diagnostic classification.")
