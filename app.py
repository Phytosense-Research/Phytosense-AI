import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import os

# --- STEALTH MODE FOR DOUBLE-BLIND REVIEW ---
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.set_page_config(page_title="Phytosense AI - Benchmarking", page_icon="🌿")
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title("🌿 Phytosense AI: Edge-Pathology Benchmark")
st.markdown("Anonymous Dashboard for Double-Blind Peer Review.")
st.divider()

# --- LOAD LABELS ---
@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

labels = load_labels()

# --- SIDEBAR: MODEL SELECTION ---
st.sidebar.header("Model Configuration")

architecture = st.sidebar.selectbox(
    "Select Architecture",
    ["MobileNetV2", "MobileNetV3 Small", "VGG19", "EfficientNetB0", "EfficientNetV2B0", "NASNetMobile"]
)

precision = st.sidebar.selectbox(
    "Select Precision",
    ["FP32 (Float32)", "FP16 (Float16)", "INT8 (Quantized)"]
)

# Exact mapping to match your files
file_map = {
    "MobileNetV2": "mobilenetv2",
    "MobileNetV3 Small": "mobilenetv3_small",
    "VGG19": "vgg19",
    "EfficientNetB0": "efficientnetb0",
    "EfficientNetV2B0": "efficientnetv2b0",
    "NASNetMobile": "nasnetmobile"
}

arch_prefix = file_map[architecture]
if precision == "FP32 (Float32)":
    model_filename = f"models/{arch_prefix}_baseline_fp32.tflite"
elif precision == "FP16 (Float16)":
    model_filename = f"models/{arch_prefix}_baseline_fp16.tflite"
else:
    model_filename = f"models/{arch_prefix}_int8.tflite"

# --- MAIN APP: IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Upload a Leaf Sample (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Sample", use_container_width=True)

    if st.button("Run Inference", type="primary"):
        if not os.path.exists(model_filename):
            st.error(f"File not found: {model_filename}. Please check your models folder.")
        else:
            with st.spinner(f"Running {architecture} ({precision})..."):
                
                # 1. Load TFLite Model
                start_load = time.time()
                interpreter = tf.lite.Interpreter(model_path=model_filename)
                interpreter.allocate_tensors()
                load_time = (time.time() - start_load) * 1000

                input_details = interpreter.get_input_details()[0]
                output_details = interpreter.get_output_details()[0]
                
                _, height, width, _ = input_details['shape']
                
                # 2. Preprocess Image
                resized_img = image.resize((width, height), Image.Resampling.BILINEAR)
                input_data = np.array(resized_img, dtype=np.float32)
                input_data = np.expand_dims(input_data, axis=0)

                # Normalize
                if "VGG" in architecture:
                    input_data[..., 0] -= 103.939
                    input_data[..., 1] -= 116.779
                    input_data[..., 2] -= 123.68
                else:
                    input_data = (input_data / 127.5) - 1.0

                # Handle INT8 Quantization scaling
                if input_details['dtype'] == np.int8:
                    scale, zero_point = input_details['quantization']
                    input_data = (input_data / scale) + zero_point
                    input_data = np.clip(np.round(input_data), -128, 127).astype(np.int8)

                # 3. Run Inference
                interpreter.set_tensor(input_details['index'], input_data)
                
                start_inf = time.time()
                interpreter.invoke()
                inf_time = (time.time() - start_inf) * 1000

                output_data = interpreter.get_tensor(output_details['index'])[0]

                # De-quantize if INT8
                if output_details['dtype'] == np.int8:
                    scale, zero_point = output_details['quantization']
                    output_data = (output_data.astype(np.float32) - zero_point) * scale

                # 4. Softmax & Results
                output_data = np.exp(output_data - np.max(output_data))
                probabilities = output_data / output_data.sum()
                
                top_3_indices = np.argsort(probabilities)[-3:][::-1]
                
                st.subheader("Diagnosis Results")
                
                if probabilities[top_3_indices[0]] < 0.60:
                    st.warning("⚠️ Confidence is below 60%. Image may not be a recognized leaf.")
                else:
                    st.success(f"**{labels[top_3_indices[0]]}**")

                col1, col2 = st.columns(2)
                col1.metric("Inference Time", f"{int(inf_time)} ms")
                col2.metric("Model Load Time", f"{int(load_time)} ms")

                for i in top_3_indices:
                    st.write(f"{labels[i]}: {probabilities[i]*100:.1f}%")
                    st.progress(float(probabilities[i]))