import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# 1. Clean Title
st.title("🌿 Phytosense AI")
st.caption("Anonymous Dashboard for Double-Blind Peer Review.")

# 2. Upload & Preprocess
uploaded_file = st.file_uploader("Upload a Leaf Sample (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Sample", width=300)

    # Resize to the exact input shape your model was trained on (e.g., 224x224 or 256x256)
    target_size = (224, 224) 
    img_resized = image.resize(target_size)
    
    # Normalize pixel values
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    raw_preds = model.predict(img_array)[0]

    # If raw_preds values are outside [0, 1] or don't sum to ~1.0, apply softmax:
    if not np.isclose(np.sum(raw_preds), 1.0, atol=1e-2):
        probabilities = tf.nn.softmax(raw_preds).numpy()
    else:
        probabilities = raw_preds

    top_idx = np.argmax(probabilities)
    top_conf = probabilities[top_idx] * 100

    # Display Results
    st.subheader("Diagnosis Results")
    if top_conf < 60.0:
        st.warning("⚠️ Confidence is below 60%. Image may not be a recognized leaf.")