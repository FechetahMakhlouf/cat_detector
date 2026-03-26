"""
app.py  –  Cat vs Non-Cat Detector
===================================
Run with:
    streamlit run app.py
"""

import os
import io
import numpy as np
import streamlit as st
from PIL import Image

from model import sigmoid, load_dataset, model as train_model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cat Detector",
    page_icon="🐱",
    layout="centered",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { text-align:center; font-size:2.8rem; font-weight:700; color:#FF6B6B; }
    .subtitle    { text-align:center; color:#555; margin-bottom:2rem; }
    .result-cat  { text-align:center; font-size:2rem; color:#00C851; font-weight:bold; }
    .result-nocat{ text-align:center; font-size:2rem; color:#FF4444; font-weight:bold; }
    .confidence  { text-align:center; font-size:1.1rem; color:#555; }
    .info-box    { background:#f0f4ff; border-radius:8px; padding:1rem; margin:1rem 0; }
    div[data-testid="stProgress"] > div { background-color: #FF6B6B !important; }
</style>
""", unsafe_allow_html=True)


# ── Load / Train model ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔧 Training model on cat dataset…")
def get_model():
    """Load weights from disk if available, otherwise train from scratch."""
    weights_path = "weights.npz"

    if os.path.exists(weights_path):
        data = np.load(weights_path, allow_pickle=True)
        w = data["w"]
        b = float(data["b"][0])
        num_px = int(data["num_px"][0])
        classes = data["classes"]
        return w, b, num_px, classes

    # Train from scratch
    X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset()
    num_px = X_train_orig.shape[1]

    X_train = X_train_orig.reshape(X_train_orig.shape[0], -1).T / 255.
    X_test = X_test_orig.reshape(X_test_orig.shape[0],  -1).T / 255.

    result = train_model(X_train, Y_train, X_test, Y_test,
                         num_iterations=2000, learning_rate=0.005)
    w, b = result["w"], result["b"]

    np.savez(weights_path, w=w, b=np.array([b]),
             num_px=np.array([num_px]), classes=classes)
    return w, b, num_px, classes


def predict_image(image: Image.Image, w, b, num_px) -> tuple[float, int]:
    """Returns (confidence_cat, predicted_label)."""
    img = image.resize((num_px, num_px)).convert("RGB")
    arr = np.array(img) / 255.
    vec = arr.reshape(-1, 1)          # (num_px*num_px*3, 1)
    prob = float(sigmoid(np.dot(w.T, vec) + b).squeeze())
    label = 1 if prob > 0.5 else 0
    return prob, label


# ── UI ─────────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🐱 Cat vs Non-Cat Detector</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Logistic Regression with a Neural Network mindset</p>',
            unsafe_allow_html=True)

# Sidebar – model info
with st.sidebar:
    st.header("⚙️ Model Info")
    st.markdown("""
    **Architecture:** Logistic Regression  
    **Features:** Raw pixel values (64×64×3)  
    **Activation:** Sigmoid  
    **Optimizer:** Gradient Descent  
    """)
    st.divider()
    st.header("📊 How it works")
    st.markdown("""
    1. Image resized to **64×64 px**
    2. Pixels flattened → vector of **12 288** features
    3. Normalised to **[0, 1]**
    4. Forward pass: `ŷ = σ(wᵀx + b)`
    5. If `ŷ > 0.5` → **Cat** 🐱
    """)
    st.divider()
    st.caption("Fechetah Makhlouf ~ Deep Learning Specialization – Andrew Ng ~")

# Load model
w, b, num_px, classes = get_model()

# ── Upload section ─────────────────────────────────────────────────────────────
st.subheader("📤 Upload an image")
uploaded = st.file_uploader(
    "Choose a JPG / PNG image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Your image", use_container_width=True)

    with col2:
        with st.spinner("Analysing…"):
            prob, label = predict_image(image, w, b, num_px)

        conf_cat = prob * 100
        conf_no_cat = (1 - prob) * 100

        if label == 1:
            st.markdown(f'<p class="result-cat">✅ CAT DETECTED!</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="result-nocat">❌ NOT A CAT</p>',
                        unsafe_allow_html=True)

        st.markdown("---")
        st.write("**Confidence scores**")
        st.write(f"🐱 Cat: **{conf_cat:.1f}%**")
        st.progress(prob)
        st.write(f"🚫 Non-Cat: **{conf_no_cat:.1f}%**")
        st.progress(1 - prob)

# ── Demo section ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("Train Accuracy", "≈ 99%", help="On the 209-image training set")
col2.metric("Test Accuracy",  "≈ 70%", help="On the 50-image test set")
col3.metric("Image Size",     "64×64 px")

st.markdown("""
<div class="info-box">
💡 <b>Tip:</b> The model was trained on a small dataset (209 images). 
For better accuracy, enable <b>data augmentation</b> by running:
<code>python train.py --augment --aug-factor 6 --iters 3000</code>
then restart the app.
</div>
""", unsafe_allow_html=True)
