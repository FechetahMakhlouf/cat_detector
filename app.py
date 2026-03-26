import os
import numpy as np
import streamlit as st
from PIL import Image

from model import sigmoid, load_dataset, model as train_model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cat Detector",
    page_icon="🐱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

    /* ══════════════════════════════════════════
       THEME VARIABLES — dark (default)
    ══════════════════════════════════════════ */
    :root {
        --bg-main:       #0a0a0f;
        --bg-card:       #13131f;
        --bg-sidebar:    #0d0d18;
        --text-primary:  #e8e8f0;
        --text-muted:    #888;
        --text-dim:      #c8c8d8;
        --border:        #1e1e30;
        --border-dim:    #2a2a40;
        --accent:        #ff9f43;
        --shadow:        rgba(0,0,0,0.4);
    }

    /* ── Light mode overrides — triggered by JS via data-theme on <html> ── */
    html[data-app-theme="light"] {
        --bg-main:       #f5f5fa;
        --bg-card:       #ffffff;
        --bg-sidebar:    #eeeef5;
        --text-primary:  #1a1a2e;
        --text-muted:    #666;
        --text-dim:      #333;
        --border:        #d0d0e0;
        --border-dim:    #b0b0cc;
        --accent:        #e07b20;
        --shadow:        rgba(0,0,0,0.12);
    }

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stVerticalBlock"],
    section.main > div {
        background: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }

    /* Hide only streamlit branding, keep all controls visible */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header {
        visibility: visible !important;
        background: transparent !important;
        position: absolute !important;
    }
    header * { visibility: visible !important; }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    [data-testid="stAppViewContainer"] > section > div {
        padding-top: 0rem !important;
    }

    /* ── Hero header ── */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        position: relative;
    }
    .hero-icon {
        font-size: 3.5rem;
        display: block;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 0 20px rgba(255,180,100,0.5));
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50%       { transform: translateY(-8px); }
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ff9f43, #ff6b6b, #a29bfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .hero-sub {
        margin-top: 0.4rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-muted);
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    /* ── Upload zone ── */
    .upload-label {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    [data-testid="stFileUploader"] {
        background: var(--bg-card);
        border: 1.5px dashed var(--border-dim);
        border-radius: 14px;
        padding: 0.5rem;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent);
    }

    /* ── Result cards ── */
    .result-wrapper {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--border);
        margin-top: 0.5rem;
    }
    .verdict-cat {
        font-family: 'Space Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #55efc4;
        text-align: center;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .verdict-nocat {
        font-family: 'Space Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #fd79a8;
        text-align: center;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .verdict-emoji {
        font-size: 2.8rem;
        text-align: center;
        display: block;
        margin: 0.5rem 0;
    }
    .confidence-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 1rem 0 0.3rem;
    }
    .conf-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.25rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.82rem;
        color: var(--text-primary);
    }
    .conf-pct { color: var(--text-dim); }

    /* ── Progress bars ── */
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #ff9f43, #ff6b6b) !important;
        border-radius: 4px !important;
    }
    [data-testid="stProgress"] > div {
        background: var(--border) !important;
        border-radius: 4px !important;
        height: 6px !important;
    }

    /* ── Stats cards ── */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin: 1.5rem 0;
    }
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .stat-card:hover { border-color: var(--accent); }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: var(--accent);
        font-family: 'Space Mono', monospace;
    }
    .stat-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text-dim) !important; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] a { color: var(--accent) !important; text-decoration: underline; }
    .sidebar-badge {
        display: inline-block;
        background: var(--border);
        border: 1px solid var(--border-dim);
        border-radius: 6px;
        padding: 0.2rem 0.5rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: var(--accent) !important;
        margin: 0.15rem 0;
    }

    /* ── Author card (sidebar footer) ── */
    .author-card {
        margin-top: 0.5rem;
        border-radius: 14px;
        padding: 1rem 1rem 0.85rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .author-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ff9f43, #ff6b6b, #a29bfe);
    }
    .author-avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: linear-gradient(135deg, #ff9f43, #a29bfe);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        margin: 0 auto 0.6rem;
        box-shadow: 0 4px 14px rgba(255,159,67,0.35);
    }
    .author-name {
        font-family: 'Syne', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.01em;
        color: var(--text-primary) !important;
        margin-bottom: 0.15rem;
    }
    .author-role {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.62rem !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-muted) !important;
        margin-bottom: 0.65rem;
    }
    .author-link {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        color: #fff !important;
        text-decoration: none !important;
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        transition: opacity 0.2s, transform 0.2s;
        cursor: pointer;
    }
    .author-link:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }
    .author-course {
        margin-top: 0.65rem;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.6rem !important;
        color: var(--text-muted) !important;
        letter-spacing: 0.04em;
        line-height: 1.5;
    }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: var(--accent) !important; }

    /* ── Image display ── */
    [data-testid="stImage"] img {
        border-radius: 12px !important;
        border: 1px solid var(--border);
    }

    /* ── Section heading ── */
    .section-heading {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 1.5rem 0 0.75rem;
    }
</style>

<!-- ══ JS THEME DETECTOR ══════════════════════════════════════════════════════
     Streamlit doesn't expose data-theme on <html>; we detect it by sampling
     the actual background colour that Streamlit injects and set our own
     data-app-theme attribute so our CSS variables kick in correctly.
════════════════════════════════════════════════════════════════════════════ -->
<script>
(function () {
    function detectAndApply() {
        // Streamlit paints its own background on stApp; sample it.
        var appEl = document.querySelector('[data-testid="stAppViewContainer"]')
                 || document.querySelector('.stApp')
                 || document.body;
        var bg = window.getComputedStyle(appEl).backgroundColor;
        var rgb = bg.match(/\d+/g);
        if (!rgb) return;
        var brightness = (parseInt(rgb[0]) * 299 +
                          parseInt(rgb[1]) * 587 +
                          parseInt(rgb[2]) * 114) / 1000;
        // > 128 → light background → light theme
        document.documentElement.setAttribute(
            'data-app-theme', brightness > 128 ? 'light' : 'dark'
        );
    }

    // Run immediately and on every change Streamlit makes to the DOM
    detectAndApply();
    var mo = new MutationObserver(detectAndApply);
    mo.observe(document.documentElement, { attributes: true, subtree: false });
    mo.observe(document.body,            { attributes: true, childList: true, subtree: false });

    // Fallback polling — Streamlit may update styles outside DOM mutations
    setInterval(detectAndApply, 600);
})();
</script>
""", unsafe_allow_html=True)


# ── Load / Train model ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model weights…")
def get_model():
    weights_path = "weights.npz"

    if os.path.exists(weights_path):
        data = np.load(weights_path, allow_pickle=True)
        w = data["w"]
        b = float(data["b"][0])
        num_px = int(data["num_px"][0])
        classes = data["classes"]
        return w, b, num_px, classes

    # Fallback: train from scratch
    from model import load_dataset
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
    img = image.resize((num_px, num_px)).convert("RGB")
    arr = np.array(img) / 255.
    vec = arr.reshape(-1, 1)
    prob = float(sigmoid(np.dot(w.T, vec) + b).squeeze())
    label = 1 if prob > 0.5 else 0
    return prob, label


# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <span class="hero-icon">🐱</span>
  <p class="hero-title">Cat Detector</p>
  <p class="hero-sub">Logistic Regression · Neural Network Mindset</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Architecture")
    st.markdown("""
<span class="sidebar-badge">Logistic Regression</span><br>
<span class="sidebar-badge">Sigmoid Activation</span><br>
<span class="sidebar-badge">Gradient Descent</span><br>
<span class="sidebar-badge">12 288 features</span>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📐 Pipeline")
    steps = [
        "Resize to **64 × 64 px**",
        "Flatten → 12 288-dim vector",
        "Normalise to **[0, 1]**",
        "Forward pass: `ŷ = σ(wᵀx + b)`",
        "`ŷ > 0.5` → **Cat 🐱**",
    ]
    for i, s in enumerate(steps, 1):
        st.markdown(f"`{i}.` {s}")

    st.divider()

    # ── Author card ────────────────────────────────────────────────────────────
    st.markdown("""
<div class="author-card">
  <a class="author-link" href="https://fechetahmakhlouf.github.io/MyPortfolio/" target="_blank">
    🌐 Fechetah Makhlouf
  </a>
  <p class="author-course">
    Deep Learning Specialization<br>
    Andrew Ng · Coursera
  </p>
</div>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────

w, b, num_px, classes = get_model()

# ── Upload ─────────────────────────────────────────────────────────────────────

st.markdown('<p class="section-heading">📤 Upload an image</p>',
            unsafe_allow_html=True)
uploaded = st.file_uploader(
    "JPG or PNG, any size — resized automatically to 64×64",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.image(image, use_container_width=True)

    with col2:
        with st.spinner("Running inference…"):
            prob, label = predict_image(image, w, b, num_px)

        conf_cat = prob * 100
        conf_no_cat = (1 - prob) * 100

        if label == 1:
            verdict_cls = "verdict-cat"
            verdict_text = "CAT DETECTED"
            verdict_emoji = "✅"
        else:
            verdict_cls = "verdict-nocat"
            verdict_text = "NOT A CAT"
            verdict_emoji = "❌"

        st.markdown(f"""
<div class="result-wrapper">
  <span class="verdict-emoji">{verdict_emoji}</span>
  <p class="{verdict_cls}">{verdict_text}</p>
  <p class="confidence-label">Confidence</p>
  <div class="conf-row">
    <span>🐱 Cat</span>
    <span class="conf-pct">{conf_cat:.1f}%</span>
  </div>
</div>
""", unsafe_allow_html=True)
        st.progress(prob)

        st.markdown(f"""
<div style="margin-top:0.6rem">
  <div class="conf-row">
    <span>🚫 Non-Cat</span>
    <span class="conf-pct">{conf_no_cat:.1f}%</span>
  </div>
</div>
""", unsafe_allow_html=True)
        st.progress(1 - prob)

# ── Stats ──────────────────────────────────────────────────────────────────────

st.markdown('<p class="section-heading">📊 Model Performance</p>',
            unsafe_allow_html=True)
st.markdown("""
<div class="stats-row">
  <div class="stat-card">
    <div class="stat-value">~94.6%</div>
    <div class="stat-label">Train Accuracy</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">~76.2%</div>
    <div class="stat-label">Test Accuracy</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">64px</div>
    <div class="stat-label">Input Size</div>
  </div>
</div>
""", unsafe_allow_html=True)
