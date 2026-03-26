<div align="center">

<h1>
  <br>
  <img src="https://img.shields.io/badge/🐱-Cat%20Detector-f97316?style=for-the-badge&labelColor=1e293b" alt="Cat Detector Logo"/>
  <br>
  Cat vs Non-Cat Detector
  <br>
</h1>

<p align="center">
  <strong>Upload any image. Know if there's a cat. Instantly.</strong><br>
  A logistic regression classifier built with a Neural Network mindset — deployed and live.
</p>

<p align="center">
  <a href="https://cat-detector-tit1.onrender.com">
    <img src="https://img.shields.io/badge/🌐 Live Demo-cat--detector--tit1.onrender.com-f97316?style=flat-square" alt="Live Demo"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Streamlit-1.x-ff4b4b?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  &nbsp;
  <img src="https://img.shields.io/badge/NumPy-Powered-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Deployed-Render-46e3b7?style=flat-square" alt="Render"/>
</p>

> Based on the Deep Learning Specialization – Andrew Ng (Coursera)

---

</div>

## 🌟 Aperçu

**Cat Detector** is a logistic regression classifier that detects cats in images, built with a Neural Network mindset. It includes a **Streamlit** web application and a **data augmentation** pipeline to boost training accuracy.

> 🔗 **Live Demo:** [cat-detector-tit1.onrender.com](https://cat-detector-tit1.onrender.com)

## 🚀 Quick Start

### Option 1 – Use the live app ☁️

Head directly to **[cat-detector-tit1.onrender.com](https://cat-detector-tit1.onrender.com)**, upload any image, and get the prediction instantly — no setup required.

### Option 2 – Run locally 🖥️

#### 1 – Clone & install
```bash
git clone https://github.com/<your-username>/cat-detector.git
cd cat-detector
pip install -r requirements.txt
```

#### 2 – Train the model
```bash
python train.py --lr 0.005 --iters 3000
```

This saves `weights.npz` which the app will load automatically.

#### 3 – Launch the web app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Model Architecture

```
Input image (64×64×3)
       │
       ▼
  Flatten → vector x of size 12,288
       │
       ▼
  ŷ = σ(wᵀx + b)     σ = sigmoid
       │
       ▼
  ŷ > 0.5 → Cat 🐱   else → Not a cat 🚫
```

**Cost function:** Binary Cross-Entropy  
**Optimizer:** Gradient Descent  
**Parameters:** w (12,288 × 1) + b (scalar) = 12,289 trainable values

---

## 📈 Results

| Setting | Train Acc | Test Acc |
|---|---|---|
| Baseline (2000 iters, lr=0.005) | ~94.6% | ~76.2% |
| With augmentation (3000 iters) | ~97% | ~78% |

---

## 🛠️ CLI Options

```
python train.py [OPTIONS]

Options:
  --lr FLOAT          Learning rate (default: 0.005)
  --iters INT         Gradient descent iterations (default: 2000)
  --seed INT          Random seed (default: 42)
```

---

## 📚 References

- [Deep Learning Specialization – Andrew Ng](https://www.coursera.org/specializations/deep-learning)
- [NumPy Documentation](https://numpy.org/doc/)
- [Streamlit Documentation](https://docs.streamlit.io/)