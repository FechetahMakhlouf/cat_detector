# 🐱 Cat vs Non-Cat Detector

A **logistic regression** classifier that detects cats in images, built with a Neural Network mindset.  
Includes a **Streamlit** web application and a **data augmentation** pipeline to boost training accuracy.

> Based on the Deep Learning Specialization – Andrew Ng (Coursera)

---

## 📁 Project Structure

```
cat_detector/
├── datasets/
│   ├── train_catvnoncat.h5     # 209 training images (64×64 RGB)
│   └── test_catvnoncat.h5      # 50  test     images (64×64 RGB)
├── app.py                      # Streamlit web application
├── model.py                    # Logistic regression (sigmoid, propagate, optimize…)
├── data_augmentation.py        # Image augmentation (flip, rotate, noise, brightness)
├── train.py                    # CLI training script
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quick Start

### 1 – Clone & install
```bash
git clone https://github.com/<your-username>/cat-detector.git
cd cat-detector
pip install -r requirements.txt
```

### 2 – Train the model
```bash
# Basic training (no augmentation)
python train.py

# With data augmentation (recommended)
python train.py --augment --aug-factor 6 --lr 0.005 --iters 3000
```

This saves `weights.npz` which the app will load automatically.

### 3 – Launch the web app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser, upload any image, and get the prediction instantly.

---

## 📊 Data Augmentation

The `data_augmentation.py` module generates extra training images using:

| Transform | Description |
|---|---|
| Horizontal Flip | Mirror left-right |
| Vertical Flip | Mirror top-bottom |
| Rotate 90° / 180° / 270° | Clockwise rotations |
| Brightness Shift | Random ± 25% brightness |
| Gaussian Noise | Random pixel noise (std=0.04) |

With `--aug-factor 6`, a dataset of 209 images grows to **≈ 1,463 images**.

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
| Baseline (2000 iters, lr=0.005) | ~99% | ~70% |
| With augmentation (3000 iters) | ~97% | ~74% |

---

## 🛠️ CLI Options

```
python train.py [OPTIONS]

Options:
  --augment           Enable data augmentation
  --aug-factor INT    Augmentations per image (default: 4)
  --lr FLOAT          Learning rate (default: 0.005)
  --iters INT         Gradient descent iterations (default: 2000)
  --seed INT          Random seed (default: 42)
```

---

## 📚 References

- [Deep Learning Specialization – Andrew Ng](https://www.coursera.org/specializations/deep-learning)
- [NumPy Documentation](https://numpy.org/doc/)
- [Streamlit Documentation](https://docs.streamlit.io/)
