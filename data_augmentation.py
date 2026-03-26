"""
data_augmentation.py
====================
Augment the cat / non-cat dataset by applying simple image transforms so
the logistic-regression model sees more variety during training.

Transforms available
--------------------
  - horizontal flip
  - vertical flip
  - 90 / 180 / 270° rotation
  - brightness shift (± factor)
  - gaussian noise

Usage
-----
    from data_augmentation import augment_dataset
    X_aug, Y_aug = augment_dataset(X_orig, Y_orig, augmentations_per_image=4, seed=42)
"""

import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────────

def _flip_horizontal(img):
    return img[:, ::-1, :]

def _flip_vertical(img):
    return img[::-1, :, :]

def _rotate90(img):
    return np.rot90(img, k=1)

def _rotate180(img):
    return np.rot90(img, k=2)

def _rotate270(img):
    return np.rot90(img, k=3)

def _brightness(img, factor=0.3):
    """Randomly brighten or darken. factor in [0, 1]."""
    delta = (np.random.rand() * 2 - 1) * factor       # in [-factor, +factor]
    aug   = img + delta
    return np.clip(aug, 0., 1.)

def _add_noise(img, std=0.05):
    noise = np.random.randn(*img.shape) * std
    return np.clip(img + noise, 0., 1.)


TRANSFORMS = [
    _flip_horizontal,
    _flip_vertical,
    _rotate90,
    _rotate180,
    _rotate270,
    lambda img: _brightness(img, 0.25),
    lambda img: _add_noise(img, 0.04),
]


# ── public API ─────────────────────────────────────────────────────────────────

def augment_dataset(X_orig, Y_orig, augmentations_per_image: int = 4, seed: int = 42):
    """
    Parameters
    ----------
    X_orig : ndarray, shape (m, H, W, 3), float in [0, 1]
        Original training images (already normalised).
    Y_orig : ndarray, shape (1, m)
        Labels.
    augmentations_per_image : int
        How many extra images to generate per original image.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_aug : ndarray, shape (m + m*augmentations_per_image, H, W, 3)
    Y_aug : ndarray, shape (1, m + m*augmentations_per_image)
    """
    np.random.seed(seed)
    m = X_orig.shape[0]

    aug_images = [X_orig]
    aug_labels = [Y_orig]

    for i in range(m):
        img   = X_orig[i]           # (H, W, 3)
        label = Y_orig[0, i]

        # randomly pick `augmentations_per_image` transforms (with replacement)
        chosen = np.random.choice(len(TRANSFORMS), size=augmentations_per_image, replace=False
                                  if augmentations_per_image <= len(TRANSFORMS) else True)
        for idx in chosen:
            new_img = TRANSFORMS[idx](img)
            aug_images.append(new_img[np.newaxis])          # (1, H, W, 3)
            aug_labels.append(np.array([[label]]))          # (1, 1)

    X_aug = np.concatenate(aug_images, axis=0)              # (m_total, H, W, 3)
    Y_aug = np.concatenate(aug_labels, axis=1)              # (1, m_total)

    # shuffle
    perm  = np.random.permutation(X_aug.shape[0])
    X_aug = X_aug[perm]
    Y_aug = Y_aug[:, perm]

    print(f"[Augmentation] Original: {m} images  →  Augmented: {X_aug.shape[0]} images")
    return X_aug, Y_aug


def preprocess_for_model(X_raw):
    """Flatten and normalise a raw image array (H5 uint8) → (features, m) float."""
    m  = X_raw.shape[0]
    num_px = X_raw.shape[1]
    X_flat = X_raw.reshape(m, -1).T / 255.
    return X_flat, num_px
