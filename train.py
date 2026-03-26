"""
train.py
========
Train the logistic-regression cat detector, optionally with data augmentation,
and save the weights to weights.npz.

Usage
-----
    python train.py                        # no augmentation
    python train.py --augment              # 4 extra images per original
    python train.py --augment --aug-factor 6 --lr 0.005 --iters 3000
"""

import argparse
import os
import numpy as np
from model import load_dataset, model
from data_augmentation import augment_dataset, preprocess_for_model


def parse_args():
    p = argparse.ArgumentParser(description="Train cat-vs-non-cat logistic regression")
    p.add_argument("--augment",     action="store_true", help="Enable data augmentation")
    p.add_argument("--aug-factor",  type=int,   default=4,     help="Augmentations per image")
    p.add_argument("--lr",          type=float, default=0.005, help="Learning rate")
    p.add_argument("--iters",       type=int,   default=2000,  help="Gradient descent iterations")
    p.add_argument("--seed",        type=int,   default=42,    help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("── Loading dataset ──────────────────────────")
    X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset()
    print(f"  Train: {X_train_orig.shape[0]} images | Test: {X_test_orig.shape[0]} images")

    # Normalise raw pixels → [0, 1]
    X_train_norm = X_train_orig / 255.
    X_test_norm  = X_test_orig  / 255.

    if args.augment:
        print("── Augmenting training data ─────────────────")
        X_train_aug, Y_train_aug = augment_dataset(
            X_train_norm, Y_train, augmentations_per_image=args.aug_factor, seed=args.seed
        )
    else:
        X_train_aug, Y_train_aug = X_train_norm, Y_train

    # Flatten: (m, H, W, 3) → (H*W*3, m)
    X_train_flat = X_train_aug.reshape(X_train_aug.shape[0], -1).T
    X_test_flat  = X_test_norm.reshape(X_test_norm.shape[0], -1).T

    num_px = X_train_orig.shape[1]

    print(f"── Training  (lr={args.lr}, iters={args.iters}) ───────")
    result = model(X_train_flat, Y_train_aug, X_test_flat, Y_test,
                   num_iterations=args.iters, learning_rate=args.lr, print_cost=True)

    train_acc = 100 - np.mean(np.abs(result["Y_prediction_train"] - Y_train_aug)) * 100
    test_acc  = 100 - np.mean(np.abs(result["Y_prediction_test"]  - Y_test))  * 100
    print(f"\n  Train accuracy : {train_acc:.2f}%")
    print(f"  Test  accuracy : {test_acc:.2f}%")

    # ── Save weights ──────────────────────────────
    np.savez("weights.npz",
             w=result["w"], b=np.array([result["b"]]),
             num_px=np.array([num_px]),
             classes=classes)
    print("\n✓ Weights saved to weights.npz")


if __name__ == "__main__":
    main()
