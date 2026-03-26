"""
train.py
========
Train the logistic-regression cat detector and save weights to weights.npz.

Usage
-----
    python train.py
    python train.py --lr 0.005 --iters 2000
"""

import argparse
import numpy as np
from model import load_dataset, model


def parse_args():
    p = argparse.ArgumentParser(
        description="Train cat-vs-non-cat logistic regression")
    p.add_argument("--lr",    type=float, default=0.005, help="Learning rate")
    p.add_argument("--iters", type=int,   default=2000,
                   help="Gradient descent iterations")
    p.add_argument("--seed",  type=int,   default=42,    help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("── Loading dataset ──────────────────────────")
    X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset()
    print(
        f"  Train: {X_train_orig.shape[0]} images | Test: {X_test_orig.shape[0]} images")

    # Flatten + normalise
    num_px = X_train_orig.shape[1]
    X_train = X_train_orig.reshape(X_train_orig.shape[0], -1).T / 255.
    X_test = X_test_orig.reshape(X_test_orig.shape[0],  -1).T / 255.

    print(f"── Training  (lr={args.lr}, iters={args.iters}) ───────")
    result = model(X_train, Y_train, X_test, Y_test,
                   num_iterations=args.iters, learning_rate=args.lr, print_cost=True)

    train_acc = 100 - \
        np.mean(np.abs(result["Y_prediction_train"] - Y_train)) * 100
    test_acc = 100 - \
        np.mean(np.abs(result["Y_prediction_test"] - Y_test)) * 100
    print(f"\n  Train accuracy : {train_acc:.2f}%")
    print(f"  Test  accuracy : {test_acc:.2f}%")

    np.savez("weights.npz",
             w=result["w"], b=np.array([result["b"]]),
             num_px=np.array([num_px]), classes=classes)
    print("\n✓ Weights saved to weights.npz")


if __name__ == "__main__":
    main()
