"""
train.py — Train the Cat vs. Dog CNN.

Quick start
-----------
# 1. Prepare data first (creates ./data/train and ./data/val)
python setup_data.py

# 2. Train (defaults work for the sample dataset)
python train.py

# 3. For the full Kaggle dataset, bump epochs and image size:
python train.py --data-dir data --epochs 30 --img-size 150

The script saves:
    models/best_model.keras   — weights of the epoch with the best val accuracy
    models/final_model.keras  — weights after the last epoch
    plots/training_curves.png — accuracy & loss curves (great for the report!)
"""

import argparse
import os

# Silence TensorFlow's overly chatty startup messages.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_model


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Train the Cat vs. Dog CNN classifier."
    )
    p.add_argument("--data-dir",   default="data",    help="Root of dataset")
    p.add_argument("--model-dir",  default="models",  help="Where to save weights")
    p.add_argument("--plots-dir",  default="plots",   help="Where to save plots")
    p.add_argument("--img-size",   type=int, default=150, help="Image side length")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs",     type=int, default=25)
    p.add_argument("--lr",         type=float, default=1e-3, help="Initial lr")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def make_generators(data_dir: str, img_size: int, batch_size: int, seed: int):
    """
    Build train and validation generators.

    Augmentation is applied only to the training set — the validation
    generator just normalises pixel values so we get an honest reading
    of how well the model generalises.
    """
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,          # rotate up to ±15°
        width_shift_range=0.1,      # shift horizontally by up to 10 %
        height_shift_range=0.1,     # shift vertically by up to 10 %
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",        # fill empty pixels after a transform
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        seed=seed,
    )
    val_gen = val_aug.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,          # keep order consistent for evaluation
        seed=seed,
    )
    return train_gen, val_gen


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def make_callbacks(model_dir: str):
    """Set up the training callbacks."""
    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, "best_model.keras")

    callbacks = [
        # Save the best checkpoint automatically.
        ModelCheckpoint(
            best_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Halve the learning rate if val_loss plateaus for 4 epochs.
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        # Stop early if val_loss doesn't improve for 8 epochs in a row.
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        # Keep a CSV log — handy if you want to analyse results later.
        CSVLogger(os.path.join(model_dir, "training_log.csv")),
    ]
    return callbacks


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(history, plots_dir: str):
    """
    Save accuracy and loss curves to disk.

    Having both on one figure makes it easy to spot overfitting:
    if training accuracy keeps climbing while val accuracy plateaus,
    you're overfitting.
    """
    os.makedirs(plots_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history.history["accuracy"]) + 1)

    # --- Accuracy ---
    ax1.plot(epochs, history.history["accuracy"],    "b-o", label="Training")
    ax1.plot(epochs, history.history["val_accuracy"], "r-o", label="Validation")
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Loss ---
    ax2.plot(epochs, history.history["loss"],     "b-o", label="Training")
    ax2.plot(epochs, history.history["val_loss"], "r-o", label="Validation")
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Training Curves — Cats vs. Dogs CNN", fontsize=16, y=1.02)
    plt.tight_layout()
    out = os.path.join(plots_dir, "training_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nTraining curves saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # Reproducibility — set seeds before anything else.
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 55)
    print("  Cats vs. Dogs CNN — Training")
    print("=" * 55)
    print(f"  Image size  : {args.img_size} × {args.img_size}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Max epochs  : {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print("=" * 55, "\n")

    # Build data generators.
    train_gen, val_gen = make_generators(
        args.data_dir, args.img_size, args.batch_size, args.seed
    )
    print(f"\nClass indices: {train_gen.class_indices}")
    # class_indices tells us: {'cats': 0, 'dogs': 1}
    # So model output > 0.5 → dog, ≤ 0.5 → cat.

    # Build model.
    model = build_model(input_shape=(args.img_size, args.img_size, 3))
    model.summary()

    # Train.
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=make_callbacks(args.model_dir),
        verbose=1,
    )

    # Save the final weights separately from the best checkpoint.
    final_path = os.path.join(args.model_dir, "final_model.keras")
    model.save(final_path)
    print(f"\nFinal model saved → {final_path}")

    # Plot training curves.
    plot_history(history, args.plots_dir)

    # Quick summary.
    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nBest validation accuracy: {best_val_acc:.4f}  "
          f"({best_val_acc * 100:.1f} %)")
    print("\nTraining complete. Run evaluate.py to see the confusion matrix.")


if __name__ == "__main__":
    main()
