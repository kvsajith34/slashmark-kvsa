"""
predict.py — Predict whether an image contains a cat or a dog.

Usage
-----
python predict.py path/to/your/image.jpg
python predict.py path/to/your/image.jpg --model models/best_model.keras
python predict.py images/*.jpg          # batch mode — multiple images
"""

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


CLASS_NAMES = ["Cat", "Dog"]
IMG_SIZE    = 150     # must match what you used at training time


def preprocess(img_path: str, img_size: int = IMG_SIZE) -> np.ndarray:
    """Load one image from disk and prepare it for the model."""
    img = load_img(img_path, target_size=(img_size, img_size))
    x   = img_to_array(img) / 255.0          # normalise to [0, 1]
    return np.expand_dims(x, axis=0)         # add batch dimension → (1,H,W,3)


def predict_one(model, img_path: str, img_size: int = IMG_SIZE):
    """
    Run inference on a single image.

    Returns
    -------
    label : str   — "Cat" or "Dog"
    conf  : float — confidence in that prediction (0 – 1)
    """
    x     = preprocess(img_path, img_size)
    score = float(model.predict(x, verbose=0)[0][0])

    # score is the probability that the image is a DOG (class index 1).
    if score > 0.5:
        return "Dog", score
    else:
        return "Cat", 1.0 - score


def show_predictions(results: list, save_path: str = None):
    """
    Display a grid of images with their predicted labels.

    Parameters
    ----------
    results   : list of (img_path, label, confidence) tuples
    save_path : if given, save the figure here instead of (only) showing it
    """
    n    = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for i, (img_path, label, conf) in enumerate(results):
        ax  = axes[i // cols][i % cols]
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        ax.imshow(img)
        color = "#2ecc71" if label == "Cat" else "#3498db"
        ax.set_title(
            f"{label}  ({conf * 100:.1f} % confident)",
            fontsize=10, color=color, fontweight="bold",
        )
        ax.axis("off")

    # Turn off unused axes.
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Cats vs. Dogs — Predictions", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Prediction grid saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Predict cat or dog for one or more images."
    )
    parser.add_argument(
        "images", nargs="+",
        help="Path(s) to image file(s).",
    )
    parser.add_argument(
        "--model", default="models/best_model.keras",
        help="Path to the saved Keras model.",
    )
    parser.add_argument(
        "--img-size", type=int, default=IMG_SIZE,
        help="Image size used during training.",
    )
    parser.add_argument(
        "--save", default=None,
        help="If provided, save the prediction grid to this path instead of "
             "displaying it.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(
            f"Model file not found: {args.model}\n"
            "Run train.py first, or point --model at your .keras file."
        )

    print(f"Loading model from {args.model} …")
    model = load_model(args.model)

    results = []
    print("\nPredictions:")
    print("-" * 45)
    for img_path in args.images:
        if not os.path.isfile(img_path):
            print(f"  [skip] {img_path}  — file not found")
            continue
        label, conf = predict_one(model, img_path, args.img_size)
        results.append((img_path, label, conf))
        print(f"  {os.path.basename(img_path):30s}  →  {label}  "
              f"({conf * 100:.1f} %)")

    if results:
        show_predictions(results, save_path=args.save)


if __name__ == "__main__":
    main()
