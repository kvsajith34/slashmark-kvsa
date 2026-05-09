"""
evaluate.py — Evaluate the trained model on the validation set.

What this script produces
-------------------------
  plots/confusion_matrix.png      — annotated heatmap
  plots/misclassified_samples.png — grid of images the model got wrong
  plots/feature_maps.png          — what the first Conv layer "sees"

Usage
-----
python evaluate.py                          # uses best_model.keras by default
python evaluate.py --model models/final_model.keras
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def get_args():
    p = argparse.ArgumentParser(
        description="Evaluate the Cat vs. Dog classifier."
    )
    p.add_argument("--model",      default="models/best_model.keras")
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--plots-dir",  default="plots")
    p.add_argument("--img-size",   type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names, plots_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion Matrix — Cats vs. Dogs", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(plots_dir, "confusion_matrix.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved → {out}")
    return cm


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def plot_roc(y_true, y_scores, plots_dir):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(plots_dir, "roc_curve.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curve saved → {out}")


# ---------------------------------------------------------------------------
# Misclassified samples
# ---------------------------------------------------------------------------

def plot_misclassified(val_gen, y_true, y_pred, class_names, plots_dir,
                       n_show=12):
    """Show images the model got wrong — usually the most informative thing."""
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print("No misclassified images — perfect score!")
        return

    n_show = min(n_show, len(wrong_idx))
    chosen = np.random.choice(wrong_idx, n_show, replace=False)

    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for ax, idx in zip(axes, chosen):
        img_path = os.path.join(val_gen.directory, val_gen.filenames[idx])
        img = load_img(img_path, target_size=(150, 150))
        ax.imshow(img)
        true_lbl  = class_names[y_true[idx]]
        pred_lbl  = class_names[y_pred[idx]]
        ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}", fontsize=9,
                     color="red")
        ax.axis("off")

    # Hide any leftover axes.
    for ax in axes[n_show:]:
        ax.axis("off")

    fig.suptitle("Misclassified Examples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(plots_dir, "misclassified_samples.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Misclassified samples saved → {out}")


# ---------------------------------------------------------------------------
# Feature-map visualisation
# ---------------------------------------------------------------------------

def plot_feature_maps(model, sample_img_path: str, plots_dir: str,
                      img_size: int):
    """
    Show what the first Conv layer activates on.  This is a great way to
    build intuition about what the network has learned.
    """
    img = load_img(sample_img_path, target_size=(img_size, img_size))
    x   = img_to_array(img) / 255.0
    x   = np.expand_dims(x, axis=0)   # shape (1, H, W, 3)

    # Find the first Conv2D layer.
    conv_layer = next(
        (l for l in model.layers if "conv2d" in l.name), None
    )
    if conv_layer is None:
        print("Could not find a Conv2D layer for feature-map visualisation.")
        return

    feat_model = Model(inputs=model.input, outputs=conv_layer.output)
    features   = feat_model.predict(x, verbose=0)   # (1, H', W', filters)

    n_filters = min(32, features.shape[-1])
    cols = 8
    rows = (n_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i, ax in enumerate(axes[:n_filters]):
        fmap = features[0, :, :, i]
        ax.imshow(fmap, cmap="viridis")
        ax.set_title(f"f{i}", fontsize=7)
        ax.axis("off")

    for ax in axes[n_filters:]:
        ax.axis("off")

    fig.suptitle(
        f"First Conv Layer ({conv_layer.name}) — Feature Maps",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(plots_dir, "feature_maps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature maps saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    os.makedirs(args.plots_dir, exist_ok=True)

    print("Loading model …")
    model = load_model(args.model)

    # Build the validation generator — no augmentation, just rescale.
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = val_datagen.flow_from_directory(
        os.path.join(args.data_dir, "val"),
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="binary",
        shuffle=False,
    )

    # class_indices: {'cats': 0, 'dogs': 1}
    class_names = {v: k for k, v in val_gen.class_indices.items()}
    class_names = [class_names[i] for i in sorted(class_names)]

    print("Running predictions on the validation set …")
    y_scores = model.predict(val_gen, verbose=1).ravel()
    y_pred   = (y_scores > 0.5).astype(int)
    y_true   = val_gen.classes

    # --- Metrics ---
    acc = np.mean(y_pred == y_true)
    print(f"\nValidation accuracy : {acc:.4f}  ({acc * 100:.1f} %)")
    print(f"Total images        : {len(y_true)}")
    print(f"Correct             : {int(np.sum(y_pred == y_true))}")
    print(f"Incorrect           : {int(np.sum(y_pred != y_true))}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # --- Plots ---
    plot_confusion_matrix(y_true, y_pred, class_names, args.plots_dir)
    plot_roc(y_true, y_scores, args.plots_dir)
    plot_misclassified(val_gen, y_true, y_pred, class_names, args.plots_dir)

    # Feature maps — pick the first validation image we can find.
    sample_path = None
    for cls in class_names:
        candidate_dir = Path(args.data_dir) / "val" / cls
        imgs = list(candidate_dir.glob("*.jpg")) + list(
            candidate_dir.glob("*.jpeg")
        ) + list(candidate_dir.glob("*.png"))
        if imgs:
            sample_path = str(imgs[0])
            break

    if sample_path:
        plot_feature_maps(model, sample_path, args.plots_dir, args.img_size)

    print("\nEvaluation complete. All plots saved to:", args.plots_dir)


if __name__ == "__main__":
    main()
