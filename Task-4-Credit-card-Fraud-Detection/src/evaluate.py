"""
evaluate.py

Everything related to measuring and visualising model performance.

A quick note on why AUC-PR matters more than AUC-ROC for fraud:
  AUC-ROC can look great (>0.99) even when precision is terrible, because
  the huge number of true negatives always inflates the TPR/FPR curve.
  AUC-PR forces the model to do well on the minority class specifically —
  which is exactly what we care about here.

Threshold tuning:
  sklearn's .predict() uses 0.5 by default, but for fraud we usually want
  to catch more fraud at the cost of more false alarms. The optimal
  threshold depends on the business cost ratio (FN cost / FP cost).

Cost model (simplified):
  FN (missed fraud) → average chargeback loss ≈ $120
  FP (false alarm)  → customer friction / investigation cost ≈ $2
  → we weight false negatives 60× more than false positives
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


PLOTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# business cost assumptions
COST_FN = 120   # missed fraud — chargeback + issuer fee
COST_FP = 2     # false alarm — investigation + customer friction


# ── core metrics ───────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, model_name: str = "model") -> dict:
    """
    Compute and print the full set of evaluation metrics.
    Returns a dict with scores and the predicted probabilities.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc_roc = roc_auc_score(y_test, probs)
    auc_pr  = average_precision_score(y_test, probs)

    print(f"\n{'─'*55}")
    print(f"  {model_name.upper()}")
    print(f"{'─'*55}")
    print(f"  AUC-ROC : {auc_roc:.4f}  (higher = better overall separation)")
    print(f"  AUC-PR  : {auc_pr:.4f}  (higher = better on minority class)")
    print(f"\n{classification_report(y_test, preds, target_names=['Legit','Fraud'])}")

    return {"auc_roc": auc_roc, "auc_pr": auc_pr, "probs": probs}


# ── threshold tuning ───────────────────────────────────────────────────────────

def tune_threshold(probs: np.ndarray, y_test: np.ndarray, model_name: str = "model") -> float:
    """
    Find the decision threshold that minimises expected cost given the
    FN/FP cost ratio defined above.

    Also plots precision, recall, and F1 vs threshold so the business
    can pick their own operating point if needed.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    # expected cost at each threshold
    costs = []
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        cost = fn * COST_FN + fp * COST_FP
        costs.append(cost)

    best_idx  = int(np.argmin(costs))
    best_thresh = float(thresholds[best_idx])
    best_cost   = costs[best_idx]

    print(f"\n  [{model_name}] Cost-optimal threshold: {best_thresh:.3f}  "
          f"(expected cost: ${best_cost:,.0f})")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Threshold Analysis — {model_name}", fontsize=13, fontweight="bold")

    ax = axes[0]
    f1 = 2 * precision * recall / np.where((precision + recall) == 0, 1, precision + recall)
    ax.plot(thresholds, precision[:-1], label="Precision", color="#2196F3")
    ax.plot(thresholds, recall[:-1],    label="Recall",    color="#FF5722")
    ax.plot(thresholds, f1[:-1],        label="F1",        color="#4CAF50", linestyle="--")
    ax.axvline(best_thresh, color="gray", linestyle=":", label=f"Best = {best_thresh:.3f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(thresholds, costs, color="#9C27B0")
    ax.axvline(best_thresh, color="gray", linestyle=":")
    ax.scatter([best_thresh], [best_cost], color="red", zorder=5, label=f"Min cost = ${best_cost:,.0f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Expected Cost ($)")
    ax.set_title(f"Business Cost vs Threshold\n(FN=${COST_FN}, FP=${COST_FP})")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"threshold_{model_name}.png", dpi=150)
    plt.close()

    return best_thresh


# ── curve plots ────────────────────────────────────────────────────────────────

def plot_roc_pr_curves(results: dict, y_test: np.ndarray):
    """
    Overlay ROC and PR curves for all models on the same figure.
    results = {model_name: {"probs": np.ndarray, "auc_roc": float, "auc_pr": float}}
    """
    palette = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Comparison: ROC vs Precision-Recall", fontsize=13, fontweight="bold")

    for i, (name, res) in enumerate(results.items()):
        color = palette[i % len(palette)]
        probs = res["probs"]

        # ROC
        fpr, tpr, _ = roc_curve(y_test, probs)
        axes[0].plot(fpr, tpr, color=color, lw=1.8,
                     label=f"{name}  (AUC={res['auc_roc']:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_test, probs)
        axes[1].plot(rec, prec, color=color, lw=1.8,
                     label=f"{name}  (AP={res['auc_pr']:.3f})")

    # ROC aesthetics
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # PR aesthetics
    baseline = y_test.mean()
    axes[1].axhline(baseline, color="k", linestyle="--", lw=0.8,
                    label=f"Random ({baseline:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve\n(more informative for imbalanced data)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_pr_curves.png", dpi=150)
    plt.close()
    print(f"\n[plots] Saved ROC/PR curves → {PLOTS_DIR / 'roc_pr_curves.png'}")


def plot_confusion_matrix(model, X_test, y_test, threshold: float, model_name: str):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold={threshold:.3f})")
    plt.tight_layout()
    path = PLOTS_DIR / f"confusion_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"[plots] Saved confusion matrix → {path}")


def plot_class_distribution(y: np.ndarray):
    counts = np.bincount(y)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Legit (0)", "Fraud (1)"], counts, color=["#2196F3", "#FF5722"], width=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{cnt:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Class Distribution (raw data)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / "class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"[plots] Saved class distribution → {path}")
