#!/usr/bin/env python3
"""
train.py — Credit Card Fraud Detection Pipeline
================================================

Usage
-----
# train all models with default settings (combined SMOTE+undersampling)
python train.py

# train a specific model with a different resampling strategy
python train.py --model random_forest --resample smote

# skip cross-validation for a quick run
python train.py --no-cv

Options
-------
--model       one of: logistic, random_forest, gradient_boost, all (default: all)
--resample    one of: smote, undersample, combined (default: combined)
--no-cv       skip cross-validation (faster, good for iteration)
--data        path to creditcard.csv (default: data/creditcard.csv)
--test-size   fraction held out for final evaluation (default: 0.2)
"""

import argparse
import sys
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

# make sure `src` is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_data, CSV_PATH
from src.features import engineer_features, get_feature_columns
from src.models import get_models, cross_validate_model, train_final
from src.evaluate import (
    evaluate,
    tune_threshold,
    plot_roc_pr_curves,
    plot_confusion_matrix,
    plot_class_distribution,
)

MODELS_DIR = Path(__file__).resolve().parent / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Credit Card Fraud Detection — Training Pipeline")
    p.add_argument("--model",     default="all",
                   choices=["logistic", "random_forest", "gradient_boost", "all"])
    p.add_argument("--resample",  default="combined",
                   choices=["smote", "undersample", "combined"])
    p.add_argument("--no-cv",     action="store_true",
                   help="Skip cross-validation (faster iteration)")
    p.add_argument("--data",      default=str(CSV_PATH),
                   help="Path to creditcard.csv")
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()

    print("\n══════════════════════════════════════════════")
    print("  Credit Card Fraud Detection — Training")
    print("══════════════════════════════════════════════\n")

    # ── 1. load & engineer features ───────────────────────────────────────────
    df = load_data(args.data)
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    y = df["Class"].values

    plot_class_distribution(y)

    # ── 2. train/test split (stratified to keep fraud in both sets) ───────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )

    print(f"\n[split]  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"[split]  Fraud in test: {y_test.sum()} ({y_test.mean()*100:.2f}%)\n")

    # ── 3. pick which models to train ─────────────────────────────────────────
    all_models = get_models()
    if args.model == "all":
        models_to_run = all_models
    else:
        models_to_run = {args.model: all_models[args.model]}

    results = {}      # stores test-set evaluation dicts
    best_thresholds = {}

    for name, model in models_to_run.items():
        print(f"\n{'═'*55}")
        print(f"  MODEL: {name.upper()}")
        print(f"{'═'*55}")

        # ── cross-validation ─────────────────────────────────────────────────
        if not args.no_cv:
            print(f"\n[cv] Running {5}-fold stratified CV (resampling inside folds)...")
            cv_scores = cross_validate_model(
                model, X_train, y_train,
                resample_strategy=args.resample,
            )
            print(
                f"\n  CV AUC-ROC : {cv_scores['auc_roc'].mean():.4f} "
                f"± {cv_scores['auc_roc'].std():.4f}"
            )
            print(
                f"  CV AUC-PR  : {cv_scores['auc_pr'].mean():.4f} "
                f"± {cv_scores['auc_pr'].std():.4f}"
            )

        # ── final training on full train set ─────────────────────────────────
        print(f"\n[train] Fitting {name} on full training set...")
        model = train_final(model, X_train, y_train, resample_strategy=args.resample)

        # ── hold-out evaluation ──────────────────────────────────────────────
        res = evaluate(model, X_test, y_test, model_name=name)
        results[name] = res

        # ── threshold tuning ─────────────────────────────────────────────────
        thresh = tune_threshold(res["probs"], y_test, model_name=name)
        best_thresholds[name] = thresh

        # ── confusion matrix at tuned threshold ──────────────────────────────
        plot_confusion_matrix(model, X_test, y_test, threshold=thresh, model_name=name)

        # ── save model ───────────────────────────────────────────────────────
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump({"model": model, "threshold": thresh, "features": feature_cols}, model_path)
        print(f"\n[saved] Model → {model_path}")

    # ── 4. compare all models side-by-side ───────────────────────────────────
    if len(results) > 1:
        plot_roc_pr_curves(results, y_test)

    # ── 5. summary table ──────────────────────────────────────────────────────
    print(f"\n\n{'═'*55}")
    print("  FINAL SUMMARY")
    print(f"{'═'*55}")
    print(f"  {'Model':<20}  {'AUC-ROC':>8}  {'AUC-PR':>8}  {'Threshold':>10}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*10}")
    for name, res in results.items():
        print(
            f"  {name:<20}  {res['auc_roc']:>8.4f}  {res['auc_pr']:>8.4f}"
            f"  {best_thresholds[name]:>10.3f}"
        )
    print(f"\n[done] Plots saved to outputs/plots/")
    print("[done] Models saved to outputs/models/\n")


if __name__ == "__main__":
    main()
