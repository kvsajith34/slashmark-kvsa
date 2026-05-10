#!/usr/bin/env python3
"""
predict.py — Score new transactions against a saved model
==========================================================

Usage
-----
# score a CSV of new transactions
python predict.py --input transactions.csv --model random_forest

# use a custom threshold (overrides the one saved during training)
python predict.py --input transactions.csv --model random_forest --threshold 0.3

Output
------
A CSV with two extra columns appended:
  fraud_score   — raw probability (0–1); higher = more suspicious
  fraud_flag    — 1 if score exceeds the decision threshold, else 0

The input CSV must have the same columns as creditcard.csv
(Time, V1–V28, Amount). Class column is optional.
"""

import argparse
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.features import engineer_features, get_feature_columns

MODELS_DIR = Path(__file__).resolve().parent / "outputs" / "models"


def parse_args():
    p = argparse.ArgumentParser(description="Fraud Detection — Inference")
    p.add_argument("--input",     required=True, help="CSV of transactions to score")
    p.add_argument("--model",     default="random_forest",
                   choices=["logistic", "random_forest", "gradient_boost"])
    p.add_argument("--threshold", type=float, default=None,
                   help="Override the saved threshold (0–1)")
    p.add_argument("--output",    default=None,
                   help="Where to save results (default: <input>_scored.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── load model ────────────────────────────────────────────────────────────
    model_path = MODELS_DIR / f"{args.model}.joblib"
    if not model_path.exists():
        print(f"[error] No saved model found at {model_path}")
        print("        Run train.py first.")
        sys.exit(1)

    bundle   = joblib.load(model_path)
    model    = bundle["model"]
    threshold = args.threshold if args.threshold is not None else bundle["threshold"]
    feature_cols = bundle["features"]

    print(f"[predict] Model: {args.model}  |  Threshold: {threshold:.3f}")

    # ── load & engineer features ──────────────────────────────────────────────
    df = pd.read_csv(args.input)
    df = engineer_features(df)

    # sanity check — all expected features present
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[error] Input CSV is missing columns: {missing}")
        sys.exit(1)

    X = df[feature_cols].values

    # ── score ─────────────────────────────────────────────────────────────────
    probs = model.predict_proba(X)[:, 1]
    flags = (probs >= threshold).astype(int)

    df["fraud_score"] = np.round(probs, 4)
    df["fraud_flag"]  = flags

    flagged = flags.sum()
    print(f"[predict] {len(df):,} transactions scored  |  {flagged} flagged as fraud")

    # ── save results ──────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        in_path  = Path(args.input)
        out_path = in_path.with_stem(in_path.stem + "_scored")

    df.to_csv(out_path, index=False)
    print(f"[predict] Results saved → {out_path}")


if __name__ == "__main__":
    main()
