"""
data_loader.py

Handles loading the Kaggle creditcard.csv dataset. If the file isn't
present (e.g. in CI or during a demo), it falls back to generating a
synthetic dataset that matches the original's structure and class ratio.

The real dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
Place creditcard.csv inside the /data folder before training.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "creditcard.csv"


def load_data(path: str | Path = CSV_PATH, verbose: bool = True) -> pd.DataFrame:
    """
    Load the creditcard dataset from disk.
    Falls back to synthetic generation when the file is missing.
    """
    path = Path(path)

    if path.exists():
        if verbose:
            print(f"[data] Loading dataset from {path} ...")
        df = pd.read_csv(path)
    else:
        if verbose:
            print(
                "[data] creditcard.csv not found — generating synthetic data.\n"
                "       Drop the real file into /data to use actual Kaggle data."
            )
        df = _make_synthetic(n_legit=56_800, n_fraud=100, seed=42)

    if verbose:
        fraud_pct = df["Class"].mean() * 100
        print(
            f"[data] {len(df):,} rows loaded  |  "
            f"{int(df['Class'].sum())} fraud  |  "
            f"{fraud_pct:.3f}% positive rate"
        )

    return df


# ── synthetic data generator ───────────────────────────────────────────────────

def _make_synthetic(n_legit: int = 56_800, n_fraud: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generate a DataFrame that mirrors the Kaggle creditcard.csv schema:
      Time, V1–V28 (PCA projections), Amount, Class

    The PCA features for legit vs fraud transactions are drawn from
    slightly different distributions so the classifier has something
    real to learn — it's not just noise.
    """
    rng = np.random.default_rng(seed)
    n_pca = 28

    def make_block(n, fraud=False):
        # Fraud transactions cluster differently in PCA space
        shift = rng.uniform(-2, 2, size=n_pca) if fraud else np.zeros(n_pca)
        scale = 1.5 if fraud else 1.0
        V = rng.normal(loc=shift, scale=scale, size=(n, n_pca))
        time = rng.uniform(0, 172_800, size=n)  # 2 days in seconds
        # Fraud tends to be small or large round amounts
        amount = (
            rng.choice([9.99, 49.99, 99.99, 199.0, 399.0], size=n)
            if fraud
            else np.abs(rng.exponential(scale=88, size=n))
        )
        label = np.ones(n, dtype=int) if fraud else np.zeros(n, dtype=int)
        return time, V, amount, label

    t_l, V_l, a_l, y_l = make_block(n_legit, fraud=False)
    t_f, V_f, a_f, y_f = make_block(n_fraud, fraud=True)

    V_cols = {f"V{i}": np.concatenate([V_l[:, i - 1], V_f[:, i - 1]]) for i in range(1, n_pca + 1)}

    df = pd.DataFrame(
        {
            "Time": np.concatenate([t_l, t_f]),
            **V_cols,
            "Amount": np.concatenate([a_l, a_f]),
            "Class": np.concatenate([y_l, y_f]),
        }
    )

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
