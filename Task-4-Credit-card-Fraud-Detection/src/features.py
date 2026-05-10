"""
features.py

The raw Kaggle dataset already has PCA-transformed features (V1–V28),
so there's no raw transaction data to engineer from scratch. What we
*can* do is build on top of the structure that's still in the clear:
Time and Amount.

These three additions consistently help in practice:
  1. log1p(Amount)         — right-skewed distribution, log normalises it
  2. Hour of day           — fraud patterns often cluster at night / off-hours
  3. Amount percentile bin — coarse binning that captures "unusually large"
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with extra features appended.
    Original columns are kept intact so nothing downstream breaks.
    """
    out = df.copy()

    # log-transform Amount — the raw values span from $0 to $25k+
    # which causes tree-based models no problems but kills LR/SVM
    out["log_amount"] = np.log1p(out["Amount"])

    # hour of day derived from the Time column (seconds since first transaction)
    # wraps around midnight using modulo
    out["hour_of_day"] = (out["Time"] % 86_400) // 3600

    # rough amount category — helps the model notice "unusually large"
    # transactions without needing to learn the exact boundary itself
    out["amount_bin"] = pd.cut(
        out["Amount"],
        bins=[-1, 10, 100, 1_000, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of columns to use as model input.
    Excludes raw Amount (we keep log_amount instead) and the target.
    """
    exclude = {"Class", "Amount"}
    return [c for c in df.columns if c not in exclude]
