"""
balancer.py

Fraud datasets are brutally imbalanced — typically 0.17% positive.
This module offers three strategies:

  smote       — oversample minority with synthetic neighbours (SMOTE)
  undersample — randomly remove majority examples
  combined    — undersample majority first, then SMOTE minority up
                (usually the best tradeoff: less noise than pure SMOTE,
                 less data loss than pure undersampling)

The strategy is applied *inside* each cross-validation fold during
training so we never leak resampling information into the validation set.
"""

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


STRATEGIES = ("smote", "undersample", "combined")


def make_resampler(strategy: str = "combined"):
    """
    Return an imblearn Pipeline that resamples X, y.

    Parameters
    ----------
    strategy : one of 'smote', 'undersample', 'combined'
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got '{strategy}'")

    if strategy == "smote":
        return ImbPipeline(
            steps=[("smote", SMOTE(random_state=42, k_neighbors=5))]
        )

    if strategy == "undersample":
        return ImbPipeline(
            steps=[("under", RandomUnderSampler(random_state=42))]
        )

    # combined: bring majority down to ~10x minority, then SMOTE minority up to match
    return ImbPipeline(
        steps=[
            ("under", RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
            ("smote", SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=5)),
        ]
    )


def resample(X, y, strategy: str = "combined"):
    """Convenience wrapper — fit-resample in one call."""
    resampler = make_resampler(strategy)
    return resampler.fit_resample(X, y)
