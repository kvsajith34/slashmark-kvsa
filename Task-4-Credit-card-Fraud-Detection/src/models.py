"""
models.py

Three models in increasing complexity:

  logistic   — fast baseline; interpretable; terrible with raw imbalance
               but decent after resampling; good sanity check
  random_forest — typically the workhorse for tabular fraud data
  gradient_boost — often slightly better than RF but slower to train;
                   we use sklearn's HistGradientBoosting (fast, native
                   missing-value support)

All classifiers expose predict_proba so threshold tuning works uniformly.
Cross-validation is done with StratifiedKFold to preserve class ratios
in every fold (critical when positive rate is < 1%).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    make_scorer,
    average_precision_score,
    roc_auc_score,
)

from .balancer import resample


# ── model catalogue ────────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Return a dict of {name: sklearn estimator}.

    Logistic Regression needs scaling (it's gradient-based).
    The tree models don't, but it doesn't hurt them either.
    """
    return {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.01,
                        class_weight="balanced",  # built-in cost-sensitivity
                        solver="lbfgs",
                        max_iter=1_000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced_subsample",  # each tree gets balanced weights
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boost": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
        ),
    }


# ── cross-validation ───────────────────────────────────────────────────────────

def cross_validate_model(
    model,
    X,
    y,
    resample_strategy: str = "combined",
    n_splits: int = 5,
) -> dict:
    """
    Run stratified k-fold CV with resampling applied inside each fold.

    Resampling inside the fold (not before) is essential — applying SMOTE
    to the full dataset before splitting would leak synthetic neighbours
    from the training set into validation, inflating scores artificially.

    Returns a dict with per-fold AUC-ROC and AUC-PR scores.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    roc_scores, pr_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # resample only the training portion
        X_tr_res, y_tr_res = resample(X_tr, y_tr, strategy=resample_strategy)

        model.fit(X_tr_res, y_tr_res)
        probs = model.predict_proba(X_val)[:, 1]

        roc = roc_auc_score(y_val, probs)
        pr  = average_precision_score(y_val, probs)
        roc_scores.append(roc)
        pr_scores.append(pr)

        print(
            f"  Fold {fold}/{n_splits}  |  "
            f"AUC-ROC: {roc:.4f}  |  AUC-PR: {pr:.4f}"
        )

    return {
        "auc_roc": np.array(roc_scores),
        "auc_pr":  np.array(pr_scores),
    }


def train_final(model, X_train, y_train, resample_strategy: str = "combined"):
    """
    Resample the full training set and fit the model.
    Returns the fitted model.
    """
    X_res, y_res = resample(X_train, y_train, strategy=resample_strategy)
    model.fit(X_res, y_res)
    return model
