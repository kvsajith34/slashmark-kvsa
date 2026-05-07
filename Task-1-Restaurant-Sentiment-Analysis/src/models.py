# models.py
# Simple and clean model definitions

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def get_logistic_regression():
    """Logistic Regression - Best performer"""
    return LogisticRegression(
        C=5.0,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )

def get_naive_bayes():
    """Naive Bayes - Fast baseline"""
    return MultinomialNB(alpha=0.2)

def get_svm():
    """Linear SVM with probability calibration"""
    svc = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    return CalibratedClassifierCV(svc, cv=3)

# Registry
MODEL_REGISTRY = {
    'logistic_regression': get_logistic_regression,
    'naive_bayes': get_naive_bayes,
    'svm': get_svm,
}

def get_model(model_name='logistic_regression'):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name]()

print("✅ Models loaded successfully")