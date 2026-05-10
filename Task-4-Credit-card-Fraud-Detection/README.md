# Credit Card Fraud Detection

An end-to-end ML pipeline for detecting fraudulent credit card transactions. Built with scikit-learn and imbalanced-learn, covering everything from class imbalance handling to cost-sensitive threshold tuning.

---

## What this does

Real fraud data is brutally skewed — typically fewer than 0.2% of transactions are fraudulent. A naive model that predicts "legit" for every transaction would hit 99.8% accuracy while being completely useless. This pipeline tackles that properly:

- **Handles imbalance** with SMOTE oversampling, random undersampling, or a combined strategy (default)
- **Engineers features** on top of the PCA-compressed Kaggle data (log-amount, hour-of-day, amount bins)
- **Trains three models** — logistic regression (baseline), random forest, and gradient boosting
- **Cross-validates correctly** — resampling happens *inside* each fold, not before splitting
- **Evaluates with AUC-PR** alongside AUC-ROC (PR curves tell the real story on imbalanced data)
- **Tunes the decision threshold** to minimise business cost, not just classification error

---

## Quickstart

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection
cd credit-card-fraud-detection

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**With the real Kaggle dataset (recommended)**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder. The file is ~143 MB and excluded from version control.

**Without the real dataset (demo mode)**

Skip the download step — the pipeline auto-generates synthetic data that mirrors the Kaggle schema and class ratio.

```bash
python train.py
```

---

## CLI reference

### train.py

```
python train.py [OPTIONS]

Options:
  --model     logistic | random_forest | gradient_boost | all   (default: all)
  --resample  smote | undersample | combined                     (default: combined)
  --no-cv     skip cross-validation for faster iteration
  --data      path to creditcard.csv
  --test-size fraction to hold out for evaluation                (default: 0.2)
```

Examples:
```bash
# train all models with default settings
python train.py

# train only random forest, SMOTE only, skip CV
python train.py --model random_forest --resample smote --no-cv

# point to a custom CSV location
python train.py --data /path/to/creditcard.csv
```

### predict.py

Score new transactions with a saved model:

```bash
python predict.py --input new_transactions.csv --model random_forest
```

The output CSV gets two columns appended: `fraud_score` (0–1 probability) and `fraud_flag` (1 = flagged).

---

## Project structure

```
credit-card-fraud-detection/
├── train.py              # entry point — trains, evaluates, saves
├── predict.py            # inference on new transaction CSV files
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data_loader.py    # loads CSV or generates synthetic fallback
│   ├── features.py       # feature engineering (log amount, hour, bins)
│   ├── balancer.py       # SMOTE, undersampling, combined pipeline
│   ├── models.py         # model definitions + CV + final training
│   └── evaluate.py       # metrics, ROC/PR curves, threshold tuning
│
├── data/
│   └── README.md         # download instructions for creditcard.csv
│
└── outputs/
    ├── plots/            # ROC/PR curves, confusion matrices, threshold charts
    └── models/           # saved .joblib model bundles
```

---

## Imbalance strategies

| Strategy | What it does | Best when |
|---|---|---|
| `smote` | Synthesises new fraud samples using KNN | Enough fraud samples to interpolate |
| `undersample` | Randomly removes legit samples | Data is large and you can afford to lose some |
| `combined` | Undersample majority to 10:1, then SMOTE up to 1:1 | General-purpose (**default**) |

Resampling is applied inside each CV fold to avoid data leakage.

---

## Why AUC-PR, not just AUC-ROC?

AUC-ROC measures separation between classes overall — it's great for balanced problems. On imbalanced data, the massive true-negative count artificially inflates the curve. AUC-PR (Average Precision) ignores true negatives entirely and focuses on how well the model identifies the minority class. For fraud detection, that's the number that actually matters.

---

## Threshold tuning

Default scikit-learn classifiers output 1 if `P(fraud) ≥ 0.5`. That's rarely the right cutoff for fraud. This pipeline models the business cost explicitly:

- **False negative** (missed fraud) ≈ $120 — chargeback + issuer fee
- **False positive** (false alarm) ≈ $2 — investigation + customer friction

The optimal threshold minimises `FN × $120 + FP × $2`. You can change these constants in `src/evaluate.py`.

---

## Dataset

[ULB Machine Learning Group — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions over two days in September 2013
- 492 fraud cases (0.172%)
- Features V1–V28 are PCA projections of the original variables (confidential)
- `Time` and `Amount` are in the clear

---

## Tech stack

- Python 3.10+
- scikit-learn — models, CV, metrics
- imbalanced-learn — SMOTE, undersampling
- pandas / numpy — data handling
- matplotlib / seaborn — visualisation
- joblib — model persistence
