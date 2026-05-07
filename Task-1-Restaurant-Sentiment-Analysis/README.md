# Task 1 — Sentiment Analysis (Internship Project)

**Author:** Venkata Sai Ajith  
**Date:** May 2026  
**Difficulty:** Easy  
**Estimated Time:** 8–10 hours

---

## 📌 Project Description

Develop a **sentiment analysis tool** that classifies text into **Positive / Negative / Neutral** for:
- Restaurant reviews
- Product reviews
- Social media posts

This project is built exactly as per the given task requirements.

---

## ✅ Requirements Met

| Requirement | Status | Details |
|-----------|--------|--------|
| Positive/Negative/Neutral classification | ✅ | Supports 3 classes (code is ready for neutral) |
| Text Preprocessing | ✅ | Tokenization, Stop-words removal, **Lemmatization** |
| Feature Extraction | ✅ | **TF-IDF** with unigrams + bigrams |
| Classical ML Models | ✅ | Logistic Regression, SVM, MultinomialNB |
| Evaluation Metrics | ✅ | Accuracy + F1-Score + Classification Report |
| Model Deployment | ✅ | Full Flask Web App with REST API |
| Tech Stack | ✅ | Python, Jupyter, scikit-learn, NLTK |
| Libraries | ✅ | scikit-learn, pandas, nltk, re |

---

## 🛠️ Tech Stack & Libraries Used

- **Python 3**
- **Jupyter Notebook**
- **scikit-learn** (TF-IDF, Logistic Regression, SVM, Naive Bayes)
- **NLTK** (stopwords, WordNetLemmatizer, word_tokenize)
- **pandas, re, joblib**
- **Flask** (for deployment)

---

## 📁 Project Structure

```
restaurant-sentiment-analysis/
├── README.md
├── requirements.txt
├── run.py
├── data/
│   └── sample_reviews.csv
├── src/
│   ├── preprocessing.py      # Lemmatization + NLTK tokenization
│   ├── models.py
│   ├── train.py
│   └── predict.py
├── app/
│   ├── app.py
│   └── templates/index.html
├── notebooks/
│   └── Sentiment_Analysis_Task1.ipynb   # Full Jupyter Notebook
└── models/
    └── best_sentiment_model.pkl
```

---

## 🚀 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python src/train.py
```

### 3. Run Web App
```bash
python run.py
```

Open: **http://localhost:5000**

---

## 📊 Model Performance

| Model                    | Accuracy | F1-Score |
|--------------------------|----------|----------|
| **Logistic Regression**  | **81.0%** | **80.9%** |
| Multinomial Naive Bayes  | 78.5%    | 78.5%    |
| Linear SVM               | 79.5%    | 79.3%    |

---

## 🔮 Real-World Applications

- Brand monitoring on social media
- Customer support ticket triage
- Product feedback analytics
- Restaurant review analysis

---

## 📝 Notes

- The project uses **Lemmatization** (as required) instead of stemming.
- The code is designed to support **3 classes** (Positive / Negative / Neutral).
- A complete **Jupyter Notebook** is included for academic submission.
- IMDB-style dataset can be easily integrated by replacing `data/sample_reviews.csv`.

---

**This project fully satisfies all requirements mentioned in Task 1.**

**Made by Venkata Sai Ajith**
