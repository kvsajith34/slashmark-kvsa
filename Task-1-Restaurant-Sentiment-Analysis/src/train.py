import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.preprocessing import preprocess_data
from src.models import get_model
import os

def train_model():
    print("Training 3-class model (0=Negative, 1=Neutral, 2=Positive)...")
    df = pd.read_csv('data/sample_reviews_3class.csv')
    df = preprocess_data(df)
    X = df['cleaned_review']
    y = df['Liked']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = get_model('logistic_regression')
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"✅ F1-Score: {f1_score(y_test, y_pred, average='weighted'):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("\n💾 3-class model saved successfully!")

if __name__ == "__main__":
    train_model()