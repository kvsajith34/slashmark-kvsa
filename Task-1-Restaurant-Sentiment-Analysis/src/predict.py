import joblib
from src.preprocessing import clean_text

def predict_sentiment(review_text):
    model = joblib.load('models/best_sentiment_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    cleaned = clean_text(review_text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    
    if prediction == 2:
        sentiment = "Positive 😊"
    elif prediction == 1:
        sentiment = "Neutral 😐"
    else:
        sentiment = "Negative 😞"
    
    confidence = round(max(proba) * 100, 2)
    return {'sentiment': sentiment, 'confidence': confidence, 'prediction': int(prediction)}