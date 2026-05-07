from flask import Flask, render_template, request, jsonify
import sys
import os

# Add src folder to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.predict import predict_sentiment
from src.train import train_model

app = Flask(__name__)

# Train the model automatically if it doesn't exist yet
if not os.path.exists('models/best_sentiment_model.pkl'):
    print("First time running? Training the model now...")
    train_model()
    print("Model trained and ready!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get('review', '')
    
    if not review.strip():
        return jsonify({'error': 'Please type something!'}), 400
    
    try:
        result = predict_sentiment(review)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/examples')
def get_examples():
    # Some example reviews I added so users can try quickly
    examples = [
        "The food was amazing and the service was excellent!",
        "Terrible experience. Food was cold and service was slow.",
        "Best pizza I've ever had in my life!",
        "Waited 45 minutes for cold food. Never coming back.",
        "The pasta was delicious and the staff was very friendly."
    ]
    return jsonify({'examples': examples})

if __name__ == '__main__':
    print("\n🍽️ Restaurant Sentiment Analyzer is starting...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)