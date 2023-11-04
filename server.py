from flask import Flask, request, jsonify
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from waitress import serve
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model and vectorizer from JSON files
with open("sentiment_analysis_model.json", "r") as model_file:
    model_params = json.load(model_file)
    classifier = MultinomialNB(alpha=model_params["alpha"], class_log_prior=model_params["class_log_prior"])

with open("vectorizer.json", "r") as vectorizer_file:
    vectorizer_params = json.load(vectorizer_file)
    vectorizer = CountVectorizer(vocabulary=vectorizer_params)

@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    try:
        data = request.get_json()
        text = data["text"]

        # Preprocess the input text using the loaded vectorizer
        input_vector = vectorizer.transform([text])

        # Make a prediction
        prediction = classifier.predict(input_vector)
        sentiment = prediction[0]

        response = {"sentiment": sentiment}
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/")
def index():
    return "Welcome to Sentiment Analysis API"


if __name__ == "__main__":
    print("Server is running ...")
    serve(app, host="0.0.0.0", port=5000)
