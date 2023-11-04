from flask import Flask, request, jsonify
import joblib
from waitress import serve
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model and vectorizer
classifier = joblib.load("sentiment_analysis_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    try:
        data = request.get_json()
        text = data["text"]

        # Preprocess the input text using the same vectorizer
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
