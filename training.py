import json
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download("stopwords")
nltk.download("punkt")

data = pd.read_csv("Tweets.csv")

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    words = word_tokenize(text)
    words = [
        word.lower()
        for word in words
        if word.isalnum() and word.lower() not in stop_words
    ]
    return " ".join(words)

data["cleaned_text"] = data["text"].apply(preprocess_text)

X = data["cleaned_text"]
y = data["airline_sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_bow, y_train)

y_pred = classifier.predict(X_test_bow)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Save the vectorizer as JSON
vectorizer_json = json.dumps(vectorizer.vocabulary_)

with open("vectorizer.json", "w") as vectorizer_file:
    vectorizer_file.write(vectorizer_json)

# Save the classifier parameters
classifier_params = {
    "alpha": classifier.alpha,
    "class_log_prior": classifier.class_log_prior_.tolist(),
    "class_count": classifier.class_count_.tolist(),
    "feature_count": classifier.feature_count_.tolist()
}

with open("classifier.json", "w") as classifier_file:
    classifier_file.write(json.dumps(classifier_params))

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)
