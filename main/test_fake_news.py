import pickle

# Load model and vectorizer
with open("../data/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("../data/logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Take user input
news_text = input("Enter news text: ")

# Transform and predict
news_vector = tfidf.transform([news_text])
prediction = model.predict(news_vector)[0]

# Map prediction to label
label_map = {0: "TRUE", 1: "FAKE"}
print(f"Prediction: {label_map[prediction]}")
