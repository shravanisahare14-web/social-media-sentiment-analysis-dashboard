import pickle
from src.preprocess import preprocess

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict_sentiment(text):
    processed = preprocess(text)
    vector = vectorizer.transform([processed])
    return model.predict(vector)[0]