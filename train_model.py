import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import clean_text

os.makedirs("model", exist_ok=True)

train_df = pd.read_csv("data/twitter_training.csv", header=None)
test_df  = pd.read_csv("data/twitter_validation.csv", header=None)

train_df.columns = ["id", "topic", "sentiment", "text"]
test_df.columns  = ["id", "topic", "sentiment", "text"]

train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"]  = test_df["text"].apply(clean_text)

X_train = train_df["clean_text"]
y_train = train_df["sentiment"]

X_test = test_df["clean_text"]
y_test = test_df["sentiment"]

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc:.4f}")

joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model and vectorizer saved successfully.")
