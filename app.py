import streamlit as st
import joblib
import os

from preprocess import clean_text

st.set_page_config(page_title="Twitter Sentiment Analysis", layout="centered")

MODEL_PATH = "model/sentiment_model.pkl"
VECT_PATH = "model/vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    st.error("Assets not found. Please run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

st.title("Twitter Sentiment Analysis Dashboard")

tweet = st.text_area("Enter a tweet")

if st.button("Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(tweet)
        vector = vectorizer.transform([cleaned])

        # 🔹 Prediction + confidence
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]
        confidence = max(proba)

        # 🔹 Neutral logic
        if confidence < 0.6:
            st.info("Neutral 😐 (low confidence)")
        else:
            if prediction.lower() == "positive":
                st.success(f"Positive 😊 (confidence: {confidence:.2f})")
            elif prediction.lower() == "negative":
                st.error(f"Negative 😠 (confidence: {confidence:.2f})")
            else:
                st.info(f"Sentiment: {prediction}")