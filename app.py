import streamlit as st
import joblib
import os
import time

from preprocess import clean_text

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.markdown("""
<style>
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(15px);}
    to {opacity: 1; transform: translateY(0);}
}

@keyframes glow {
    0% {box-shadow: 0 0 5px #555;}
    50% {box-shadow: 0 0 20px #888;}
    100% {box-shadow: 0 0 5px #555;}
}

body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.block-container {
    padding-top: 2rem;
}

.card {
    animation: fadeIn 0.8s ease-in-out;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

.positive {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    animation: glow 2s infinite;
}

.negative {
    background: linear-gradient(135deg, #cb2d3e, #ef473a);
    color: white;
    animation: glow 2s infinite;
}

.neutral {
    background: linear-gradient(135deg, #232526, #414345);
    color: #f1f1f1;
    animation: glow 2s infinite;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "model/sentiment_model.pkl"
VECT_PATH = "model/vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    st.error("🚨 Model files not found. Please run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

st.markdown("<h1 style='text-align:center;'>💬 Twitter Sentiment Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; opacity:0.8;'>AI-powered sentiment detection with confidence-based neutrality</p>", unsafe_allow_html=True)

st.divider()

tweet = st.text_area(
    "✍️ Enter a tweet",
    height=120,
    placeholder="Type something like: I love this phone!"
)

if st.button("🚀 Analyze Sentiment", use_container_width=True):

    if tweet.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        with st.spinner("Analyzing sentiment..."):
            time.sleep(0.6)

            cleaned = clean_text(tweet)
            vector = vectorizer.transform([cleaned])

            probs = model.predict_proba(vector)[0]
            max_prob = probs.max()
            prediction = model.classes_[probs.argmax()]

        st.divider()

        if max_prob < 0.6:
            st.markdown(
                "<div class='card neutral'>😐 Neutral<br><small>Low confidence prediction</small></div>",
                unsafe_allow_html=True
            )

        elif prediction.lower() == "positive":
            st.markdown(
                "<div class='card positive'>😊 Positive</div>",
                unsafe_allow_html=True
            )

        elif prediction.lower() == "negative":
            st.markdown(
                "<div class='card negative'>😠 Negative</div>",
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f"<div class='card neutral'>{prediction}</div>",
                unsafe_allow_html=True
            )

        st.progress(float(max_prob))
        st.caption(f"Prediction confidence: **{max_prob:.2f}**")

st.markdown("<br><hr style='opacity:0.2;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; opacity:0.6;'>Built with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)
