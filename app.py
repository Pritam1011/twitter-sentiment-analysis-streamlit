import streamlit as st
import joblib
import os
import time
from preprocess import clean_text

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Gold Standard Sentiment AI",
    page_icon="✨",
    layout="centered"
)

# ---------------- PREMIUM BLACK & GOLD UI ----------------
st.markdown("""
<style>
/* Global Font Override to Times New Roman */
html, body, [class*="st-"], div, p, h1, h2, h3, button, textarea {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Main Background Animation */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at center, #1e1e1e 0%, #000000 100%);
}

/* Subtle Shimmer Animation for the Header */
@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

/* Premium Gold Text Styling */
.gold-header {
    text-align: center;
    background: linear-gradient(90deg, #8a6d3b, #f1d384, #8a6d3b, #e1b941, #8a6d3b);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem;
    font-weight: bold;
    margin-bottom: 0px;
    animation: shimmer 5s linear infinite;
}

.gold-subtext {
    text-align: center;
    color: #c5a059;
    letter-spacing: 4px;
    font-size: 0.8rem;
    text-transform: uppercase;
    margin-bottom: 30px;
}

/* Card Styling */
.card {
    padding: 40px;
    border-radius: 4px; /* Sharper corners for a more formal look */
    text-align: center;
    font-size: 28px;
    border: 1px solid #c5a059;
    background: rgba(10, 10, 10, 0.8);
    box-shadow: inset 0 0 15px rgba(197, 160, 89, 0.1), 0 10px 30px rgba(0,0,0,0.5);
    margin-top: 20px;
}

.positive { color: #f1d384; text-shadow: 0 0 10px rgba(241, 211, 132, 0.5); }
.negative { color: #ff4b4b; text-shadow: 0 0 10px rgba(255, 75, 75, 0.3); }
.neutral { color: #ffffff; }

/* Custom Button - Times New Roman & Gold */
div.stButton > button {
    background: transparent !important;
    color: #f1d384 !important;
    border: 1px solid #f1d384 !important;
    font-size: 1.2rem !important;
    padding: 10px 20px !important;
    transition: all 0.4s ease !important;
    width: 100%;
    border-radius: 0px !important;
}

div.stButton > button:hover {
    background: #f1d384 !important;
    color: black !important;
    box-shadow: 0 0 20px rgba(241, 211, 132, 0.4) !important;
}

/* Text Area Styling */
textarea {
    background-color: transparent !important;
    color: #f1d384 !important;
    border: 1px solid #444 !important;
    font-size: 1.2rem !important;
}

/* Custom Horizontal Line */
hr {
    border: 0;
    height: 1px;
    background-image: linear-gradient(to right, rgba(197, 160, 89, 0), rgba(197, 160, 89, 0.75), rgba(197, 160, 89, 0));
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOGIC ----------------
MODEL_PATH = "model/sentiment_model.pkl"
VECT_PATH = "model/vectorizer.pkl"

@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECT_PATH)
    return None, None

model, vectorizer = load_assets()

if model is None:
    st.error("Model assets missing.")
    st.stop()

# ---------------- HEADER ----------------
st.markdown("<h1 class='gold-header'>The Sentinel</h1>", unsafe_allow_html=True)
st.markdown("<p class='gold-subtext'>Elite Linguistic Analysis</p>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
tweet = st.text_area("Input Text for Analysis:", height=150, placeholder="Enter text...")

# ---------------- BUTTON ----------------
if st.button("EXECUTE ANALYSIS"):
    if tweet.strip() == "":
        st.warning("Please enter text.")
    else:
        with st.spinner("Processing..."):
            time.sleep(1)
            cleaned = clean_text(tweet)
            vector = vectorizer.transform([cleaned])
            probs = model.predict_proba(vector)[0]
            max_prob = probs.max()
            prediction = model.classes_[probs.argmax()]

        st.markdown("<hr>", unsafe_allow_html=True)

        # ---------------- RESULTS ----------------
        if max_prob < 0.6:
            st.markdown("<div class='card neutral'>NEUTRAL</div>", unsafe_allow_html=True)
        elif prediction.lower() == "positive":
            st.markdown("<div class='card positive'>POSITIVE</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card negative'>NEGATIVE</div>", unsafe_allow_html=True)

        # Gold Confidence Bar
        st.markdown(f"""
            <div style="margin-top:20px; font-size: 1.1rem; color: #c5a059;">
                Confidence Metric: {max_prob:.2%}
                <div style="width: 100%; background: #222; height: 3px; margin-top: 10px;">
                    <div style="width: {max_prob*100}%; background: #f1d384; height: 3px; box-shadow: 0 0 10px #f1d384;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<br><br><p style='text-align:center; color:#444; font-style: italic;'>Est. 2026 • Machine Intelligence Division</p>", unsafe_allow_html=True)
