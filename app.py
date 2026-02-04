import streamlit as st
import joblib
import os
import time
from preprocess import clean_text

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="The Sentinel | Royal AI",
    page_icon="⚜️",
    layout="centered"
)

# ---------------- ADVANCED ROYAL CSS ----------------
st.markdown("""
<style>
/* Global Font Reset to Times New Roman */
* {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Deep Obsidian Canvas */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at center, #0f0f0f 0%, #000000 100%);
}

/* Gold Shimmer Header */
@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

.gold-title {
    text-align: center;
    font-size: 4.5rem;
    font-weight: 700;
    margin-bottom: 0;
    background: linear-gradient(90deg, #8a6d3b, #f1d384, #f9e5af, #f1d384, #8a6d3b);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 8s linear infinite;
    letter-spacing: -1px;
}

.sub-title {
    text-align: center;
    color: #c5a059;
    font-size: 0.8rem;
    letter-spacing: 7px;
    text-transform: uppercase;
    margin-top: -10px;
    opacity: 0.7;
}

/* Glassmorphism Input */
.stTextArea textarea {
    background: rgba(255, 255, 255, 0.02) !important;
    color: #f1d384 !important;
    border: 1px solid rgba(197, 160, 89, 0.2) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    font-size: 1.2rem !important;
}

/* Royal Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #c5a059 0%, #8a6d3b 100%) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 12px 0 !important;
    font-weight: bold !important;
    text-transform: uppercase;
    letter-spacing: 3px;
    width: 100%;
    transition: 0.4s all ease;
}

div.stButton > button:hover {
    box-shadow: 0 0 25px rgba(138, 109, 59, 0.5) !important;
    transform: translateY(-2px);
}

/* --- REFINED SMALL ETCHED BOX DESIGN --- */
.result-card {
    padding: 25px 15px; /* Reduced padding for smaller size */
    border-radius: 15px;
    text-align: center;
    font-size: 2rem;    /* Reduced font size for sleekness */
    font-weight: 700;
    letter-spacing: 10px;
    margin: 20px auto;
    max-width: 500px;   /* Constrains the width */
    background: #1a1a1a; 
    border: 1px solid rgba(197, 160, 89, 0.35);
    box-shadow: 0 10px 30px rgba(0,0,0,0.8);
}

.positive {
    color: #4ade80;
    text-shadow: 0 0 15px rgba(74, 222, 128, 0.2);
}

.negative {
    color: #f87171;
    text-shadow: 0 0 15px rgba(248, 113, 113, 0.2);
}

.neutral {
    color: #f1d384;
    text-shadow: 0 0 15px rgba(241, 211, 132, 0.2);
}

.confidence-text {
    text-align: center;
    color: #c5a059;
    font-size: 1rem;
    font-weight: bold;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 10px;
}

/* Progress Bar Customization */
div[data-testid="stProgress"] > div > div > div > div {
    background-color: #c5a059 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIC ----------------
MODEL_PATH = "model/sentiment_model.pkl"
VECT_PATH = "model/vectorizer.pkl"

@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECT_PATH)
    return None, None

model, vectorizer = load_assets()

# ---------------- UI ----------------
st.markdown("<h1 class='gold-title'>The Sentinel</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Imperial Linguistic Intelligence</p>", unsafe_allow_html=True)

tweet = st.text_area("", height=150, placeholder="Transcribe the message for analysis...")

if st.button("Execute Analysis"):
    if model is None or vectorizer is None:
        st.error("Imperial Archives (Model Files) not found in the /model directory.")
    elif not tweet.strip():
        st.error("The archive requires a proclamation (text input).")
    else:
        with st.spinner("Decoding Dialect..."):
            time.sleep(0.8)
            cleaned = clean_text(tweet)
            vector = vectorizer.
