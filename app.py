import streamlit as st
import joblib
import os
import time
from preprocess import clean_text

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Royal Sentiment Analytics",
    page_icon="👑",
    layout="centered"
)

# ---------------- ROYAL UI STYLING ----------------
st.markdown("""
<style>
/* Global Times New Roman Force */
* {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Deep Obsidian Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%);
}

/* Metallic Shimmer for Headers */
@keyframes gold-glow {
    0% { filter: brightness(100%) contrast(100%); }
    50% { filter: brightness(130%) contrast(110%); }
    100% { filter: brightness(100%) contrast(100%); }
}

.royal-header {
    text-align: center;
    background: linear-gradient(to bottom, #cfb53b 0%, #8a6d3b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.8rem;
    font-weight: bold;
    letter-spacing: -1px;
    margin-bottom: 0px;
    animation: gold-glow 4s ease-in-out infinite;
}

.royal-subtext {
    text-align: center;
    color: #8a6d3b;
    letter-spacing: 5px;
    font-size: 0.9rem;
    text-transform: uppercase;
    border-bottom: 1px solid #8a6d3b;
    display: table;
    margin: 0 auto 40px auto;
    padding-bottom: 5px;
}

/* Sentiment Cards */
.card {
    padding: 40px;
    border-radius: 2px;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-top: 25px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.8);
}

.positive-royal { 
    background: linear-gradient(135deg, #003311 0%, #006400 100%);
    color: #90ee90;
    border: 2px solid #00a300;
}

.negative-royal { 
    background: linear-gradient(135deg, #330000 0%, #8b0000 100%);
    color: #ff9999;
    border: 2px solid #ff0000;
}

.neutral-royal { 
    background: linear-gradient(135deg, #1a1a1a 0%, #333333 100%);
    color: #cfb53b;
    border: 2px solid #cfb53b;
}

/* Input Area */
textarea {
    background-color: #0f0f0f !important;
    color: #cfb53b !important;
    border: 1px solid #333 !important;
    border-left: 5px solid #cfb53b !important;
}

/* Royal Button */
div.stButton > button {
    background: #cfb53b !important;
    color: #000 !important;
    border: none !important;
    font-size: 1.1rem !important;
    font-weight: bold !important;
    padding: 15px !important;
    width: 100%;
    transition: 0.3s all ease;
    border-radius: 0px !important;
}

div.stButton > button:hover {
    background: #f1d384 !important;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(207, 181, 59, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------
MODEL_PATH = "model/sentiment_model.pkl"
VECT_PATH = "model/vectorizer.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECT_PATH)
    return None, None

model, vectorizer = load_model()

# ---------------- HEADER ----------------
st.markdown("<h1 class='royal-header'>THE SENTINEL</h1>", unsafe_allow_html=True)
st.markdown("<p class='royal-subtext'>Imperial Text Classification</p>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
tweet = st.text_area("Proclaim your thoughts:", height=150, placeholder="Type here...")

# ---------------- BUTTON ----------------
if st.button("CLASSIFY SENTIMENT"):
    if not tweet.strip():
        st.warning("The archive requires input text.")
    else:
        with st.spinner("Analyzing Dialect..."):
            time.sleep(1)
            cleaned = clean_text(tweet)
            vector = vectorizer.transform([cleaned])
            probs = model.predict_proba(vector)[0]
            max_prob = probs.max()
            prediction = model.classes_[probs.argmax()]

        # ---------------- RESULTS ----------------
        if max_prob < 0.6:
            st.markdown("<div class='card neutral-royal'>⚜️ NEUTRAL ⚜️</div>", unsafe_allow_html=True)
        elif prediction.lower() == "positive":
            st.markdown("<div class='card positive-royal'>🌿 POSITIVE 🌿</div>", unsafe_allow_html=True)
        elif prediction.lower() == "negative":
            st.markdown("<div class='card negative-royal'>🥀 NEGATIVE 🥀</div>", unsafe_allow_html=True)

        # Confidence Bar
        st.markdown(f"""
            <div style="margin-top:30px; text-align:center;">
                <span style="color: #8a6d3b; font-size: 1.2rem;">Predictive Certainty: {max_prob:.2%}</span>
                <div style="width: 100%; background: #111; height: 4px; margin-top: 10px; border: 1px solid #333;">
                    <div style="width: {max_prob*100}%; background: #cfb53b; height: 100%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<br><br><p style='text-align:center; color:#333; font-style: italic; letter-spacing: 2px;'>NON-DUPLICABLE ROYAL AI ASSET</p>", unsafe_allow_html=True)
