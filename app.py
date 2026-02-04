import streamlit as st
import joblib
import os
import time
from preprocess import clean_text

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="The Sentinel | Royal Sentiment AI",
    page_icon="⚜️",
    layout="centered"
)

# ---------------- ADVANCED AESTHETIC CSS ----------------
st.markdown("""
<style>
/* Global Font Reset to Times New Roman */
* {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Luxury Radial Background with Animated Mesh */
[data-testid="stAppViewContainer"] {
    background-color: #050505;
    background-image: 
        radial-gradient(at 0% 0%, rgba(138, 109, 59, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(0, 51, 17, 0.1) 0px, transparent 50%);
}

/* Gold Text Shimmer Effect */
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
    letter-spacing: -2px;
}

.sub-title {
    text-align: center;
    color: #c5a059;
    font-size: 0.9rem;
    letter-spacing: 6px;
    text-transform: uppercase;
    margin-top: -10px;
    opacity: 0.8;
}

/* Premium Input Box (Glassmorphism) */
.stTextArea textarea {
    background: rgba(255, 255, 255, 0.03) !important;
    color: #f1d384 !important;
    border: 1px solid rgba(197, 160, 89, 0.3) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    font-size: 1.1rem !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.stTextArea textarea:focus {
    border-color: #f1d384 !important;
    box-shadow: 0 0 15px rgba(241, 211, 132, 0.2) !important;
}

/* Royal Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #c5a059 0%, #8a6d3b 100%) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 12px 40px !important;
    font-weight: bold !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
}

div.stButton > button:hover {
    transform: scale(1.02) translateY(-2px);
    box-shadow: 0 10px 20px rgba(138, 109, 59, 0.4) !important;
}

/* Sentiment Cards - Modern Glassmorphism */
.result-card {
    padding: 50px;
    border-radius: 20px;
    text-align: center;
    font-size: 2.5rem;
    font-weight: bold;
    letter-spacing: 4px;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 25px 50px rgba(0,0,0,0.5);
    margin: 20px 0;
}

.positive {
    background: linear-gradient(135deg, rgba(0, 100, 0, 0.3), rgba(0, 51, 17, 0.6));
    color: #4ade80;
    border-color: rgba(74, 222, 128, 0.4);
}

.negative {
    background: linear-gradient(135deg, rgba(139, 0, 0, 0.3), rgba(51, 0, 0, 0.6));
    color: #f87171;
    border-color: rgba(248, 113, 113, 0.4);
}

.neutral {
    background: linear-gradient(135deg, rgba(138, 109, 59, 0.2), rgba(20, 20, 20, 0.8));
    color: #f1d384;
    border-color: rgba(197, 160, 89, 0.4);
}

/* Gold Progress Bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #8a6d3b, #f1d384) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA LOGIC ----------------
MODEL_PATH = "model/sentiment_model.pkl"
VECT_PATH = "model/vectorizer.pkl"

@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECT_PATH)
    return None, None

model, vectorizer = load_assets()

if model is None:
    st.error("Model assets missing from the royal archive.")
    st.stop()

# ---------------- HEADER ----------------
st.markdown("<h1 class='gold-title'>The Sentinel</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Precision Linguistic Intelligence</p>", unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
tweet = st.text_area("", height=150, placeholder="Transcribe your message for the Sentinel...")

# ---------------- ACTION ----------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze = st.button("Analyze Sentiment", use_container_width=True)

if analyze:
    if not tweet.strip():
        st.toast("Input text is required.", icon="⚠️")
    else:
        with st.spinner("Decoding Royal Cipher..."):
            time.sleep(1)
            cleaned = clean_text(tweet)
            vector = vectorizer.transform([cleaned])
            probs = model.predict_proba(vector)[0]
            max_prob = probs.max()
            prediction = model.classes_[probs.argmax()]

        # ---------------- DISPLAY RESULTS ----------------
        if max_prob < 0.6:
            st.markdown("<div class='result-card neutral'>⚜️ NEUTRAL ⚜️</div>", unsafe_allow_html=True)
        elif prediction.lower() == "positive":
            st.markdown("<div class='result-card positive'>🌿 ROYAL GREEN 🌿</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-card negative'>🥀 ROYAL RED 🥀</div>", unsafe_allow_html=True)

        # Aesthetic Confidence Meter
        st.markdown(f"""
            <div style='text-align: center; padding: 20px;'>
                <p style='color: #c5a059; letter-spacing: 2px; margin-bottom: 5px;'>CONFIDENCE SCORE: {max_prob:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
        st.progress(float(max_prob))

# ---------------- FOOTER ----------------
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; border-top: 1px solid rgba(197, 160, 89, 0.2); padding-top: 20px;'>
        <p style='color: #444; font-size: 0.8rem; font-style: italic;'>
            Licensed for the Machine Intelligence Division • 2026
        </p>
    </div>
""", unsafe_allow_html=True)
