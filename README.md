# 🐦 Twitter Sentiment Analysis Dashboard

An interactive Streamlit-based web application that performs sentiment analysis on text using Natural Language Processing (NLP) and Machine Learning techniques.

---

## 🚀 Live Demo

🔗 **Click here to try the application:**  
👉 https://twitter-sentiment-analysis-app-ep9sde2anqkwdojfqkesvo.streamlit.app/

---

## 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- NLTK (Natural Language Toolkit)
- Streamlit
- Joblib (Model Serialization)

---

## ✨ Features

- Text preprocessing using NLTK
- TF-IDF based feature extraction
- Logistic Regression classifier
- Real-time sentiment prediction
- Interactive and responsive Streamlit dashboard
- Confidence-based Neutral sentiment handling
- Pre-trained model and vectorizer saved for reuse

---

## 📂 Project Structure

- `app.py` – Main Streamlit application
- `preprocess.py` – Text preprocessing logic
- `train_model.py` – Model training script
- `model/` – Saved ML model and vectorizer
- `data/` – Dataset files
- `requirements.txt` – Python dependencies
- `.gitignore` – Ignored files and folders

---

## 🤖 Model Details

- **Vectorization:** TF-IDF (Term Frequency – Inverse Document Frequency)
- **Algorithm:** Logistic Regression
- **Evaluation Accuracy:** ~81%
- **Output Classes:** Positive, Negative, Neutral

---

## 🧠 How It Works

1. User enters text in the Streamlit interface.
2. Text is cleaned and preprocessed.
3. Text is converted into numerical form using TF-IDF.
4. The trained Logistic Regression model predicts sentiment.
5. Result is displayed with confidence score.

---

## 📦 Installation (Run Locally)

```bash
git clone https://github.com/Pritam1011/twitter-sentiment-analysis-streamlit.git
cd twitter-sentiment-analysis-streamlit
pip install -r requirements.txt
streamlit run app.py
