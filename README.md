# Twitter Sentiment Analysis Dashboard

An interactive Streamlit-based web application that performs sentiment analysis on Twitter text using Natural Language Processing (NLP) and Machine Learning techniques.

The system classifies input text into Positive, Negative, or Neutral sentiment based on model confidence.

---

## Features

- Text preprocessing using NLTK  
- Sentiment classification using Machine Learning  
- TF-IDF based feature extraction  
- Interactive and responsive Streamlit dashboard  
- Confidence-based Neutral sentiment handling  
- Trained model and vectorizer saved for reuse  

---

## Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- NLTK  
- Streamlit  

---

## Model Details

- Vectorization: TF-IDF (Term Frequency–Inverse Document Frequency)  
- Classifier: Logistic Regression  
- Evaluation Accuracy: ~81%  
- Neutral Logic:  
  If prediction confidence < 0.6, sentiment is shown as Neutral  

---

## Example Test Inputs

Positive:
I absolutely love this phone! Amazing performance.

Negative:
This product is terrible and completely useless.

Neutral:
The phone was delivered yesterday.

---

## Notes

- The venv/ folder is excluded using .gitignore  
- Model files are loaded using joblib  
- Designed for both local execution and Streamlit Cloud deployment  

---

## License

This project is for educational and learning purposes.
