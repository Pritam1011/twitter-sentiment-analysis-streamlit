import re
import string
import nltk
from nltk.corpus import stopwords

# download only once
nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()

    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]

    return " ".join(words)
