# app.py

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Load model and vectorizer
with open("model/news_model.pkl", "rb") as f:
    model, tfidf = pickle.load(f)

# Text cleaning function
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    tokens = text.split()  # simple whitespace tokenizer
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(tokens)

# Streamlit UI
st.title("📰 News Article Classifier")
st.subheader("Enter a news article below to predict if it's Fake or Real.")

user_input = st.text_area("Enter News Article Text Here")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)

    if prediction[0] == 1:
        st.success("✅ Prediction: REAL News")
    else:
        st.error("❌ Prediction: FAKE News")
