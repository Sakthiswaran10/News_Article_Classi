import pandas as pd
import numpy as np
import nltk
import re
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

nltk.download("stopwords")

df_fake = pd.read_csv(r"D:\Elevate labs\News_Article_Classifier\News_Article_Classi\Dataset\Fake.csv")
df_fake["label"] = 0

df_real = pd.read_csv(r"D:\Elevate labs\News_Article_Classifier\News_Article_Classi\Dataset\True.csv")
df_real["label"] = 1

df = pd.concat([df_fake, df_real], ignore_index=True)
df = df[["text", "label"]]

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    try:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = re.findall(r'\b\w+\b', text)  
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word.isalpha()]
        return " ".join(tokens)
    except Exception as e:
        print(f"[Error cleaning text]: {e}")
        return ""

df["cleaned"] = df["text"].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["cleaned"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import os
os.makedirs("model", exist_ok=True)
with open("model/news_model.pkl", "wb") as f:
    pickle.dump((model, tfidf), f)

