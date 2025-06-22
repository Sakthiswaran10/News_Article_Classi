# 📰 News Article Classifier

This is a machine learning project that classifies news articles as **Fake** or **Real** using natural language processing (NLP) techniques and a Multinomial Naive Bayes classifier. It includes a training pipeline and a user-friendly web interface built with Streamlit.

---

## 🚀 Features

- Combines real and fake news datasets.
- Cleans and preprocesses text using NLTK (lowercasing, punctuation removal, stopwords, stemming).
- Converts text to numerical features using TF-IDF vectorization.
- Trains a Multinomial Naive Bayes model to classify news.
- Saves and loads models using `pickle`.
- Provides an interactive Streamlit web app for real-time predictions.

---

## 🗂️ Project Structure

News_Article_Classifier/
├── model/
│ └── news_model.pkl # Trained model and TF-IDF vectorizer
├── Fake.csv # Fake news dataset
├── True.csv # Real news dataset
├── news_classifier.py # Script to train and save the model
├── streamlit_app.py # Streamlit web application
└── README.md # Project documentation


---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Sakthiswaran10/News_Article_Classi.git
cd News_Article_Classi

2. Install Required Libraries
pip install pandas numpy scikit-learn nltk streamlit

3. Download NLTK Resources
Open a Python shell and run:

import nltk
nltk.download('stopwords')
nltk.download('punkt')

📚 Dataset
This project uses news datasets containing fake and real articles.

Fake.csv: Contains fake news articles.

True.csv: Contains real news articles.

Source: Kaggle – Fake and Real News Dataset

Each row contains article text and other metadata. We only use the text column for training.

🏋️ Train the Model
To train the classifier and save the model and vectorizer, run:

python news_classifier.py

This script will:

Preprocess the text.

Vectorize it using TF-IDF.

Train the model.

Evaluate its performance (accuracy, F1 score).

Save the trained model and vectorizer as model/news_model.pkl.

Example Output:
Accuracy: 0.926
F1 Score: 0.922

🌐 Run the Streamlit Web App
After training the model:
streamlit run streamlit_app.py
It will launch a web interface in your browser at http://localhost:8501.

You can paste any news article into the text box and click Predict to check if it's real or fake.

🧪 Example Use
Input News:
NASA's Perseverance rover has successfully collected its 20th rock sample on Mars...

Output:
✅ Prediction: REAL News

🛠️ Built With
Python 3

Pandas & NumPy

Scikit-learn

NLTK

Streamlit

💡 Future Improvements
Use advanced models like BERT for better accuracy.

Add probability/confidence score to predictions.

Deploy online via Streamlit Cloud or Hugging Face Spaces.

Extend to detect satire or biased articles.