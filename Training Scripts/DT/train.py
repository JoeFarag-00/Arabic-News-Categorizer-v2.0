import os
import string
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

stemmer = ISRIStemmer()
tokenizer = nltk.RegexpTokenizer(r"\w+")

print ("\ninitializing DT model training...")


with open(Path("Stopwords/Stopwords_List.txt"), "r", encoding="utf-8") as f:
    arabic_stopwords = set(f.read().splitlines())

dataset_folders = {
    "Politics": "Dataset/Train/Politics",
    "Entertainment": "Dataset/Train/Entertainment",
    "Economy": "Dataset/Train/Economy",
    "Sports": "Dataset/Train/Sports"
}


def Tokenize_Categories(text):
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', ' ', text)
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in arabic_stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [token.translate(str.maketrans(
        "", "", string.punctuation)) for token in tokens]
    tokens = [token for token in tokens if not token.isdigit()]
    tokens = [token for token in tokens if token]
    return " ".join(tokens)

print ("\nPreprocessing Dataset...")
texts = []
categories = []
for category, folder_path in dataset_folders.items():
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            text = f.read()
            texts.append(Tokenize_Categories(text))
            categories.append(category)

vectorizer = TfidfVectorizer()

print ("\nRunning DT Model...")
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X, categories, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1 score:", f1_score(y_test, y_pred, average="weighted"))

print ("\nSaved TF-IDF and DT weights...")
joblib.dump(vectorizer, "Models/DT/DT_tfidf_vectorizer.pkl")
joblib.dump(clf, "Models/DT/DT_model.pkl")