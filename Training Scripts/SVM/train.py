import os
import string
import re
import nltk
import numpy as np
import joblib
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


stemmer = ISRIStemmer()
tokenizer = nltk.RegexpTokenizer(r"\w+")

with open("Stopwords/Stopwords_List.txt", "r", encoding="utf-8") as f:
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
    tokens = [token.translate(str.maketrans("", "", string.punctuation)) for token in tokens]
    tokens = [token for token in tokens if not token.isdigit()]
    tokens = [token for token in tokens if token]
    return " ".join(tokens)

texts = {}
for category, folder in dataset_folders.items():
    texts[category] = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            text = f.read()
            tokens = Tokenize_Categories(text)
            texts[category].append(tokens)

all_texts = []
all_labels = []
for category, category_texts in texts.items():
    all_texts.extend(category_texts)
    all_labels.extend([category] * len(category_texts))

train_texts, test_texts, train_labels, test_labels = train_test_split(all_texts, all_labels, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(tokenizer=Tokenize_Categories)
train_data = vectorizer.fit_transform(train_texts)
test_data = vectorizer.transform(test_texts)

label_to_index = {label: index for index, label in enumerate(set(all_labels))}
train_labels = [label_to_index[label] for label in train_labels]
test_labels = [label_to_index[label] for label in test_labels]

svm_params = {
    "C": 1.0,
    "kernel": "rbf",
    "class_weight": "balanced"
}

svm_model = SVC(**svm_params)
svm_model.fit(train_data, train_labels)

test_predictions = svm_model.predict(test_data)

joblib.dump(svm_model, "Models/svm_model.pkl")
print("Test accuracy:", accuracy_score(test_labels, test_predictions))
print("Test precision:", precision_score(test_labels, test_predictions, average="macro"))
print("Test recall:", recall_score(test_labels, test_predictions, average="macro"))
print("Test f1 score:", f1_score(test_labels, test_predictions, average="macro"))