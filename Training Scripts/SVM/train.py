import os
import string
import re
import nltk
import numpy as np
import joblib
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
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

def preprocess(text):
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', ' ', text)
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in arabic_stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    text = ' '.join(tokens)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

texts = {}
for category, folder in dataset_folders.items():
    texts[category] = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            text = f.read()
            text = preprocess(text)
            texts[category].append(text)

all_texts = []
all_labels = []
for category, category_texts in texts.items():
    all_texts.extend(category_texts)
    all_labels.extend([category] * len(category_texts))

train_texts, test_texts, train_labels, test_labels = train_test_split(all_texts, all_labels, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=preprocess)),
    ('svm', SVC(class_weight='balanced'))
])

param_grid = {
    'tfidf__max_features': [1000, 5000, 10000],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf']
}

grid = GridSearchCV(pipeline, param_grid, cv=3)
grid.fit(train_texts, train_labels)

best_pipeline = grid.best_estimator_
joblib.dump(best_pipeline.named_steps['tfidf'], "Models/tfidf_vectorizer.pkl")
joblib.dump(best_pipeline.named_steps['svm'], "Models/svm_model.pkl")

test_predictions = best_pipeline.predict(test_texts)

print("Test accuracy:", accuracy_score(test_labels, test_predictions))
print("Test precision:", precision_score(test_labels, test_predictions, average="macro"))
print("Test recall:", recall_score(test_labels, test_predictions, average="macro"))
print("Test f1 score:", f1_score(test_labels, test_predictions, average="macro"))
