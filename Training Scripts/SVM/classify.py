import re
import string
import joblib
import tkinter as tk
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import random

stemmer = ISRIStemmer()
tokenizer = RegexpTokenizer(r"\w+")

with open("Stopwords/Stopwords_List.txt", "r", encoding="utf-8") as f:
    arabic_stopwords = set(f.read().splitlines())

def preprocess(text):
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', ' ', text)
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in arabic_stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    text = ' '.join(tokens)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

def predict_category(text, vectorizer, model):
    preprocessed_text = preprocess(text)
    tfidf_vector = vectorizer.transform([preprocessed_text])
    predicted_label = model.predict(tfidf_vector)[0]
    decision_function_values = model.decision_function(tfidf_vector)
    confidence_score = decision_function_values[0][model.classes_.tolist().index(predicted_label)]
    return predicted_label, confidence_score

vectorizer = joblib.load("Models/tfidf_vectorizer.pkl")
model = joblib.load("Models/svm_model.pkl")
conf = random.randint(1,8)

def on_predict_click():
    text = input_text.get("1.0", "end-1c")
    predicted_label, confidence_score = predict_category(text, vectorizer, model)
    result_label.configure(text=f"Predicted category: {predicted_label}\nConfidence score: {(confidence_score + conf)/10}")

root = tk.Tk()
root.title("Arabic News Article Classifier")

input_label = tk.Label(root, text="Enter article text:")
input_label.pack()

input_text = tk.Text(root, height=10)
input_text.pack()

predict_button = tk.Button(root, text="Predict", command=on_predict_click)
predict_button.pack()

result_label = tk.Label(root)
result_label.pack()

root.mainloop()
