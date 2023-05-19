import tkinter as tk
from tkinter import ttk
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

root = tk.Tk()
root.title("Arabic News Article Categorization")

# Load the saved vectorizer and trained classifier from .pkl files
dt_vec = joblib.load("Models/DT/DT_tfidf_vectorizer.pkl")
dt_model = joblib.load("Models/DT/DT_model.pkl")

stemmer = ISRIStemmer()
tokenizer = nltk.RegexpTokenizer(r"\w+")

with open("Stopwords/Stopwords_List.txt", "r", encoding="utf-8") as f:
    arabic_stopwords = set(f.read().splitlines())


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


def predict_category():
    input_text = input_textbox.get("1.0", "end-1c")
    if input_text:
        preprocessed_text = Tokenize_Categories(input_text)
        vectorized_text = dt_vec.transform([preprocessed_text])
        predicted_prob = dt_model.predict_proba(vectorized_text)[0]
        predicted_category = dt_model.predict(vectorized_text)[0]
        result_label.config(text=f"Predicted category: {predicted_category}, Confidence: {predicted_prob.max():.2%}")
    else:
        result_label.config(text="Please enter some text.")


input_textbox = tk.Text(root, height=10)
input_textbox.pack(pady=10)

submit_button = ttk.Button(root, text="Submit", command=predict_category)
submit_button.pack(pady=10)

result_label = ttk.Label(root, text="")
result_label.pack()

root.mainloop()