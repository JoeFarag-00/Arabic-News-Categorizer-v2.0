import os
import string
import re
import nltk
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from pathlib import Path
import tkinter as tk

stemmer = ISRIStemmer()
tokenizer = nltk.RegexpTokenizer(r"\w+")

with open(Path("Stopwords/Stopwords_List.txt"), "r", encoding="utf-8") as f:
    arabic_stopwords = set(f.read().splitlines())

w2v_params = {
    "vector_size": 100, 
    "window": 5,     
    "min_count": 5,  
    "workers": 7     
}

dnn_params = {
    "embedding_size": w2v_params["vector_size"],
    "dropout_rate": 0.4,
    "hidden_units": 62,
    "batch_size": 72,
    "epochs": 10,
    "patience": 10
}

def Tokenize_Article(text):
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', ' ', text)
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in arabic_stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [token.translate(str.maketrans("", "", string.punctuation)) for token in tokens]
    tokens = [token for token in tokens if not token.isdigit()]
    tokens = [token for token in tokens if token]
    return tokens

def Load_Word2Vec_Model():
    w2v_model = Word2Vec.load("Models/DNN/word2vec.model")
    return w2v_model

def Load_DNN_Model():
    dnn_model = load_model("Models/DNN/dnn_model.h5")
    return dnn_model

def Classify_Article(article_text):
    label_to_index = {"Politics": 0, "Entertainment": 1, "Economy": 2, "Sports": 3}
    article_tokens = Tokenize_Article(article_text)
    max_sequence_length = Load_DNN_Model().input_shape[1]
    article_sequences = Tokenizer().texts_to_sequences([article_tokens])
    article_data = pad_sequences(article_sequences, maxlen=max_sequence_length, padding="post")
    predicted_probabilities = Load_DNN_Model().predict(article_data)[0]
    predicted_label_index = np.argmax(predicted_probabilities)
    predicted_label = [label for label, index in label_to_index.items() if index == predicted_label_index][0]
    return predicted_label

def Submit_Article():
    article_text = article_textbox.get("1.0", "end-1c")
    predicted_label = Classify_Article(article_text)
    result_label.config(text=f"Predicted Category: {predicted_label}")

root = tk.Tk()
root.title("Arabic News Classifier")

article_textbox = tk.Text(root, height=20, width=50)
article_textbox.pack()

submit_button = tk.Button(root, text="Submit Article", command=Submit_Article)
submit_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()