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
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

stemmer = ISRIStemmer()
tokenizer = nltk.RegexpTokenizer(r"\w+")

with open(Path("Stopwords/Stopwords_List.txt"), "r", encoding="utf-8") as f:
    arabic_stopwords = set(f.read().splitlines())

dataset_folders = {
    "Politics": "Dataset/train/Politics",
    "Entertainment": "Dataset/train/Entertainment",
    "Economy": "Dataset/train/Economy",
    "Sports": "Dataset/train/Sports"
}
categories= ["Politics","Entertainment","Economy","Sports"]

w2v_params = {
    "vector_size": 300, 
    "window": 5,     
    "min_count": 5,  
    "workers": 7     
}

dnn_params = {
    "embedding_size": w2v_params["vector_size"],
    "dropout_rate": 0.3,
    "hidden_units": 100,
    "batch_size": 72,
    "epochs": 10,
    "patience": 10
}

def Tokenize_Categories(text):
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', ' ', text)
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in arabic_stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [token.translate(str.maketrans("", "", string.punctuation)) for token in tokens]
    tokens = [token for token in tokens if not token.isdigit()]
    tokens = [token for token in tokens if token]
    return tokens

def Display_Passed_Tokens():
    word_counts = {}
    for category, category_texts in texts.items():
        word_counts[category] = {}
        for tokens in category_texts:
            for token in tokens:
                if token not in word_counts[category]:
                    word_counts[category][token] = 1
                else:
                    word_counts[category][token] += 1

    for category, word_count_dict in word_counts.items():
        with open(f"Preprocessed_Text/Passed/{category}_word_counts.txt", "w", encoding="utf-8") as f:
            for token, count in sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{token}: {count}\n")

def Display_Removed_Tokens():
    removed_tokens = {}
    for category, category_texts in texts.items():
        removed_tokens[category] = {}
        for tokens in category_texts:
            for token in tokens:
                if token in arabic_stopwords or not token.isalpha():
                    if token not in removed_tokens[category]:
                        removed_tokens[category][token] = 1
                    else:
                        removed_tokens[category][token] += 1

    for category, token_counts in removed_tokens.items():
        with open(f"Preprocessed_Text/Removed/{category}_removed_tokens.txt", "w", encoding="utf-8") as f:
            for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{token}: {count}\n")

texts = {}
removed_tokens = {}
for category, folder_path in dataset_folders.items():
    texts[category] = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            text = f.read()
            tokens = Tokenize_Categories(text)
            texts[category].append(tokens)
            for token in set(tokens):
                if token not in removed_tokens:
                    removed_tokens[token] = 1
                else:
                    removed_tokens[token] += 1

Display_Passed_Tokens()
Display_Removed_Tokens()

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

w2v_model = Word2Vec(sentences=train_texts, **w2v_params)
w2v_model.save("Models/DNN/word2vec.model")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

max_sequence_length = max([len(sequence) for sequence in train_sequences])
train_data = pad_sequences(train_sequences, maxlen=max_sequence_length, padding="post")
test_data = pad_sequences(test_sequences, maxlen=max_sequence_length, padding="post")

label_to_index = {label: index for index, label in enumerate(set(all_labels))}
train_labels = [label_to_index[label] for label in train_labels]
test_labels = [label_to_index[label] for label in test_labels]
train_labels = np.eye(len(label_to_index))[train_labels]
test_labels = np.eye(len(label_to_index))[test_labels]

from keras.layers import Flatten

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=dnn_params["embedding_size"], input_length=max_sequence_length))
model.add(Flatten())
model.add(Dense(dnn_params["hidden_units"], activation="relu"))
model.add(Dropout(dnn_params["dropout_rate"]))
model.add(Dense(len(label_to_index), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
early_stopping = EarlyStopping(monitor="val_loss", patience=dnn_params["patience"], mode="min", restore_best_weights=True)
model_checkpoint = ModelCheckpoint("Models/DNN/dnn_model.h5", monitor="val_loss", mode="min", save_best_only=True)
history = model.fit(train_data, train_labels, batch_size=dnn_params["batch_size"], epochs=dnn_params["epochs"], callbacks=[early_stopping, model_checkpoint], validation_split=0.2)

test_loss, test_accuracy = model.evaluate(test_data, test_labels, batch_size=dnn_params["batch_size"])
test_predictions = model.predict(test_data)
test_predictions = np.argmax(test_predictions, axis=1)
test_labels = np.argmax(test_labels, axis=1)

print("Test accuracy:", test_accuracy)
print("Test precision:", precision_score(test_labels, test_predictions, average="macro"))
print("Test recall:", recall_score(test_labels, test_labels, average="macro"))
print("Test f1 score:", f1_score(test_labels, test_predictions, average="macro"))