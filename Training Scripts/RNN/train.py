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
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print ("\ninitializing CNN model training...")

stemmer = ISRIStemmer()
tokenizer = nltk.RegexpTokenizer(r"\w+")

with open(Path("Stopwords/Stopwords_List.txt"), "r", encoding="utf-8") as f:
    arabic_stopwords = set(f.read().splitlines())

dataset_folders = {
    "Politics": "Dataset/Train/Politics",
    "Entertainment": "Dataset/Train/Entertainment",
    "Economy": "Dataset/Train/Economy",
    "Sports": "Dataset/Train/Sports"
}

w2v_params = {
    "vector_size": 300,    
    "window": 5,     
    "min_count": 5,  
    "workers": 7     
}

lstm_params = {
    "embedding_size": w2v_params["vector_size"],
    "lstm_units": 64,
    "dropout_rate": 0.4,
    "hidden_units": 64,
    "batch_size": 20,
    "epochs": 2,
    "patience": 2
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

print ("\nPreprocessing Dataset...")

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
w2v_model.save("Models/LSTM/word2vec.model")
print ("\nSaved Word2vec weights...")

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

print ("\nRunning LSTM Model...")
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=lstm_params["embedding_size"], input_length=max_sequence_length))
model.add(Bidirectional(LSTM(units=lstm_params["lstm_units"], return_sequences=True)))
model.add(Dropout(lstm_params["dropout_rate"]))
model.add(Bidirectional(LSTM(units=lstm_params["lstm_units"])))
model.add(Dropout(lstm_params["dropout_rate"]))
model.add(Dense(lstm_params["hidden_units"], activation="relu"))
model.add(Dropout(lstm_params["dropout_rate"]))
model.add(Dense(len(label_to_index), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor="val_loss", patience=lstm_params["patience"], mode="min", restore_best_weights=True)
model_checkpoint = ModelCheckpoint("Models/LSTM/lstm_model.h5", monitor="val_loss", mode="min", save_best_only=True)

history = model.fit(train_data, train_labels, batch_size=lstm_params["batch_size"], epochs=lstm_params["epochs"], callbacks=[early_stopping, model_checkpoint], validation_split=0.2)

test_loss, test_accuracy = model.evaluate(test_data, test_labels, batch_size=lstm_params["batch_size"])
test_predictions = model.predict(test_data)
test_predictions = np.argmax(test_predictions, axis=1)
test_labels = np.argmax(test_labels, axis=1)

print("Test accuracy:", test_accuracy)
print("Test precision:", precision_score(test_labels, test_predictions, average="macro"))
print("Test recall:", recall_score(test_labels, test_predictions, average="macro"))
print("Test f1 score:", f1_score(test_labels, test_predictions, average="macro"))