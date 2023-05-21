import os
import string
import re
import nltk
import numpy as np
from nltk.stem import ISRIStemmer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, TimeDistributed, Dropout,Lambda, Concatenate, Reshape
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from pathlib import Path


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

han_params = {
    "embedding_size": w2v_params["vector_size"],
    "lstm_units": 64,
    "dropout_rate": 0.4,
    "batch_size": 32,
    "epochs": 5,
    "patience": 3
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
w2v_model.save("Models/HAN/word2vec.model")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

max_sentence_length = 30
max_word_length = 30
train_data = pad_sequences(train_sequences, maxlen=max_sentence_length * max_word_length, padding="post")
test_data = pad_sequences(test_sequences, maxlen=max_sentence_length * max_word_length, padding="post")

train_data = train_data.reshape(-1, max_sentence_length, max_word_length)
test_data = test_data.reshape(-1, max_sentence_length, max_word_length)

label_to_index = {label: index for index, label in enumerate(set(all_labels))}
train_labels = [label_to_index[label] for label in train_labels]
test_labels = [label_to_index[label] for label in test_labels]
train_labels = np.eye(len(label_to_index))[train_labels]
test_labels = np.eye(len(label_to_index))[test_labels]

word_embeddings = np.zeros((len(tokenizer.word_index) + 1, w2v_params["vector_size"]))
for word, index in tokenizer.word_index.items():
    if word in w2v_model.wv:
        word_embeddings[index] = w2v_model.wv[word]

word_input = Input(shape=(max_word_length,))
word_embedding = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=han_params["embedding_size"], weights=[word_embeddings])(word_input)
word_lstm = Bidirectional(LSTM(units=han_params["lstm_units"], return_sequences=True))(word_embedding)
word_attention = TimeDistributed(Dense(1, activation='tanh'))(word_lstm)
word_attention = Reshape((max_word_length,))(word_attention)
word_attention = Dropout(han_params["dropout_rate"])(word_attention)
word_attention = Dense(max_word_length, activation='softmax')(word_attention)
word_attention = Reshape((max_word_length, 1))(word_attention)
word_representation = Concatenate(axis=-1)([word_lstm, word_attention])
word_representation = Lambda(lambda x: K.sum(x, axis=1))(word_representation)
word_encoder = Model(inputs=word_input, outputs=word_representation)

sentence_input = Input(shape=(max_sentence_length, max_word_length,))
sentence_encoder = TimeDistributed(word_encoder)(sentence_input)
sentence_lstm = Bidirectional(LSTM(units=han_params["lstm_units"], return_sequences=True))(sentence_encoder)
sentence_attention = TimeDistributed(Dense(1, activation='tanh'))(sentence_lstm)
sentence_attention = Reshape((max_sentence_length,))(sentence_attention)
sentence_attention = Dropout(han_params["dropout_rate"])(sentence_attention)
sentence_attention = Dense(max_sentence_length, activation='softmax')(sentence_attention)
sentence_attention = Reshape((max_sentence_length, 1))(sentence_attention)
sentence_representation = Concatenate(axis=-1)([sentence_lstm, sentence_attention])
sentence_representation = Lambda(lambda x: K.sum(x, axis=1))(sentence_representation)
dropout = Dropout(han_params["dropout_rate"])(sentence_representation)
output = Dense(units=len(label_to_index), activation="softmax")(dropout)
model = Model(inputs=sentence_input, outputs=output)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor="val_loss", patience=han_params["patience"])
checkpoint = ModelCheckpoint("Models/HAN/han_model.h5", save_best_only=True)

model.fit(train_data, train_labels, batch_size=han_params["batch_size"], epochs=han_params["epochs"], validation_split=0.2, callbacks=[early_stopping, checkpoint])

model.load_weights("Models/HAN/best_model.h5")
predictions = np.argmax(model.predict(test_data), axis=-1)
test_labels = np.argmax(test_labels, axis=-1)

print(f"Accuracy: {accuracy_score(predictions, test_labels)}")
print(f"Precision: {precision_score(predictions, test_labels, average='macro')}")
print(f"Recall: {recall_score(predictions, test_labels, average='macro')}")
print(f"F1-Score: {f1_score(predictions, test_labels, average='macro')}")