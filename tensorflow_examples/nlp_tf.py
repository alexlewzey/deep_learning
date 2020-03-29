"""making a spam classifiers with tf and word embeddings"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import plot

import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPool2D, Dropout, SimpleRNN, Conv2D, Embedding, \
    GlobalAveragePooling1D, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import pandas as pd
from tensorflow.python.keras.layers import GlobalMaxPooling1D

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv(r'C:\Users\alewz\Google Drive\data\spam.csv', encoding='latin-1')
data = data.iloc[:, :2]
data.columns = ['label', 'text']

data['b_label'] = data['label'].map({'ham': 0, 'spam': 1})
Y = data['b_label'].values

X_train, X_test, y_train, y_test = train_test_split(data['text'], Y, test_size=0.33)

# convert sentences into sequences #####################################################################################

MAX_VOCAB_SIZE = 20_000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

word2idx = tokenizer.word_index
V = len(word2idx)

data_train = pad_sequences(sequences=sequences_train)
T = data_train.shape[1]
data_test = pad_sequences(sequences=sequences_test, maxlen=T)


# embedding dimensionality
D = 20

# hidden state dimensionality
M = 15

i = Input(shape=(T,))
x = Embedding(V + 1, D)(i)
x = LSTM(M, return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
r = model.fit(data_train, y_train, epochs=10, validation_data=(data_test, y_test))

fig, ax = plt.subplots(nrows=2)
ax[0].plot(r.history['loss'], label='loss')
ax[0].plot(r.history['val_loss'], label='val_loss')

ax[1].plot(r.history['accuracy'], label='accuracy')
ax[1].plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
