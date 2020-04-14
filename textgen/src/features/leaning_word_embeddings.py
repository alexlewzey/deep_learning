import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from plotly.subplots import make_subplots
from tensorflow.keras import layers, utils, losses
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

import tensorflow_datasets as tfds

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)

train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes=([None], []))
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes=([None], []))
encoder = info.features['text'].encoder

features_example = next(iter(train_batches))
sequence_length = features_example[0].shape

embedding_dim = 16

i = layers.Input(shape=(None,))
x = layers.Embedding(encoder.vocab_size, embedding_dim)(i)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(1)(x)

model = Model(i, x)

model.compile(optimizer='adam',
              loss=losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

r = model.fit(train_batches, epochs=10, validation_data=test_batches, validation_steps=20)

e = model.layers[1]
