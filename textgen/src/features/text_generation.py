import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from my_helpers import mltk
from plotly.offline import plot
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras import layers, utils, losses, optimizers, callbacks, preprocessing
from tensorflow.keras.models import Model

from src.definitions import Models

SEQUENCE_LENGTH = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10_000
EMBEDDING_DIM = 256
EPOCHS = 10


def split_input_target(batch):
    input_text = batch[:-1]
    target_text = batch[1:]
    return input_text, target_text


def build_model(batch_size=BATCH_SIZE, embedding_dim=EMBEDDING_DIM) -> Model:
    i = layers.Input(batch_shape=[batch_size, None])
    x = layers.Embedding(vocab_size, embedding_dim)(i)
    x = layers.GRU(1024, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')(x)
    x = layers.Dense(vocab_size)(x)
    return Model(i, x)


def train_model():
    model = build_model()
    model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True))
    checkpoints_callback = callbacks.ModelCheckpoint(filepath=str(Models.CHECKPOINT_TEXT_SEQ), save_weights_only=True)
    r = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoints_callback])
    mltk.plot_tf_history(r.history)
    return model


def load_model():
    model = build_model(batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(str(Models.CKPT_DIR_TEXT_SEQ)))
    model.build(tf.TensorShape([1, None]))
    return model


def generate_text(model, start_string=u'ROMEO:', num_generate=1000) -> str:
    text_generated = list(start_string)
    input_eval = [char2idx[c] for c in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    model.reset_states()
    for i in range(num_generate):
        pred = model(input_eval)
        pred = tf.squeeze(pred, 0)
        pred_id = tf.random.categorical(pred, num_samples=1)
        pred_id = pred_id[-1, 0].numpy()

        text_generated.append(idx2char[pred_id])

        input_eval = tf.expand_dims([pred_id], 0)
    return ''.join(text_generated)


def make_features():
    text_as_int = np.array([char2idx[c] for c in text])
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(SEQUENCE_LENGTH + 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    return dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def load_data() -> str:
    path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                           'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    with open(path_to_file, 'rb') as f:
        text = f.read().decode('utf-8')
    return text


text = load_data()

vocab = sorted(set(text))
vocab_size = len(vocab)
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = np.array(vocab)

dataset = make_features()

# model = train_model()

model_loaded = load_model()

prose = generate_text(model_loaded)
print(prose)
