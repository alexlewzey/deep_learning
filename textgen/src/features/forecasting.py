import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from my_helpers import mltk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, utils, losses, optimizers, callbacks, preprocessing
from tensorflow.keras.models import Model

from src.definitions import Processed, Models

Models.CKPT_DIR_FORECAST.mkdir(exist_ok=True)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
df['Date Time'] = pd.to_datetime(df['Date Time'])
df = df.set_index('Date Time')
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df.values)

uni_data = df['T (degC)']
uni_data = uni_data[uni_data.index < datetime(2010, 1, 1)]

OBVS_YEAR = 52557

fig = px.line(uni_data.reset_index().sort_values('Date Time'), x='Date Time', y='T (degC)')
fig.update_xaxes(rangeslider_visible=True)
plot(fig)

TRAIN_SPLIT = 300_000
Tx = 100
Ty = 1

x_train, y_train = mltk.make_ts_features(uni_data[:TRAIN_SPLIT].values, Tx=Tx, Ty=Ty)
x_test, y_test = mltk.make_ts_features(uni_data[TRAIN_SPLIT:].values, Tx=Tx, Ty=Ty)

BATCH_SIZE: int = 256
BUFFER_SIZE: int = 10_000

train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val = val.cache().batch(BATCH_SIZE).repeat()

i = layers.Input(shape=x_train.shape[-2:])
x = layers.LSTM(32, return_sequences=True)(i)
x = layers.LSTM(16, activation='relu')(x)
x = layers.Dense(Ty)(x)

model = Model(i, x)
model.compile(optimizer='adam', loss='mae')

EVAL_INTERVAL = 200
EPOCHS = 10

checkpoints = callbacks.ModelCheckpoint(Models.CKPT_FORECAST, save_weights_only=True)
r = model.fit(train, epochs=EPOCHS, validation_data=val, steps_per_epoch=EVAL_INTERVAL, validation_steps=50,
              callbacks=[checkpoints])

fig = mltk.plot_tf_history(r.history, validation=True)

for x, y in train.take(1):
    break

f = model.predict(x)[-1]
plt.plot(f, label='x')
plt.plot(y[-1], label='y')
plt.legend()
plt.show()
