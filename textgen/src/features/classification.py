"""tabular dataset examples with tensorflow"""
from typing import List

import pandas as pd
import tensorflow as tf
from my_helpers import mltk
from plotly.offline import plot
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers, losses, callbacks
from tensorflow.keras.models import Model

from src.definitions import Models

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def df_to_dataset(df: pd.DataFrame, target: str, shuffle: bool = True, batch_size: int = 32):
    target = df.pop(target)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), target))
    if shuffle:
        ds = ds.shuffle(buffer_size=df.shape[0])
    ds = ds.batch(batch_size)
    return ds


URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
df = pd.read_csv(URL)

train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

bsize = 100
train_ds = df_to_dataset(train, target='target', batch_size=bsize)
val_ds = df_to_dataset(val, target='target', shuffle=False, batch_size=bsize)
test_ds = df_to_dataset(test, target='target', shuffle=False, batch_size=bsize)

feature_columns: List = []

cols_numeric = [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
]

i = {}

for col in cols_numeric:
    feature_columns.append(feature_column.numeric_column(col))
    i[col] = tf.keras.Input(shape=(1,), name=col)

thal = feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(
    thal_one_hot
)
i['thal'] = tf.keras.Input(shape=(1,), name='thal', dtype=tf.string)

x = layers.DenseFeatures(feature_columns)(i)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(1)(x)

model = Model(i, x)
model.compile(
    optimizer='adam',
    loss=losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'],
)
checkpoints = callbacks.ModelCheckpoint(Models.CKPT_TABULAR, save_weights_only=True)
r = model.fit(train_ds, epochs=31, validation_data=val_ds)
fig = mltk.plot_tf_history(r.history, keys=('loss', 'accuracy'), validation=True)
plot(fig)

