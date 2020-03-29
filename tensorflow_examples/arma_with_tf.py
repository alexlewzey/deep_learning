"""making an ARMA model with tensorflow and showing its short comings ie more complicated models actually perform
worse"""
from typing import Sequence, List, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import plot

from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPool2D, Input
from tensorflow.keras.models import Model

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def make_ts_features(series: Sequence, T: int) -> Tuple[np.array, np.array]:
    """
    split a series into ~ N / T test train splits. example of one of the splits [t0, t1, ... tT-1] [tT]
    Args:
        series:
        T: number of time steps we are using to make predictions

    Returns:

    """
    if T > len(series):
        raise ValueError('Window length cannot be longer than series...')
    X: List = []
    y: List = []
    nsplits = len(series) - T
    for i in range(nsplits):
        X.append(series[i:i + T])
        y.append(series[i + T: i + T + 1])
    X = np.array(X).reshape(-1, T)
    y = np.array(y).reshape(-1, 1)
    print(f'X shape: {X}, y shape: {y}')
    return X, y


def train_test_split_ts(X: np.array, y: np.array, pct_train=0.7):
    """split time series X and y set into corresponding training and test sets"""
    n_windows = int(len(X) * pct_train)
    X_train = X[:n_windows]
    y_train = y[:n_windows]
    X_test = X[n_windows:]
    y_test = y[n_windows:]
    return X_train, X_test, y_train, y_test


def predict_ts(last_x, predictor: Callable) -> pd.array:
    predictions: List = []
    while len(predictions) < len(y_test):
        pred = predictor(last_x.reshape(1, -1))[0, 0]
        predictions.append(pred)

        last_x = np.roll(last_x, -1)
        last_x[-1] = pred
    return predictions


noise = 5
x = np.linspace(0, 50, 300)
y = 9 * np.cos(x) + 2.5 * x + np.random.randint(-noise, noise, x.shape)
T: int = 5
# plt.plot(x, y)
# plt.show()

X, y = make_ts_features(y, T)
X_train, X_test, y_train, y_test = train_test_split_ts(X, y, pct_train=0.5)

i = Input(shape=(T,))
x = Dense(1)(i)

model = Model(i, x)

model.compile(optimizer='adam', loss='mse')
r = model.fit(X_train, y_train, epochs=80, validation_data=(X_test, y_test))

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


predictions = predict_ts(X[-1], model.predict)

fig, ax = plt.subplots(ncols=2)
ax[0].plot(y_train)
ax[1].plot(predictions, label='preds')
ax[1].plot(y_test.ravel(), label='y_test')
plt.legend()
plt.show()
