import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import plot

from tensorflow.keras.layers import Input, Dense, Flatten, MaxPool2D, Dropout, SimpleRNN, Conv2D, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from my_helpers import mltk

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

noise = 5
x = np.linspace(0, 20 * np.pi, 800)
y_org = 9 * np.cos(x) + 2.5 * x + np.random.randint(-noise, noise, x.shape)
y_org = np.sin(x)
T: int = 100
# plt.plot(x, y_org)
# plt.show()
D = 1
M = 1

X, y = mltk.make_ts_features(y_org, Tx=T, Ty=100)
X = X.reshape(-1, T, 1)
X_train, X_test, y_train, y_test = mltk.train_test_split_ts(X, y, pct_train=0.5)


i = Input(shape=(T, 1))
x = LSTM(100)(i)
model = Model(i, x)

model.compile(optimizer=Adam(lr=0.01), loss='mse')
r = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test))

# plt.plot(r.history['loss'], label='loss')
# plt.plot(r.history['val_loss'], label='val_loss')
# plt.legend()
# plt.show()

predictions = mltk.predict_ts(last_x=X_test[0], predictor=model.predict, y_test=y_test)

plt.plot(predictions, label='preds')
plt.plot([arr[0] for arr in y_test], label='y_test')
plt.legend()
plt.show()
