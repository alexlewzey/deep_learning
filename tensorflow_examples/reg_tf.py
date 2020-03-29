import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import plot
from my_helpers import dataviz
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow_core.python.keras import Input

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

X = np.random.random((1000, 2)) * 6 - 3
y = (np.cos(2 * X[:, 0]) + np.cos(3 * X[:, 1])).reshape(-1, 1)

comp = np.concatenate([X, y], axis=1)
df = pd.DataFrame(comp, columns=['x1', 'x2', 'y'])

# fig = px.scatter_3d(df, 'x1', 'x2', 'y')
# plot(fig)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(1)
])

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer='adam', loss='mse')
r = model.fit(X, y, epochs=500)

plt.plot(r.history['loss'])
plt.show()


predicted_surface = dataviz.make_predicted_surface(X[:, 0], X[:, 1], model.predict)
fig = dataviz.model_decisions_3d(df, 'x1', 'x2', 'y', predicted_surface=predicted_surface)
plot(fig)
