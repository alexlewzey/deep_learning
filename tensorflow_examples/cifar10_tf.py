import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from my_helpers import dataviz
from plotly.offline import plot
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, MaxPool2D, BatchNormalization, Flatten
from tensorflow_core.python.keras.datasets.cifar10 import load_data

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = x_train / 255, x_test / 255
y_train, y_test = y_train.flatten(), y_test.flatten()

K = len(set(y_train))

i = Input(x_train[0].shape)
x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(i)
x = BatchNormalization()(x)
x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(i)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)

x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)

x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2,2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

print(f'train accuracy: {model.evaluate(x_train, y_train)}')
print(f'test accuracy: {model.evaluate(x_test, y_test)}')

fig, ax = plt.subplots(nrows=2)
ax[0].plot(r.history['loss'], label='loss')
ax[0].plot(r.history['val_loss'], label='val_loss')

ax[1].plot(r.history['accuracy'], label='accuracy')
ax[1].plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

pred = model.predict(x_test).argmax(axis=1)

print(confusion_matrix(y_test, pred))
dataviz.heatmap(confusion_matrix(y_test, pred))
