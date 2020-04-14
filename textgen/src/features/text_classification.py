import pandas as pd
from my_helpers import mltk
from plotly.offline import plot
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.models import Model

import tensorflow_datasets as tfds

from src.definitions import Models


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_examples, test_examples = dataset['train'], dataset['test']
encoder = info.features['text'].encoder

sample = 'The quick brow fox jumped over the lazy dog.'
sample_encoded = encoder.encode(sample)

for idx in sample_encoded:
    print(f'{idx} - {encoder.decode([idx])}')

BUFFER_SIZE = 10000
BATCH_SIZE = 64
EMBEDDING_DIMS = 64
EPOCHS = 1

train_dataset = (train_examples
                 .shuffle(BUFFER_SIZE)
                 .padded_batch(BATCH_SIZE, padded_shapes=([None], [])))
test_dataset = (test_examples
                .padded_batch(BATCH_SIZE, padded_shapes=([None], [])))

i = layers.Input(shape=(None,))
x = layers.Embedding(encoder.vocab_size, EMBEDDING_DIMS, mask_zero=True)(i)
x = layers.Bidirectional(layers.LSTM(units=32))(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(1)(x)

model = Model(i, x)

model.compile(
    optimizer=optimizers.Adam(0.0004),
    loss=losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

r = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, validation_steps=30)
fig = mltk.plot_tf_history(r.history, keys=('loss', 'accuracy'))
plot(fig)
test_loss, test_acc = model.evaluate(test_dataset)

model.save(Models.TEXT_CLASS)
