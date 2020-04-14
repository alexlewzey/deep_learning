from src import External
import pandas as pd
import plotly.express as px
from tensorflow.keras.layers import Embedding
import tensorflow as tf

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

lyrics = pd.read_csv(External.LYRICS55000)

print(lyrics.head())
print(lyrics.info())

#%%
# distribution of song count by artist
num_artists = lyrics['artist'].value_counts().to_frame()
fig = px.histogram(num_artists, 'artist', nbins=30, marginal='box')
# plot(fig)

# distribution of # of words by song
lyrics['tokens'] = lyrics['text'].str.split()
lyrics['nwords'] = lyrics['tokens'].str.len()
fig = px.histogram(lyrics, x='nwords', nbins=30, marginal='box')
# plot(fig)

# # of words in vocab
vocab = ' '.join(lyrics['text'].tolist()).split()
vocab = set(vocab)
len(vocab)

#%%

embedding_layer = Embedding(2, 5)
result = embedding_layer(tf.constant(1, 2, 3))

