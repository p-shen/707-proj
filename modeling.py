
# coding: utf-8

'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model.

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
'''

from __future__ import print_function

import os
import sys
import numpy as np
import json
from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, GaussianNoise, regularizers
from keras.optimizers import Adam
from keras.models import Model
from keras.models import load_model
import datetime
import time

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H')

BASE_DIR = './'
GLOVE_EMBEDDING = '/home/pzs2/702/702-medication-project/glove.6B/glove.6B.100d.txt'
SAVE_DIR = BASE_DIR + 'models/' + st + '/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
EPOCHS = 50
BATCH_SIZE = 512
NUM_HIDDEN_UNITS = 256

data_x_file = "/home/pzs2/707/data/data_x_10k.txt"
data_y_cat = "/home/pzs2/707/data/data_y_10k_binary.txt"


# first, build index mapping words in the embeddings set
# to their embedding vector

embeddings_index = {}
with open(GLOVE_EMBEDDING) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


# read in the text files
texts_file = open(data_x_file, "r")
texts = texts_file.readlines()
texts_file.close()

labels = np.loadtxt(data_y_cat)


# prepare text samples and their labels
print('Processing text dataset')

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# TODO: we need to change this to a number not categorical
# labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
# labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
y_train.reshape(len(y_train), 1).shape


# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedded_sequences = embedding_layer(sequence_input)


print('Training model.')
tbCallBack = TensorBoard(log_dir='./Graph/{}/'.format(st),
                         histogram_freq=0, write_graph=True, write_images=True)

# train a 1D convnet with global maxpooling
x = Conv1D(NUM_HIDDEN_UNITS, 5, activation='relu')(embedded_sequences)
# x = GaussianNoise(0.1)(x)
x = MaxPooling1D(5)(x)
x = Conv1D(NUM_HIDDEN_UNITS, 5, activation='relu')(x)
x = Dropout(0.3)(x)
x = Conv1D(NUM_HIDDEN_UNITS, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(NUM_HIDDEN_UNITS, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(NUM_HIDDEN_UNITS, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
opt = Adam(lr=0.0001)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['acc'], )

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2,
          callbacks=[tbCallBack],
          verbose=1)

loss, acc = model.evaluate(x_val, y_val,
                           batch_size=512)

print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


# save the model output

import pickle

os.makedirs(SAVE_DIR)

model.save(SAVE_DIR + "model.h5")

# saving tokenizer
with open(SAVE_DIR + 'tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
