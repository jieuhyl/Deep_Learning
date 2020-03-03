# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:17:07 2019

@author: Jie.Hu
"""

import numpy as np 
import pandas as pd 
import sys
import os
import gc
import logging
import datetime
import warnings
import pickle
import operator
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import GRU, LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.losses import binary_crossentropy
from keras import backend as K
import keras.layers as L
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.layers import LSTM,Flatten
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import Callback
import matplotlib.pyplot as plt


COMMENT_TEXT_COL = 'comment_text'
EMBED_SIZE = 300
MAX_LEN = 200
MAX_FEATURES = 50000
#EMB_PATH = "C:/Users/Jie.Hu/Desktop/Data Science/Practice/nlp/glove.840B.300d.txt"
#DATA_PATH = "C:/Users/Jie.Hu/Desktop/Data Science/Practice/nlp/"


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('C:/Users/Jie.Hu/Desktop/Data Science/Practice/Kaggle_nlp_1/train.csv').fillna(' ')
test = pd.read_csv('C:/Users/Jie.Hu/Desktop/Data Science/Practice/Kaggle_nlp_1/test.csv').fillna(' ')

df = pd.concat([train, test])
y = train[class_names].values
X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size = 0.2, random_state=1337)

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
            
            
def build_vocabulary(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab
vocabulary = build_vocabulary(train['comment_text'])
# check first 10
print({k: vocabulary[k] for k in list(vocabulary)[:10]})
    

def load_embeddings(file):

    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
    return embeddings_index

GLOVE_PATH = 'C:/Users/Jie.Hu/Desktop/Data Science/Practice/nlp/'
print("Extracting GloVe embedding started")
embed_glove = load_embeddings(os.path.join(GLOVE_PATH,'glove.840B.300d.txt'))
print("Embedding completed")
len(embed_glove)


def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
    return unknown_words
print("Verify the intial vocabulary coverage")
oov_glove = check_coverage(vocabulary, embed_glove)



tokenizer = Tokenizer(num_words = MAX_FEATURES, lower = True) 
#tokenizer.fit_on_texts(train[COMMENT_TEXT_COL])
tokenizer.fit_on_texts(list(train[COMMENT_TEXT_COL]) + list(test[COMMENT_TEXT_COL]))
word_index = tokenizer.word_index
X_train = tokenizer.texts_to_sequences(X_train[COMMENT_TEXT_COL])
X_valid= tokenizer.texts_to_sequences(X_valid[COMMENT_TEXT_COL])
X_test= tokenizer.texts_to_sequences(test[COMMENT_TEXT_COL])

X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_valid= pad_sequences(X_valid, maxlen=MAX_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_LEN)

all_embs = np.stack(embed_glove.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

nb_words = min(MAX_FEATURES, len(word_index))
#embedding_matrix = np.zeros((nb_words, EMBED_SIZE))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBED_SIZE))
for word, i in word_index.items():
    if i >= MAX_FEATURES: continue
    embedding_vector = embed_glove.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector    
        

file_path = "best_model.hdf5"
ra_val = RocAucEvaluation(validation_data=(X_valid, y_valid), interval = 1)
check_point = ModelCheckpoint(file_path, monitor = "val_loss", mode = "min",
                              save_best_only = True, verbose = 1)
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 1)


def build_model(lr = 0.0, lr_d = 0.0, dr = 0.0):
    inp = Input(shape = (MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights = [embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(dr)(x)

    x = Bidirectional(GRU(128, return_sequences = True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    x = concatenate([avg_pool, max_pool])

    x = Dense(6, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(X_train, y_train, batch_size = 256, epochs = 5, validation_data = (X_valid, y_valid), 
                        verbose = 1, callbacks = [ra_val, check_point, early_stop])
    model = load_model(file_path)
    return model, history

model, history = build_model(lr = 0.001, lr_d = 0.0001, dr = 0.1)

y_pred = model.predict(X_test, batch_size = 1024, verbose = 1)

submission = pd.read_csv("C:/Users/Jie.Hu/Desktop/Data Science/Practice/Kaggle_nlp_1/sample_submission.csv")
submission[class_names] = y_pred
submission.to_csv("submission8.csv", index = False)


''' ensemble '''
# repeated evaluation
pred = 0
n_seeds = 2
for i in range(n_seeds):
    model, _ = build_model(lr = 0.001, lr_d = 0.0001, dr = 0.1)
    pred += model.predict(X_test, batch_size = 1024, verbose = 1)/n_seeds
            
        