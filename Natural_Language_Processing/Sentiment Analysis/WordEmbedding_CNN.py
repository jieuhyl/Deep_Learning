# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 01:41:24 2020

@author: Jie.Hu
"""

# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import gc
import re
import operator 
import math
import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, Dense, add, CuDNNLSTM, CuDNNGRU, Dropout, MaxPooling1D, concatenate, Bidirectional, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback



# %% [code]

# load data
train = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
test = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

print("Train shape : ",train.shape)
print("Test shape : ",test.shape)

df = pd.concat([train[['id','comment_text']], test], axis=0)

df['comment_text'] = df['comment_text'].apply(lambda x: x.lower())

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

df['comment_text'] = df['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"_":" ", "`":" "}

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])    
    for p in punct:
        text = text.replace(p, f' {p} ')     
    return text

df['comment_text'] = df['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

train = df.iloc[:train.shape[0],:]
test = df.iloc[train.shape[0]:,:]

train_orig = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")

train = pd.concat([train,train_orig[['target']]],axis=1)
#del(train_orig)
#gc.collect()

train['target'] = np.where(train['target'] >= 0.5, True, False)
train_df, validate_df = model_selection.train_test_split(train, test_size=0.1)
print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))

MAX_FEATURES = 50000
EMBED_SIZE = 300
MAX_LEN = 220
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'

tokenizer = Tokenizer(num_words=MAX_FEATURES)
#tokenizer.fit_on_texts(train[COMMENT_TEXT_COL])
tokenizer.fit_on_texts(list(train_df[TEXT_COLUMN]) + list(validate_df[TEXT_COLUMN]))
word_index = tokenizer.word_index
train_text = tokenizer.texts_to_sequences(train_df[TEXT_COLUMN])
train_labels = train_df[TOXICITY_COLUMN]
validate_text = tokenizer.texts_to_sequences(validate_df[TEXT_COLUMN])
validate_labels = validate_df[TOXICITY_COLUMN]

train_text = pad_sequences(train_text, maxlen=MAX_LEN)
validate_text = pad_sequences(validate_text, maxlen=MAX_LEN)

test_text = tokenizer.texts_to_sequences(test[TEXT_COLUMN])
test_text = pad_sequences(test_text, maxlen=MAX_LEN)

del([df, train, test, train_orig])
gc.collect()


EMBED_PATHS = [ '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
                '../input/glove840b300dtxt/glove.840B.300d.txt']

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)

def build_embedding_matrix(word_index, path):

    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, EMBED_SIZE))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
        except:
            embedding_matrix[i] = embeddings_index["unknown"]
            
    del embedding_index
    gc.collect()
    return embedding_matrix

def build_embeddings(word_index):

    embedding_matrix = np.concatenate(
        [build_embedding_matrix(word_index, f) for f in EMBED_PATHS], axis=-1) 
    return embedding_matrix

embedding_matrix = build_embeddings(word_index)

# %% [code]
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

file_path = "best_model.hdf5"
ra_val = RocAucEvaluation(validation_data=(validate_text, validate_labels), interval = 1)
check_point = ModelCheckpoint(file_path, monitor = "val_loss", mode = "min",
                              save_best_only = True, verbose = 1)
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 1)

# define the network
def build_model(lr = 0.0, lr_d = 0.0, dr = 0.0):
    inp = Input(shape = (MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights = [embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(dr)(x)
    
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(5, padding='same')(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(3, padding='same')(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    x = concatenate([avg_pool, max_pool])
    x = Dropout(dr)(Dense(128, activation='relu') (x))
    x = Dropout(dr)(Dense(64, activation='relu') (x))
    result = Dense(1, activation="sigmoid")(x)
    
    
    model = Model(inputs = inp, outputs = result)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(train_text, train_labels, batch_size = 256, epochs = 5, validation_data=(validate_text, validate_labels), 
                        verbose = 1, callbacks = [ra_val, check_point, early_stop])
    model = load_model(file_path)
    return model, history

# run network
pred = 0
n_seeds = 1
for i in range(n_seeds):
    model, _ = build_model(lr = 0.01, lr_d = 0.001, dr = 0.2)
    pred += model.predict(test_text, batch_size = 1024, verbose = 1)/n_seeds

submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission['prediction'] = pred
submission.reset_index(drop=False, inplace=True)
submission.to_csv('submission.csv', index=False)