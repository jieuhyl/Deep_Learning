# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:34:46 2019

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
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
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

COMMENT_TEXT_COL = 'Sentence'
EMB_MAX_FEAT = 300
MAX_LEN = 220
MAX_FEATURES = 100000
BATCH_SIZE = 32
NUM_EPOCHS = 3
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 512
EMB_PATH = "C:/Users/Jie.Hu/Desktop/Data Science/Practice/nlp/glove.840B.300d.txt"
DATA_PATH = "C:/Users/Jie.Hu/Desktop/Data Science/Practice/nlp/"


def get_logger():
    '''
        credits to: https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480
    '''
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
    
logger = get_logger()

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path, encoding="utf8") as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_embedding_matrix(word_index, path):
    '''
     credits to: https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold
    '''
    logger.info('Build embedding matrix')
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, EMB_MAX_FEAT))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
        except:
            embedding_matrix[i] = embedding_index["unknown"]
            
    del embedding_index
    gc.collect()
    return embedding_matrix


def load_data():
    logger.info('Load train and test data')
    df = pd.read_csv(os.path.join(DATA_PATH,'df_nlp.csv'))
    train, test = train_test_split(df, test_size=0.3, random_state=1337)
    return train, test


def perform_preprocessing(train, test):
    '''
        credits to: https://www.kaggle.com/artgor/cnn-in-keras-on-folds
        credits to: https://www.kaggle.com/taindow/simple-cudnngru-python-keras
    '''
    logger.info('data preprocessing')
    punct_mapping = {"_":" ", "`":" "}
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct += '©^®` <→°€™› ♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√'
    def clean_special_chars(text, punct, mapping):
        for p in mapping:
            text = text.replace(p, mapping[p])    
        for p in punct:
            text = text.replace(p, f' {p} ')     
        return text

    for df in [train, test]:
        df[COMMENT_TEXT_COL] = df[COMMENT_TEXT_COL].astype(str)
        df[COMMENT_TEXT_COL] = df[COMMENT_TEXT_COL].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
    
    return train, test


def run_proc_and_tokenizer(train, test):
    '''
        credits go to: https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution/ 
    '''
  
    logger.info('Fitting tokenizer')
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(list(train[COMMENT_TEXT_COL]) + list(test[COMMENT_TEXT_COL]))
    word_index = tokenizer.word_index
    X_train = tokenizer.texts_to_sequences(list(train[COMMENT_TEXT_COL]))
    X_test = tokenizer.texts_to_sequences(list(test[COMMENT_TEXT_COL]))
    X_train = pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = pad_sequences(X_test, maxlen=MAX_LEN)
    y_train = train['Sentiment'].values
    y_test = test['Sentiment'].values
    
    return X_train, y_train, X_test, y_test, word_index



def build_model(word_index, embedding_matrix):
    
    logger.info('Build model')
    embedding_layer = Embedding(len(word_index) + 1,
                            EMB_MAX_FEAT,
                            weights=[embedding_matrix],
                            input_length=MAX_LEN,
                            trainable=False)
    sequence_input = Input(shape=(MAX_LEN,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(2)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model

def run_model(X_train, y_train, X_test, y_test, model):
    '''
        credits to: https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution/
    '''
    logger.info('Run model')

    model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = y_pred.flatten() 
    return print("Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred)*100))


def submit(sub_preds):
    logger.info('Prepare submission')
    submission = pd.read_csv(os.path.join(DATA_PATH,'sample_submission.csv'), index_col='id')
    submission['prediction'] = sub_preds
    submission.reset_index(drop=False, inplace=True)
    submission.to_csv('submission.csv', index=False)

def main():
    train, test = load_data()
    train, test = perform_preprocessing(train, test)
    X_train, y_train, X_test, y_test, word_index = run_proc_and_tokenizer(train, test)
    embedding_matrix = build_embedding_matrix(word_index, EMB_PATH)
    model = build_model(word_index, embedding_matrix)
    run_model(X_train, y_train, X_test, y_test, model)
#    submit(sub_preds)
    
if __name__ == "__main__":
    main()
    




