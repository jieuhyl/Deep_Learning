# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:03:05 2019

try to fit GRU and BidirectionalLSTM
also try ensemble modeling

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


COMMENT_TEXT_COL = 'Sentence'
EMBED_SIZE = 300
MAX_LEN = 200
MAX_FEATURES = 10000
BATCH_SIZE = 32
NUM_EPOCHS = 3
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 512
#EMB_PATH = "C:/Users/Jie.Hu/Desktop/Data Science/Practice/nlp/glove.840B.300d.txt"
#DATA_PATH = "C:/Users/Jie.Hu/Desktop/Data Science/Practice/nlp/"


data_PATH = "C:/Users/Jie.Hu/Desktop/Data Science/Practice/nlp/"
df = pd.read_csv(os.path.join(data_PATH,'df_nlp.csv'), encoding='utf8')

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
vocabulary = build_vocabulary(df['Sentence'])
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


#lower case ================================
def add_lower(embedding, vocab):
    count = 0
    for word in vocab.keys():
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")
    
df['Sentence'] = df['Sentence'].apply(lambda x: x.lower())
 
print("Check coverage for vocabulary with lower case")
oov_glove = check_coverage(vocabulary, embed_glove)
add_lower(embed_glove, vocabulary) # operates on the same vocabulary
oov_glove = check_coverage(vocabulary, embed_glove)  


# contraction =====================================
contraction_mapping = {"writerdirector": "writer director", 
                       "computergenerated": "computer generated",
                       "ordell": "order",
                       "bowfinger": "bow finger",
                       "bmovie":"movie",
                       "ain't": "is not",
                       "isn't": "is not",
                       "go'church": "go church",
                       "teen'couples": "teen couples"}
len(contraction_mapping)

def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

df['Sentence'] = df['Sentence'].apply(lambda x: clean_contractions(x, contraction_mapping))

vocab = build_vocabulary(df['Sentence'])
print("Check embeddings after applying contraction mapping")
oov_glove = check_coverage(vocab, embed_glove)


# punctuation ============================
puncts = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
puncts += '©^®` <→°€™› ♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√'

punct_mapping = {"‘": "'", "´": "'", "°": "", "€": "e", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '…': ' '}

def clean_special_chars(text, punct, mapping):
    '''
    credits to: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings 
    credits to: https://www.kaggle.com/anebzt/quora-preprocessing-model
    input: current text, punctuations, punctuation mapping
    output: cleaned text
    '''
    for p in punct:
        text = text.replace(p, " ")
        text = ' '.join(text.split())
    for p in mapping:
        text = text.replace(p, mapping[p])
        return text

#train['Sentence'] = train['Sentence'].apply(lambda x: clean_special_chars(x, punct_mapping, puncts))
#test['Sentence'] = test['Sentence'].apply(lambda x: clean_special_chars(x, punct_mapping, puncts))
df['Sentence'] = df['Sentence'].apply(lambda x: clean_special_chars(x, puncts, punct_mapping))

vocab = build_vocabulary(df['Sentence'])
print("Check coverage after punctuation replacement")
oov_glove = check_coverage(vocab, embed_glove)

# split train and test
df['total_length'] = df[COMMENT_TEXT_COL].str.split(" ").str.len()
df.drop('total_length', axis = 1, inplace = True)

train, test = train_test_split(df, test_size=0.3, random_state=1337)
print("train and test shape: {} {}".format(train.shape, test.shape))

tokenizer = Tokenizer() 
tokenizer.fit_on_texts(list(train[COMMENT_TEXT_COL]) + list(test[COMMENT_TEXT_COL]))
word_index = tokenizer.word_index
X_train = tokenizer.texts_to_sequences(list(train[COMMENT_TEXT_COL]))
X_test = tokenizer.texts_to_sequences(list(test[COMMENT_TEXT_COL]))
X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_LEN)
y_train = train['Sentiment'].values
y_test = test['Sentiment'].values


def build_embedding_matrix(word_index, embedding_index):
    embedding_matrix = np.zeros((len(word_index)+1, EMBED_SIZE))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
        except:
            embedding_matrix[i] = embedding_index["unknown"]
            
    return embedding_matrix

embedding_matrix = build_embedding_matrix(word_index, embed_glove)


file_path = "best_model.hdf5"
ra_val = RocAucEvaluation(validation_data=(X_test, y_test), interval = 1)
check_point = ModelCheckpoint(file_path, monitor = "val_loss", mode = "min",
                              save_best_only = True, verbose = 1)
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)


def build_model(lr = 0.0, lr_d = 0.0, dr = 0.0):
    inp = Input(shape = (MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights = [embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(dr)(x)

    x = Bidirectional(GRU(32, return_sequences = True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    x = concatenate([avg_pool, max_pool])

    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(X_train, y_train, batch_size = 128, epochs = 100, validation_data = (X_test, y_test), 
                        verbose = 1, callbacks = [ra_val, check_point, early_stop])
    model = load_model(file_path)
    return model, history

model, history = build_model(lr = 0.001, lr_d = 0, dr = 0.1)

# plot loss learning curves
plt.subplot(211)
plt.title('Loss', pad=-40)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy learning curves
plt.subplot(212)
plt.title('Accuracy', pad=-40)
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()

# confusion matrix report
y_prob  = model.predict(X_test, verbose = 1) 
#y_pred = y_prob.argmax(axis=-1)
y_classes = np.array([1 if x>=0.5 else 0 for x in y_prob])
from sklearn.metrics import classification_report
target_names = ['Dislike', 'Like']
print(classification_report(y_test, y_classes, target_names = target_names))

''' ensemble '''
# repeated evaluation
pred = 0
n_seeds = 2
for i in range(n_seeds):
    model, _ = build_model(lr = 0.001, lr_d = 0, dr = 0.1)
    pred += model.predict(X_test, verbose = 1)/n_seeds
    