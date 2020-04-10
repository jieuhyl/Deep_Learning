# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 21:29:10 2019

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
from numpy import asarray
from tqdm import tqdm_notebook
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.losses import binary_crossentropy
from keras import backend as K
import keras.layers as L
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier


data_PATH = "C:/Users/Jie.Hu/Desktop/Data Science/Practice/nlp/"
df = pd.read_csv(os.path.join(data_PATH,'nlp_try1.csv'), encoding='windows-1252')


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

'''
def build_vocabulary(texts):

    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in tqdm_notebook(sentences):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

#df = pd.concat([train ,test], sort=False)
vocabulary = build_vocabulary(df['Sentence'])


def check_coverage(vocab, embeddings_index):

    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in tqdm_notebook(vocab.keys()):
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass
    print('Found embeddings for {:.3%} of vocabulary'.format(len(known_words)/len(vocab)))
    print('Found embeddings for {:.3%} of all text'.format(nb_known_words/(nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
    return unknown_words

print("Verify the intial vocabulary coverage")
oov_glove = check_coverage(vocabulary, embed_glove)
'''

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



'''word embedding'''
EMBED_SIZE = 300 # size of word vector; this should be set to 300 to match the embedding source
MAX_FEATURES = 20 # how many unique words to use (i.e num rows in embedding vector)
MAXLEN = 10 # max length of comments text

tokenizer = Tokenizer(nb_words=MAX_FEATURES)
tokenizer.fit_on_texts(df['Sentence'])
sequences = tokenizer.texts_to_sequences(df['Sentence'])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAXLEN)

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

# 1
def embedding_matrix(word_index, embeddings_index):

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    EMBED_SIZE = all_embs.shape[1]
    nb_words = min(MAX_FEATURES, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBED_SIZE))
    for word, i in tqdm_notebook(word_index.items()):
        if i >= MAX_FEATURES:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

embedding_matrix = embedding_matrix(word_index,  embed_glove)

# 2
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

# 3        
nb_words = min(MAX_FEATURES, len(word_index))
embedding_matrix = np.zeros((nb_words, EMBED_SIZE))
for word, i in word_index.items():
    if i >= MAX_FEATURES: continue
    embedding_vector = embed_glove.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector        
        
        
# get sentence embedding ======================================================
sentence_matrix = []        
for index, row in df.iterrows():
    sentence_matrix.append(len(row['Sentence'].split()))       
    
    
sentence_matrix1 = []  
for i, row in df.iterrows(): 
    sentence_matrix2 = []
    for j in row['Sentence'].split():
        sentence_matrix2.append(embed_glove[j])
    sentence_matrix1.append(np.average(sentence_matrix2, axis=0))
    
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(sentence_matrix1)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt  
pca = PCA(n_components=2)
result = pca.fit_transform(sentence_matrix1)
# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
words = df['Sentiment'].values.tolist()
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()