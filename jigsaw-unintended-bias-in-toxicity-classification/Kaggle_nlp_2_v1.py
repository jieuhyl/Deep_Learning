# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:00:52 2019

@author: Jie.Hu
"""


# !pip install gensim  for the jupyter notebook ec2 installation
import gc
import re
import operator 

import numpy as np
import pandas as pd

from gensim.models import KeyedVectors

from sklearn import model_selection

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, Dense, GRU,concatenate, Bidirectional, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.callbacks import Callback
from keras.models import Model, load_model
import seaborn as sns
from sklearn.metrics import roc_auc_score


# load data
train = pd.read_csv('C:/Users/Jie.Hu/Desktop/Data Science/Practice/Kaggle_nlp_2/train.csv')
test = pd.read_csv('C:/Users/Jie.Hu/Desktop/Data Science/Practice/Kaggle_nlp_2/test.csv')

print("Train shape : ",train.shape)
print("Test shape : ",test.shape)

df = pd.concat([train[['id','comment_text']], test], axis=0)

# get embedding index
ft_common_crawl = 'C:/Users/Jie.Hu/Desktop/Data Science/Practice/nlp/crawl-300d-2M.vec'
embeddings_index = KeyedVectors.load_word2vec_format(ft_common_crawl)

# build vocab and check convergence
def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

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

df['comment_text'] = df['comment_text'].apply(lambda x: x.lower())

vocab = build_vocab(df['comment_text'])
oov = check_coverage(vocab, embeddings_index)

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

df['comment_text'] = df['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

vocab = build_vocab(df['comment_text'])
oov = check_coverage(vocab, embeddings_index)

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

punct_mapping = {"_":" ", "`":" "}

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])    
    for p in punct:
        text = text.replace(p, f' {p} ')     
    return text

df['comment_text'] = df['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

vocab = build_vocab(df['comment_text'])
oov = check_coverage(vocab, embeddings_index)


train = df.iloc[:train.shape[0],:]
test = df.iloc[train.shape[0]:,:]

train_orig = pd.read_csv("C:/Users/Jie.Hu/Desktop/Data Science/Practice/Kaggle_nlp_2/train.csv")

train = pd.concat([train,train_orig[['target']]],axis=1)
del(train_orig)
gc.collect()

train_df, validate_df = model_selection.train_test_split(train, test_size=0.1)
print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))


MAX_NUM_WORDS = 50000
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'

# Create a text tokenizer.
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(list(train_df[TEXT_COLUMN]) + list(validate_df[TEXT_COLUMN]))

# All comments must be truncated or padded to be the same length.
MAX_SEQUENCE_LENGTH = 200
def pad_text(texts, tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)

EMBEDDINGS_DIMENSION = 300
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,EMBEDDINGS_DIMENSION))

num_words_in_embedding = 0
for word, i in tokenizer.word_index.items():
    if word in embeddings_index.vocab:
        embedding_vector = embeddings_index[word]
        embedding_matrix[i] = embedding_vector        
        num_words_in_embedding += 1
        
train_text = pad_text(train_df[TEXT_COLUMN], tokenizer)
train_labels = train_df[TOXICITY_COLUMN]
validate_text = pad_text(validate_df[TEXT_COLUMN], tokenizer)
validate_labels = validate_df[TOXICITY_COLUMN]        

'''
MAX_FEATURES = 50000
EMBED_SIZE = 300
MAX_LEN = 200

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
'''

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
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 2)


def build_model(lr = 0.0, lr_d = 0.0, dr = 0.0):
    inp = Input(shape = (MAX_SEQUENCE_LENGTH,))
    x = Embedding(*embedding_matrix.shape, weights = [embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(dr)(x)

    x = Bidirectional(GRU(128, return_sequences = True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    x = concatenate([avg_pool, max_pool])

    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(train_text, train_labels, batch_size = 256, epochs = 4, validation_data=(validate_text, validate_labels), 
                        verbose = 1, callbacks = [ra_val, check_point, early_stop])
    model = load_model(file_path)
    return model, history

model, history = build_model(lr = 0.001, lr_d = 0.0001, dr = 0.1)