# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:25:45 2020

@author: Jie.Hu
"""


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print("Current Working Directory " , os.getcwd())
os.chdir(r'C:\Users\Jie.Hu\Desktop\NLP_Python')


df = pd.read_csv('moviereviews.tsv', sep='\t')
df.head()

df.isnull().sum()
df.dropna(inplace=True)


blanks = []  # start with an empty list
for i,lb,rv in df.itertuples():  # iterate over the DataFrame
    if type(rv)==str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list
        
print(len(blanks), 'blanks: ', blanks)


df.drop(blanks, inplace=True)
df.shape
df['label'].value_counts()



# mtx
X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1337)

# CountVectorizer =============================================================
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()

X_train = count.fit_transform(X_train)
X_train.shape

from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train,y_train)

X_test = count.transform(X_test)
predictions = clf.predict(X_test)

from sklearn import metrics
print(metrics.confusion_matrix(y_test, predictions))

# Print a classification report
print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))


# TfidfTransformer ============================================================
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

X_train = tfidf.fit_transform(X_train)
X_train.shape

from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train,y_train)

X_test = tfidf.transform(X_test)
predictions = clf.predict(X_test)

from sklearn import metrics
print(metrics.confusion_matrix(y_test, predictions))

# Print a classification report
print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))


