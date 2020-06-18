# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 02:08:47 2020

@author: Jie.Hu
"""



import pandas as pd
npr = pd.read_csv('npr.csv')
# Randomly sample 10% of your dataframe
#npr = npr.sample(frac=0.1)
# Randomly sample 100 elements from your dataframe
npr = npr.sample(n=1000)

npr.head()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(npr['Article'])


from sklearn.decomposition import NMF
nmf_model = NMF(n_components=7,random_state=42)
nmf_model.fit(dtm)

len(tfidf.get_feature_names())
len(nmf_model.components_)

single_topic = nmf_model.components_[0]
# Returns the indices that would sort this array.
single_topic.argsort()
single_topic.argsort()[0]
single_topic.argsort()[-1]
# Top 10 words for this topic:
single_topic.argsort()[-10:]

top_word_indices = single_topic.argsort()[-10:]
for index in top_word_indices:
    print(tfidf.get_feature_names()[index])
    
    
for index,topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


topic_results = nmf_model.transform(dtm)
topic_results.shape
topic_results[0]
topic_results[0].round(2)
topic_results[0].argmax()


npr.head()
topic_results.argmax(axis=1)
npr['Topic'] = topic_results.argmax(axis=1)
topic_dct = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G'}
npr['Label'] = npr['Topic'].map(topic_dct)
npr.head(10)